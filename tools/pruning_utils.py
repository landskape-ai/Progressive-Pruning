import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from tools.layers import Conv2d, Linear

__all__ = [
    "masked_parameters",
    "SynFlow",
    "Mag",
    "Taylor1ScorerAbs",
    "Rand",
    "SNIP",
    "GraSP",
    "check_sparsity_dict",
    "extract_mask",
    "prune_conv",
]


def masks(module):
    r"""Returns an iterator over modules masks, yielding the mask."""
    for name, buf in module.named_buffers():
        if "mask" in name:
            yield buf


def masked_parameters(model):
    r"""Returns an iterator over models prunable parameters, yielding both the
    mask and parameter tensors.
    """
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            for mask, param in zip(masks(module), module.parameters(recurse=False)):
                if param is not module.bias:
                    yield mask, param


class Pruner:
    def __init__(self, masked_parameters):
        self.masked_parameters = list(masked_parameters)
        self.scores = {}

    def score(self, model, loss, dataloader, device):
        raise NotImplementedError

    def _global_mask(self, sparsity):
        r"""Updates masks of model with scores by sparsity level globally."""
        # # Set score for masked parameters to -inf
        # for mask, param in self.masked_parameters:
        #     score = self.scores[id(param)]
        #     score[mask == 0.0] = -np.inf

        # Threshold scores
        global_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        k = int((1.0 - sparsity) * global_scores.numel())
        if not k < 1:
            threshold, _ = torch.kthvalue(global_scores, k)
            for mask, param in self.masked_parameters:
                score = self.scores[id(param)]
                zero = torch.tensor([0.0]).to(mask.device)
                one = torch.tensor([1.0]).to(mask.device)
                mask.copy_(torch.where(score <= threshold, zero, one))

    def _local_mask(self, sparsity):
        r"""Updates masks of model with scores by sparsity level parameter-wise."""
        for mask, param in self.masked_parameters:
            score = self.scores[id(param)]
            k = int((1.0 - sparsity) * score.numel())
            if not k < 1:
                threshold, _ = torch.kthvalue(torch.flatten(score), k)
                zero = torch.tensor([0.0]).to(mask.device)
                one = torch.tensor([1.0]).to(mask.device)
                mask.copy_(torch.where(score <= threshold, zero, one))

    def mask(self, sparsity, scope):
        r"""Updates masks of model with scores by sparsity according to scope."""
        if scope == "global":
            self._global_mask(sparsity)
        if scope == "local":
            self._local_mask(sparsity)

    @torch.no_grad()
    def apply_mask(self):
        r"""Applies mask to prunable parameters."""
        for mask, param in self.masked_parameters:
            param.mul_(mask)

    def alpha_mask(self, alpha):
        r"""Set all masks to alpha in model."""
        for mask, _ in self.masked_parameters:
            mask.fill_(alpha)

    # Based on https://github.com/facebookresearch/open_lth/blob/master/utils/tensor_utils.py#L43
    def shuffle(self):
        for mask, param in self.masked_parameters:
            shape = mask.shape
            perm = torch.randperm(mask.nelement())
            mask = mask.reshape(-1)[perm].reshape(shape)

    def invert(self):
        for v in self.scores.values():
            v.div_(v ** 2)

    def stats(self):
        r"""Returns remaining and total number of prunable parameters."""
        remaining_params, total_params = 0, 0
        for mask, _ in self.masked_parameters:
            remaining_params += mask.detach().cpu().numpy().sum()
            total_params += mask.numel()
        return remaining_params, total_params


class SynFlow(Pruner):
    def __init__(self, masked_parameters):
        super(SynFlow, self).__init__(masked_parameters)

    def score(self, model, loss, dataloader, device):
        @torch.no_grad()
        def linearize(model):
            # model.double()
            signs = {}
            for name, param in model.state_dict().items():
                signs[name] = torch.sign(param)
                param.abs_()
            return signs

        @torch.no_grad()
        def nonlinearize(model, signs):
            # model.float()
            for name, param in model.state_dict().items():
                param.mul_(signs[name])

        signs = linearize(model)

        (data, _) = next(iter(dataloader))
        input_dim = list(data[0, :].shape)
        input = torch.ones([1] + input_dim).to(
            device
        )  # , dtype=torch.float64).to(device)
        output = model(input)
        torch.sum(output).backward()

        for _, p in self.masked_parameters:
            self.scores[id(p)] = torch.clone(p.grad * p).detach().abs_()
            p.grad.data.zero_()

        nonlinearize(model, signs)


class Mag(Pruner):
    def __init__(self, masked_parameters):
        super(Mag, self).__init__(masked_parameters)

    def score(self, model, loss, dataloader, device):
        for _, p in self.masked_parameters:
            self.scores[id(p)] = torch.clone(p.data).detach().abs_()


class Rand(Pruner):
    def __init__(self, masked_parameters):
        super(Rand, self).__init__(masked_parameters)

    def score(self, model, loss, dataloader, device):
        for _, p in self.masked_parameters:
            self.scores[id(p)] = torch.randn_like(p)


# Based on https://github.com/mi-lad/snip/blob/master/snip.py#L18
class SNIP(Pruner):
    def __init__(self, masked_parameters):
        super(SNIP, self).__init__(masked_parameters)

    def score(self, model, loss, dataloader, device):

        # allow masks to have gradient
        for m, _ in self.masked_parameters:
            m.requires_grad = True

        # compute gradient
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss(output, target).backward()

        # calculate score |g * theta|
        for m, p in self.masked_parameters:
            self.scores[id(p)] = torch.clone(m.grad).detach().abs_()
            p.grad.data.zero_()
            m.grad.data.zero_()
            m.requires_grad = False

        # normalize score
        all_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        norm = torch.sum(all_scores)
        for _, p in self.masked_parameters:
            self.scores[id(p)].div_(norm)


def GraSP_fetch_data(dataloader, num_classes, samples_per_class):
    datas = [[] for _ in range(num_classes)]
    labels = [[] for _ in range(num_classes)]
    mark = dict()
    dataloader_iter = iter(dataloader)
    while True:
        inputs, targets = next(dataloader_iter)
        for idx in range(inputs.shape[0]):
            x, y = inputs[idx : idx + 1], targets[idx : idx + 1]
            category = y.item()
            if len(datas[category]) == samples_per_class:
                mark[category] = True
                continue
            datas[category].append(x)
            labels[category].append(y)
        if len(mark) == num_classes:
            break

    X, y = torch.cat([torch.cat(_, 0) for _ in datas]), torch.cat(
        [torch.cat(_) for _ in labels]
    ).view(-1)
    return X, y


# Based on https://github.com/alecwangcq/GraSP/blob/master/pruner/GraSP.py#L49
class GraSP(Pruner):
    def __init__(self, masked_parameters):
        super(GraSP, self).__init__(masked_parameters)
        self.temp = 200
        self.eps = 1e-10

    def score(self, model, loss, dataloader, device):

        # first gradient vector without computational graph
        stopped_grads = 0

        data, target = GraSP_fetch_data(dataloader, 10, 10)
        data, target = data.to(device), target.to(device)
        output = model(data) / self.temp
        L = loss(output, target)

        grads = torch.autograd.grad(
            L, [p for (_, p) in self.masked_parameters], create_graph=False
        )
        flatten_grads = torch.cat([g.reshape(-1) for g in grads if g is not None])
        stopped_grads += flatten_grads

        # second gradient vector with computational graph

        data, target = GraSP_fetch_data(dataloader, 10, 10)
        data, target = data.to(device), target.to(device)
        output = model(data) / self.temp
        L = loss(output, target)

        grads = torch.autograd.grad(
            L, [p for (_, p) in self.masked_parameters], create_graph=True
        )
        flatten_grads = torch.cat([g.reshape(-1) for g in grads if g is not None])

        gnorm = (stopped_grads * flatten_grads).sum()
        gnorm.backward()

        # calculate score Hg * theta (negate to remove top percent)
        for _, p in self.masked_parameters:
            self.scores[id(p)] = torch.clone(p.grad * p.data).detach()
            p.grad.data.zero_()

        # normalize score
        all_scores = torch.cat([torch.flatten(v) for v in self.scores.values()])
        norm = torch.abs(torch.sum(all_scores)) + self.eps
        for _, p in self.masked_parameters:
            self.scores[id(p)].div_(norm)


class Taylor1ScorerAbs(Pruner):
    def __init__(self, masked_parameters):
        super(Taylor1ScorerAbs, self).__init__(masked_parameters)

    def score(self, model, loss, dataloader, device):

        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss(output, target).backward()

        for _, p in self.masked_parameters:
            self.scores[id(p)] = torch.clone(p.grad * p).detach().abs_()
            p.grad.data.zero_()


def check_sparsity_dict(model_dict):

    sum_list = 0
    zero_sum = 0

    for key in model_dict.keys():
        if "mask" in key:
            sum_list = sum_list + float(model_dict[key].nelement())
            zero_sum = zero_sum + float(torch.sum(model_dict[key] == 0))

    print("* remain weight = ", 100 * (1 - zero_sum / sum_list), "%")

    return 100 * (1 - zero_sum / sum_list)


def extract_mask(model_dict):

    new_dict = {}

    for key in model_dict.keys():
        if "mask" in key:
            new_dict[key] = copy.deepcopy(model_dict[key])

    return new_dict


def prune_conv(model):

    for name, module in reversed(model._modules.items()):

        if len(list(module.children())) > 0:
            model._modules[name] = prune_conv(model=module)

        if isinstance(module, nn.Conv2d):
            bias = True
            if module.bias == None:
                bias = False
            layer_new = Conv2d(
                module.in_channels,
                module.out_channels,
                module.kernel_size,
                module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=bias,
            )
            model._modules[name] = layer_new

    return model
