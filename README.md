# Progressive-Pruning



### Requirements

### Run the Code

Here is an example of running the Anytime Progressive Pruning on Cifar-10 dataset 8 Mega-Batches:
```
python main_anytime_train.py \
    --data ../data \
    --dataset cifar10 \
    --arch resnet50 \
    --seed 1 \
    --epochs 50 \
    --decreasing_lr 20,40 \
    --batch_size 64 \
    --weight_decay 1e-4 \
    --meta_batch_size 6250 \
    --meta_batch_number 8 \
    --sparsity_level 4.5 \
    --snip_size 0.20 \
    --save_dir c10_r50
```
One-Shot pruning :
```
python main_anytime_one.py \
    --data ../data \
    --dataset cifar10 \
    --arch resnet50 \
    --seed 1 \
    --epochs 50 \
    --decreasing_lr 20,40 \
    --batch_size 64 \
    --weight_decay 1e-4 \
    --meta_batch_size 6250 \
    --meta_batch_number 8 \
    --sparsity_level 4.5 \
    --snip_size 0.20 \
    --save_dir c10_OSP_r18
```
Baseline :
```
python main_anytime_baseline.py \
    --data ../data \
    --dataset cifar10 \
    --arch resnet50 \
    --seed 1 \
    --epochs 50 \
    --decreasing_lr 20,40 \
    --batch_size 64 \
    --weight_decay 1e-4 \
    --meta_batch_size 6250 \
    --meta_batch_number 8 \
    --save_dir c10_BASE_r50
```

For running the code on Restricted-Imagenet Dataset, first install the robustness library from [here](https://github.com/landskape-ai/Progressive-Pruning/tree/main/robustness) and provide the imagenet_path argument as the path to the imaganet data folder. 
