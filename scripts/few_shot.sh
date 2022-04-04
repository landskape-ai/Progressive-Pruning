# Example for few shot experiments
n_shots = $1 

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
    --few_shot \
    --n_shots ${n_shots} \
    --save_dir few_shot_${n_shots}_r50