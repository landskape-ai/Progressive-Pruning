# Example for few shot experiments
n_shots = 540 # number of shots

python main_anytime_train.py \
    --data ../data \
    --dataset restricted_imagenet \
    --arch resnet50 \
    --seed 1 \
    --epochs 50 \
    --decreasing_lr 20,40 \
    --batch_size 32 \
    --weight_decay 1e-4 \
    --meta_batch_size 63 \
    --meta_batch_number 120 \
    --sparsity_level 4.5 \
    --snip_size 0.20 \
    --few_shot \
    --n_shots ${n_shots} \
    --save_dir few_shot_${n_shots}_r50