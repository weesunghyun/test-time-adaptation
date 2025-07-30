#!/bin/bash -l

conda activate tta

datasets=(
    "cifar10_c"
    "cifar100_c"
    "imagenet_c"
)

methods=(
    "ttn"
)

setting=reset_each_shift
batches=(128 64 32 16 8 4 2 1)
seeds=(1)
options=()

combinations=(
    "16 16"
    "8 8"
    "4 8"
    "4 4"
)

for dataset in "${datasets[@]}"; do
    for method in "${methods[@]}"; do
        for combo in "${combinations[@]}"; do
            read -r w_bit a_bit <<< "$combo"
            echo "w_bit:${w_bit}, a_bit:${a_bit}"

            for batch in ${batches[*]}; do
                for seed in ${seeds[*]}; do
                    if [ "$w_bit" -eq 16 ] && [ "$a_bit" -eq 16 ]; then
                        CUDA_VISIBLE_DEVICES=0 python test_time.py \
                            --cfg "cfgs/$dataset/$method.yaml" \
                            SETTING $setting \
                            RNG_SEED $seed DETERMINISM True \
                            TEST.BATCH_SIZE $batch \
                            CORRUPTION.SEVERITY 1,2,3,4,5 \
                            QUANT.QUANTIZE False \
                            SAVE_DIR "./exp_alpha/w${w_bit}a${a_bit}/"
                    else
                        CUDA_VISIBLE_DEVICES=0 python test_time.py \
                            --cfg "cfgs/$dataset/$method.yaml" \
                            SETTING $setting \
                            RNG_SEED $seed DETERMINISM True \
                            TEST.BATCH_SIZE $batch \
                            CORRUPTION.SEVERITY 1,2,3,4,5 \
                            QUANT.QUANTIZE True \
                            QUANT.WEIGHT_BITS $w_bit \
                            QUANT.ACTIVATION_BITS $a_bit \
                            SAVE_DIR "./exp_alpha/w${w_bit}a${a_bit}/"
                    fi
                done
            done
        done
    done
done
