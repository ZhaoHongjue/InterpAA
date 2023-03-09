for model_name in ResNet
do
    for dataset in CIFAR10
    do
        for batch_size in 128
        do
            for lr in 0.0001
            do
                for epochs in 15
                do
                    for seed in 0
                    do
                        for cuda in 2
                        do
                            nohup python train.py       \
                            --model_mode    $model_name \
                            --dataset       $dataset    \
                            --batch_size    $batch_size \
                            --lr            $lr         \
                            --epochs        $epochs     \
                            --seed          $seed       \
                            --cuda          $cuda       \
                            --use_gap                   \
                            --use_wandb > $model_name-$dataset-bs$batch_size-lr$lr-seed$seed.out 2>&1 &
                        done
                    done
                done
            done
        done
    done
done