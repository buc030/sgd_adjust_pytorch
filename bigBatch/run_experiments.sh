


CUDA_VISIBLE_DEVICES=0 python main_normal.py --dataset cifar10 --model resnet --save cifar10_resnet44_bs2048_baseline --epochs 100 --b 2048 --no-lr_bb_fix --disable_lr_change --iters_per_adjust 27 &
CUDA_VISIBLE_DEVICES=1 python main_normal.py --dataset cifar10 --model resnet --save cifar10_resnet44_bs2048_lr_fix --epochs 100 --b 2048 --lr_bb_fix --disable_lr_change --iters_per_adjust 27 &
CUDA_VISIBLE_DEVICES=2 python main_normal.py --dataset cifar10 --model resnet --save cifar10_resnet44_bs2048_regime_adaptation --epochs 100 --b 2048 --lr_bb_fix --regime_bb_fix --disable_lr_change --iters_per_adjust 27 &
CUDA_VISIBLE_DEVICES=3 python main_gbn.py --dataset cifar10 --model resnet --save cifar10_resnet44_bs2048_ghost_bn256 --epochs 100 --b 2048 --lr_bb_fix --mini-batch-size 256 --disable_lr_change --iters_per_adjust 27 &
CUDA_VISIBLE_DEVICES=4 python main_normal.py --dataset cifar100 --model resnet --save cifar100_wresnet16_4_bs1024_regime_adaptation --epochs 100 --b 1024 --lr_bb_fix --regime_bb_fix --disable_lr_change  --iters_per_adjust 55 &

wait

CUDA_VISIBLE_DEVICES=0 python main_normal.py --model mnist_f1 --dataset mnist --save mnist_baseline_bs2048_no_lr_fix --epochs 50 --b 2048 --no-lr_bb_fix --disable_lr_change  --iters_per_adjust 27 &
CUDA_VISIBLE_DEVICES=1 python main_normal.py --model mnist_f1 --dataset mnist --save mnist_baseline_bs2048 --epochs 50 --b 2048 --lr_bb_fix --disable_lr_change  --iters_per_adjust 27 &
CUDA_VISIBLE_DEVICES=2 python main_gbn.py --model mnist_f1 --dataset mnist --save mnist_baseline_bs4096_gbn --epochs 50 --b 4096 --lr_bb_fix --no-regime_bb_fix --mini-batch-size 128 --disable_lr_change --iters_per_adjust 14 &
CUDA_VISIBLE_DEVICES=3 python main_gbn.py --model cifar100_shallow --dataset cifar100 --save shallow_cifar100_baseline_bs4096_gbn --epochs 200 --b 4096 --lr_bb_fix --no-regime_bb_fix --mini-batch-size 128 --disable_lr_change  --iters_per_adjust 14 &
CUDA_VISIBLE_DEVICES=4 python main_gbn.py --model cifar10_shallow --dataset cifar10 --save shallow_cifar10_baseline_bs4096_gbn --epochs 200 --b 4096 --lr_bb_fix --no-regime_bb_fix --mini-batch-size 128 --disable_lr_change  --iters_per_adjust 14 &

wait


CUDA_VISIBLE_DEVICES=0 python main_normal.py --dataset cifar10 --model resnet --save cifar10_resnet44_bs2048_baseline_enable_lr_change --epochs 100 --b 2048 --no-lr_bb_fix --enable_lr_change  --iters_per_adjust 27 &
CUDA_VISIBLE_DEVICES=1 python main_normal.py --dataset cifar10 --model resnet --save cifar10_resnet44_bs2048_lr_fix_enable_lr_change --epochs 100 --b 2048 --lr_bb_fix --enable_lr_change  --iters_per_adjust 27 &
CUDA_VISIBLE_DEVICES=2 python main_normal.py --dataset cifar10 --model resnet --save cifar10_resnet44_bs2048_regime_adaptation_enable_lr_change --epochs 100 --b 2048 --lr_bb_fix --regime_bb_fix --enable_lr_change  --iters_per_adjust 27 &
CUDA_VISIBLE_DEVICES=3 python main_gbn.py --dataset cifar10 --model resnet --save cifar10_resnet44_bs2048_ghost_bn256_enable_lr_change --epochs 100 --b 2048 --lr_bb_fix --mini-batch-size 256 --enable_lr_change  --iters_per_adjust 27 &
CUDA_VISIBLE_DEVICES=4 python main_normal.py --dataset cifar100 --model resnet --save cifar100_wresnet16_4_bs1024_regime_adaptation_enable_lr_change --epochs 100 --b 1024 --lr_bb_fix --regime_bb_fix --enable_lr_change  --iters_per_adjust 55 &

wait

CUDA_VISIBLE_DEVICES=0 python main_normal.py --model mnist_f1 --dataset mnist --save mnist_baseline_bs2048_no_lr_fix_enable_lr_change --epochs 50 --b 2048 --no-lr_bb_fix --enable_lr_change  --iters_per_adjust 27 &
CUDA_VISIBLE_DEVICES=1 python main_normal.py --model mnist_f1 --dataset mnist --save mnist_baseline_bs2048_enable_lr_change --epochs 50 --b 2048 --lr_bb_fix --enable_lr_change  --iters_per_adjust 27 &
CUDA_VISIBLE_DEVICES=2 python main_gbn.py --model mnist_f1 --dataset mnist --save mnist_baseline_bs4096_gbn_enable_lr_change --epochs 50 --b 4096 --lr_bb_fix --no-regime_bb_fix --mini-batch-size 128 --enable_lr_change  --iters_per_adjust 14 &
CUDA_VISIBLE_DEVICES=3 python main_gbn.py --model cifar100_shallow --dataset cifar100 --save shallow_cifar100_baseline_bs4096_gbn_enable_lr_change --epochs 200 --b 4096 --lr_bb_fix --no-regime_bb_fix --mini-batch-size 128 --enable_lr_change  --iters_per_adjust 14 &
CUDA_VISIBLE_DEVICES=4 python main_gbn.py --model cifar10_shallow --dataset cifar10 --save shallow_cifar10_baseline_bs4096_gbn_enable_lr_change --epochs 200 --b 4096 --lr_bb_fix --no-regime_bb_fix --mini-batch-size 128 --enable_lr_change  --iters_per_adjust 14 &

