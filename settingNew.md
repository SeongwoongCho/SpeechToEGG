# mask_only/0

efficientnet b0
white noise 0,20
pos weight 3

python -m torch.distributed.launch --nproc 4 train-mask.py --batch_size 1024 --train_stride 2048 --valid_stride 4096 --n_sample 4096 --window_length 512 --hop_length 128 --window hann --epoch 1000 --weight_decay 1e-5 --lr 1e-2 --patience 50 --momentum 0.9 --exp_num 0 --ddp --mixed


# mag_only/0

efficientnet b0
white noise 0,20

python -m torch.distributed.launch --nproc 6 train-mag.py --batch_size 2048 --train_stride 2048 --valid_stride 4096 --n_sample 4096 --window_length 512 --hop_length 128 --window hann --epoch 1000 --weight_decay 1e-5 --lr 1e-2 --patience 50 --momentum 0.9 --exp_num 0 --ddp --mixed