# mask_only/0

efficientnet b0
white noise 0,20
pos weight 3

python -m torch.distributed.launch --nproc 4 train-mask.py --batch_size 1024 --train_stride 2048 --valid_stride 4096 --n_sample 4096 --window_length 512 --hop_length 128 --window hann --epoch 1000 --weight_decay 1e-5 --lr 1e-2 --patience 50 --momentum 0.9 --exp_num 0 --ddp --mixed


# mag_only/0

efficientnet b0
white noise 0,20

python -m torch.distributed.launch --nproc 6 train-mag.py --batch_size 2700 --train_stride 2048 --valid_stride 4096 --n_sample 4096 --window_length 512 --hop_length 128 --window hann --epoch 1000 --weight_decay 1e-5 --lr 1e-2 --patience 50 --momentum 0.9 --exp_num 0 --ddp --mixed

하다가 중간에 끔 b4 확인하려고

# mag_only/1

efficientnet b4
white noise 0,20

python -m torch.distributed.launch --nproc 6 train-mag.py --batch_size 1200 --train_stride 2048 --valid_stride 4096 --n_sample 4096 --window_length 512 --hop_length 128 --window hann --epoch 1000 --weight_decay 1e-5 --lr 1e-2 --patience 35 --momentum 0.9 --exp_num 1 --ddp --mixed

# mag_only/2

best from TUNING

efficientnet b4
white noise 0,20

python -m torch.distributed.launch --nproc 6 train-mag.py --batch_size 1200 --train_stride 2048 --valid_stride 4096 --n_sample 4096 --window_length 512 --hop_length 128 --window hann --epoch 1000 --weight_decay 1e-5 --lr 0.072024 --patience 35 --momentum 0.96875 --exp_num 2 --ddp --mixed

# mag_only/3

2와 같은 세팅. but mag = L1_sum / torch.sum(mask) , increase patience

\efficientnet b4
white noise 0,20

python -m torch.distributed.launch --nproc 6 train-mag.py --batch_size 1200 --train_stride 2048 --valid_stride 4096 --n_sample 4096 --window_length 512 --hop_length 128 --window hann --epoch 1000 --weight_decay 1e-5 --lr 0.072024 --patience 60 --momentum 0.96875 --exp_num 3 --ddp --mixed

# mag_only/4

3과 같은 세팅 except for patience , increase patience

\efficientnet b4
white noise 0,20
babble noise 0,20
musical noise 0,20

python -m torch.distributed.launch --nproc 6 train-mag.py --batch_size 1200 --train_stride 2048 --valid_stride 4096 --n_sample 4096 --window_length 512 --hop_length 128 --window hann --epoch 1000 --weight_decay 1e-5 --lr 0.072024 --patience 70 --momentum 0.96875 --exp_num 4 --ddp --mixed

# mag-only-mean-teacher/0
\pretrained model
mask teacher = 0/

\unlabeled data (v1_0212)
DSD100
KSS
speech_ko
zeroth_korean

\efficientnet b4
white noise 0,20
babble noise 0,20
musical noise 0,20

python -m torch.distributed.launch --nproc 6 MeanTeacher-mag.py --batch_size 1140 --train_stride 2048 --valid_stride 4096 --n_sample 4096 --window_length 512 --hop_length 128 --window hann --epoch 1000 --weight_decay 1e-5 --lr 5e-2 --patience 10 --momentum 0.95 --exp_num 4 --ddp --mixed --consistency_weight 1 --consistency_rampup 5 --ema_decay 0.999

# mag_only/TUNING/0~20

bo search로 20회 찾는다. epoch = 100

lr, weight decay, momentum




# unlabeled versioning

\v1
DSD100
KSS
speech_ko
zeroth_korean













# phase_only/0

efficientnet b4
white noise 0,20

python -m torch.distributed.launch --nproc 4 train-phase.py --batch_size 1024 --train_stride 2048 --valid_stride 4096 --n_sample 4096 --window_length 512 --hop_length 128 --window hann --epoch 1000 --weight_decay 1e-5 --lr 7e-2 --patience 50 --momentum 0.96 --exp_num 0 --ddp --mixed

# phase_only/1

with non-processing-0210 data

efficientnet b4
white noise 0,20

python -m torch.distributed.launch --nproc 4 train-phase.py --batch_size 1024 --train_stride 2048 --valid_stride 4096 --n_sample 4096 --window_length 512 --hop_length 128 --window hann --epoch 1000 --weight_decay 1e-5 --lr 7e-2 --patience 50 --momentum 0.96 --exp_num 1 --ddp --mixed