## mask

# exp 0

EfficientUnet b0 + clamp phase(-np.pi,np.pi)

--batch_size 512
--epoch 1000 
--lr 3e-4
--lr_step 400
--n_frame 64

# exp 1 : Model change / more epoch / lr scheduling policy/ mask label smoothing

EfficientUnet b1 + clamp magnitude(-12,7) + clamp phase(-np.pi,np.pi)

--batch_size 440
--epoch 10000
--lr 3e-4
--reduce LR on Plateau(optimizer, mode='min', factor=0.1, patience=100, verbose=True) watch valid loss
--n_frame 64
--mask label smoothing

# exp 2 : Model change / more epoch / modify error/summary writier -> more efficient memory usage

EfficientUnet b2 + clamp magnitude(-12,7) + clamp phase(-np.pi,np.pi) + clamp mask (-10,10)

--batch_size 600
--epoch 15000
--lr 3e-4
--reduce LR on Plateau(optimizer, mode='min', factor=0.1, patience=100, verbose=True) watch valid loss
--n_frame 64

----총 48시간 학습
----L1 signal distance 기준 0.041 달성

# exp 3 : exp2 + CBAM attention

EfficientUnet b2 CBAM+ clamp magnitude(-12,7) + clamp phase(-np.pi,np.pi) + clamp mask (-10,10)

--batch_size 600
--epoch 15000
--lr 3e-4
--reduce LR on Plateau(optimizer, mode='min', factor=0.1, patience=100, verbose=True) watch valid loss
--n_frame 64

----총 48시간 학습
----L1 signal distance 기준 0.043 달성

# exp 4 : exp2 + loss 변경

EfficientUnet b2 + clamp magnitude(-12,7) + clamp phase(-np.pi,np.pi) + clamp mask (-10,10)

loss = mask loss + mag loss + phase loss
phase loss의 masked weight를 삭제함.

--batch_size 600
--epoch 15000
--lr 3e-4
--reduce LR on Plateau(optimizer, mode='min', factor=0.1, patience=100, verbose=True) watch valid loss
--n_frame 64

----150epoch 정도 확인한 결과 phase에 대한 학습이 매우 느리다.
----그래서 stop함

# exp 5 : exp2 + mask weighted mag loss 

EfficientUnet b2 + clamp magnitude(-12,7) + clamp phase(-np.pi,np.pi) + clamp mask (-10,10)

loss = mask loss + mag loss + phase loss
mask loss에 masked weight를 준 뒤 학습함.

--batch_size 600
--epoch 15000
--lr 3e-4
--reduce LR on Plateau(optimizer, mode='min', factor=0.1, patience=100, verbose=True) watch valid loss
--n_frame 64

##### 하 ㅅㅂ 위의 실험들 val loss 까지 backward 시킴...
##### 그래도 train loss의 변화만으로 얻은 결론은 attention 별로 효과 X, weighted mask loss를 사용하는 것이 더 좋은 것! 

# exp 6 : signal Loss : L1 -> cosine distance

EfficientUnet b2 + clamp magnitude(-12,7) + clamp phase(-np.pi,np.pi) + clamp mask (-10,10)

loss = mask loss + mag loss + phase loss

--batch_size 600
--epoch 15000
--lr 3e-4
--reduce LR on Plateau(optimizer, mode='min', factor=0.1, patience=100, verbose=True) watch valid loss
--n_frame 64


# exp 6-1 : FineTune exp6 with signal loss

#### model.load_state_dict(torch.load('./models/masked/exp6/best_1682.pth'))

EfficientUnet b2 + clamp magnitude(-12,7) + clamp phase(-np.pi,np.pi) + clamp mask (-10,10)

loss = cosine distance loss 

--batch_size 512
--epoch 15000
--lr 3e-5
--reduce LR on Plateau(optimizer, mode='min', factor=0.1, patience=100, verbose=True) watch valid loss
--n_frame 64

---- FinetTuning하면 cds는 작아지는데, mask loss, mag loss, phase loss 커진다..?

# exp 7

EfficientUnet b2 + clamp magnitude(-12,7) + clamp phase(-np.pi,np.pi) + clamp mask (-10,10)

loss = cosine distance loss 

--batch_size 600
--epoch 15000
--lr 3e-4
--reduce LR on Plateau(optimizer, mode='min', factor=0.1, patience=100, verbose=True) watch valid loss
--n_frame 64

----절대 수렴안함...!!!! 개별로!!!! 

# exp 6 early stopping 500으로 늘려서 다시진행  --> 6-re

# exp 8 : change L1 loss to L2 loss 

EfficientUnet b2 + clamp magnitude(-12,7) + clamp phase(-np.pi,np.pi) + clamp mask (-10,10)

loss = L2(mask loss + mag loss + phase loss)

--batch_size 600
--epoch 15000
--lr 3e-4
--reduce LR on Plateau(optimizer, mode='min', factor=0.1, patience=500, verbose=True) watch valid metric
--n_frame 64

# exp 9 : SSL Method - Mean Teacher 

pretrained model : 6-re 5761

DataSet Size
- Train : 16089
- Valid : 12560
- Unlabeled : '725202
  - DSD100, KSS, zeroth_korean, speech_ko
  - 일단은 librispeech 빼고 진행(librispeech 포함하면 '3845087개 )

remove masking augmentation

EfficientUnet b2 + clamp magnitude(-12,7) + clamp phase(-np.pi,np.pi) + clamp mask (-10,10)

loss = supervised_loss + consistency loss

supervised_loss -> L1
consistency loss -> L1

--consistency weight 100
--ema_decay 0.999
--batch_size 384
--epoch 120
--lr 3e-4
--CosineAnnealingLR
--n_frame 64

2000iter/40min 정도 --> 1iter/1s 정도 예상 cache에 다올라간 후엔 시간이 거의 걸리지 않음.

# exp 10 : Change consistency weight 100 -> 1, reduce epoch -> 70

pretrained model : 6-re 5761

DataSet Size
- Train : 16089
- Valid : 12560
- Unlabeled : '725202
  - DSD100, KSS, zeroth_korean, speech_ko
  - 일단은 librispeech 빼고 진행(librispeech 포함하면 '3845087개 )

EfficientUnet b2 + clamp magnitude(-12,7) + clamp phase(-np.pi,np.pi) + clamp mask (-10,10)

loss = supervised_loss + consistency loss

supervised_loss -> L1
consistency loss -> L1

--consistency weight 1
--ema_decay 0.999
--batch_size 384
--epoch 70
--lr 3e-4
--CosineAnnealingLR
--n_frame 64

##### EXP 9,10 Mean Teacher model 은 나중에 재시도 .
##### HDD Read 속도 때문인 것 같다 ....  ㅅㅂ ㅅㅂ !!!!!!!!!!!

# exp 12 : exp6-re --> RAdamW + CosineAnnealing data loading 방식 변경

EfficientUnet b2 + clamp magnitude(-12,7) + clamp phase(-np.pi,np.pi) + clamp mask (-10,10)

한 파일에 저장후 mmap 으로 읽는다. 

train : (2, 257, '3261590)
valid : (2, 257, '2567915)

loss = mask loss + mag loss + phase loss

--batch_size 700
--epoch 750
--lr 3e-4
--CosineAnnealing(optimizer)
--n_frame 64

stride = 16

# exp 13 : lR 4e-3 -> 3e-3 , cosine Annealing -> reduce LR on plateau, remove background noise

EfficientUnet b2 + clamp magnitude(-12,7) + clamp phase(-np.pi,np.pi) + clamp mask (-10,10)

한 파일에 저장후 mmap 으로 읽는다. 

train : (2, 257, '3261590)
valid : (2, 257, '2567915)

loss = mask loss + mag loss + phase loss

--batch_size 700
--epoch 2000
--lr 3e-3
--n_frame 64
--patience 25

stride = 16

# exp 14 : efficientnet b4, lR 4e-3 -> 3e-3 , cosine Annealing -> reduce LR on plateau, remove background noise

EfficientUnet b4 + clamp magnitude(-12,7) + clamp phase(-np.pi,np.pi) + clamp mask (-10,10)

한 파일에 저장후 mmap 으로 읽는다. 

train : (2, 257, '3261590)
valid : (2, 257, '2567915)

loss = mask loss + mag loss + phase loss

--batch_size 512
--epoch 15000
--lr 3e-3
--n_frame 64
--patience 400

train_stride = 192
valid_stride = 64