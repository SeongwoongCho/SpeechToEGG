- train_stride 16
- valid_stride 64
- 초기 몇 epoch 만 비교하자!
    - 100 epoch 정도면 괜찮지 않을까.

- preprocessing
    - window length : 512
    - hop length : 128
    - window type : 'hann'

- augmentation
    - mix two stft inputs
    - 50% 확률로 normal noise 섞음 / 40% 확률로 Music noise 섞음
    - db level은 0 ~ 35 사이의 random uniform
    - X = add_whitenoise(X)

- model structure
    - Final activation funciton
        - Magnitude 
        - Phase 
        - Mask 

- training parameter
    - optimizer
        - [RMSProp,Adagrad, RAdam, RAdamW]
        - [Lookahead]
            - alpha, k
        - weight decay 
    - loss
        - mask loss
            - BCE
                - BCEWeight
            - DiceLoss
            - BCE + DiceLoss
        - lambda,gamma
            mask_loss + gamma * (mag_loss + lambda*phase_loss)

- to be tuned 
    - Discrete
        - optimizer 4 : 4개에서 랜덤 서치
        - mask loss 3 : 3개에서 랜덤 서치 
    - continuous
        - initial learning rate : 10**(random(-2,-5))
        - weight decay : 10**(-3,-6)
        - Lookahead parameter alpha : 0.4~0.7 float
        - Lookahead parameter k : 4~7 integer
        - pos_weight(BCE weight) : 10**(0,1)
        - loss parameter gamma : 10**(-0.5,0.5)
        - loss parameter lambda : 10**(-0.5,0.5)
    
    
1. random search를 하자.
2. bayesian optimization
    각 optimizer, 각 loss 별 탐색 ? 
    각 탐색당 20~30 회라고 하면 12*(20~30)= 360
    너무 heavy 하다 ㅅㅂ! 
    
1. random search ㄲ! 