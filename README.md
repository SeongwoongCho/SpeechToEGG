# SpeechToEGG
speech to egg transformation via deep neural network

|Model Num|Model name          |Methodology                   |Validation Loss| Hyper Parameters                    |data_prep |
|---------|--------------------|------------------------------|---------------|-------------------------------------|----------|
|1        |Wave Unet 4,10      |Cosine Distance Loss          |0.244799       |70,40000,192,1e-2,StepLr(50,0.1)     |1.4, 2.5  | 
|2        |AAI                 |Cosine Distance Loss          |0.290648       |100,40000,192,2e-3,StepLr(10,0.9)    |1.4, 2.5  |
|3        |Wave Unet 4,10      |MSE Loss                      |               |70,40000,192,1e-2,StepLr(50,0.1)     |1.4, 2.5  |
|4        |Wave Unet 4,10      |Cosine Distance Loss(EGG+DEGG)|0.646573       |70,40000,192,1e-2,StepLr(50,0.1)     |1.4, 2.5  |
|5        |Wave Unet 4,10      |Cosine Distance Loss + Ranger |0.247205       |70,40000,192,1e-2,StepLr(50,0.1)     |1.4, 2.5  |
|6        |Resv2Unet 4,10,15,5 |Cosine Distance Loss + Ranger |0.193103       |70,16000,192,1e-2,StepLr(50,0.1)     |1.4, 2.5  |
|7        |Resv2Unet 4,15,15,5 |Cosine Distance Loss + Ranger |0.161064       |110,11000,192,1e-2,StepLr(85,0.1)    |1.4, 2.5  |
|8        |Resv2Unet 4,20,9,5  |Cosine Distance Loss + Ranger |0.145366       |90,8000,192,1e-2,StepLr(85,0.1)      |1.4, 2.5  |
|9        |Resv2Unet+ 5,64,15,5|Cosine Distance Loss + Ranger |0.131211       |100,9000,192,1e-2,StepLr(80,0.1)     |1.4, 2.5  |
|10       |Resv2Unet+ 5,64,15,5|finetune 9 with MSE + Ranger  |0.015437       |6,8000,192,4e-4                      |1.4, 2.5  |
|11       |Resv2Unet+ 5,32,15,5|CDL + Ranger                  |0.144715       |70,14500,320,1e-2,StepLr(50,0.1)     |1.25, 4   |

# Test Results

conf 1. smooth = 49 (exp 1,2,3,4)  --> top db 10, step 64, find_peaks_cwt
conf 2. smooth = 15 (exp 6,7)  --> top db 25, step 64, find_peaks_cwt
conf 3. smooth = 49, top db 15, step 64, detect_peaks.detect_peaks(v,mph=0.01, mpd=45)

smooth = 49, top db 15, step 64로 이후로 계속 평가

| Test Results   |      |     |     |      |             |     |     |       | 
|----------------|------|-----|-----|------|-------------|-----|-----|-------|
|                | CMU  |  <  |  <  |  <   | saarbrucken |  <  | <   |  <    |
|                | IDR  | MR  | FAR | IDA  | IDR         | MR  | FAR | IDA   |
|================|------|-----|-----|------|-------------|-----|-----|-------|
|  1-DEGG_high(1)|95.11%|4.08%|0.80%|0.67ms| 100%        | 0%  | 0%  | 0ms   |
|  1-DEGG_low(1) |95.74%|3.66%|0.61%|0.24ms| 100%        | 0%  | 0%  | 0ms   |
|  1-EGG_high(1) |94.01%|3.99%|2.00%|0.63ms| 100%        | 0%  | 0%  | 0ms   |
|  1-EGG_low(1)  |94.21%|3.97%|1.82%|0.31ms| 100%        | 0%  | 0%  | 0ms   |
|  2-DEGG_high(1)|94.26%|4.83%|0.92%|0.73ms| 100%        | 0%  | 0%  | 0ms   |
|  2-DEGG_low(1) |95.04%|4.31%|0.65%|0.25ms| 100%        | 0%  | 0%  | 0ms   |
|  2-EGG_high(1) |92.98%|4.95%|2.07%|0.65ms| 100%        | 0%  | 0%  | 0ms   |
|  2-EGG_low(1)  |93.22%|4.89%|1.89%|0.34ms| 100%        | 0%  | 0%  | 0ms   |
|  3-DEGG_high(1)|92.52%|6.13%|1.35%|0.75ms| 100%        | 0%  | 0%  | 0ms   |
|  3-DEGG_low(1) |93.18%|5.69%|1.13%|0.30ms| 100%        | 0%  | 0%  | 0ms   |
|  3-EGG_high(1) |90.97%|6.85%|2.18%|0.69ms| 100%        | 0%  | 0%  | 0ms   |
|  3-EGG_low(1)  |91.51%|6.63%|1.86%|0.37ms| 100%        | 0%  | 0%  | 0ms   |
|  4-DEGG_high(1)|94.83%|4.44%|0.73%|0.68ms| 100%        | 0%  | 0%  | 0ms   |
|  4-DEGG_low(1) |95.38%|4.05%|0.56%|0.22ms| 100%        | 0%  | 0%  | 0ms   |
|  4-EGG_high(1) |93.43%|4.62%|1.95%|0.63ms| 100%        | 0%  | 0%  | 0ms   |
|  4-EGG_low(1)  |93.73%|4.54%|1.73%|0.30ms| 100%        | 0%  | 0%  | 0ms   |
|6-DEGG_high(15) |90.59%|2.68%|6.72%|0.93ms| 100%        | 0%  | 0%  | 0ms   |
| 6-DEGG_low(15) |91.81%|2.05%|6.14%|0.43ms| 100%        | 0%  | 0%  | 0ms   |
| 6-EGG_high(15) |90.42%|2.12%|7.46%|0.77ms| 100%        | 0%  | 0%  | 0ms   |
|  6-EGG_low(15) |90.53%|2.17%|7.30%|0.53ms| 100%        | 0%  | 0%  | 0ms   |
|7-DEGG_high(15) |91.40%|2.37%|6.23%|0.92ms| 100%        | 0%  | 0%  | 0ms   |
| 7-DEGG_low(15) |92.51%|1.83%|5.66%|0.43ms| 100%        | 0%  | 0%  | 0ms   |
| 7-EGG_high(15) |91.32%|1.72%|6.96%|0.78ms| 100%        | 0%  | 0%  | 0ms   |
| 7-EGG_low(15)  |91.60%|1.67%|6.73%|0.53ms| 100%        | 0%  | 0%  | 0ms   |
|7-DEGG_high(49) |92.61%|2.73%|4.66%|1.00ms| 100%        | 0%  | 0%  | 0ms   |
| 7-DEGG_low(49) |93.64%|2.23%|4.13%|0.48ms| 100%        | 0%  | 0%  | 0ms   |
| 7-EGG_high(49) |92.07%|2.50%|5.42%|0.86ms| 100%        | 0%  | 0%  | 0ms   |
| 7-EGG_low(49)  |92.33%|2.47%|5.20%|0.62ms| 100%        | 0%  | 0%  | 0ms   |
|9-DEGG_high(15) |93.75%|1.51%|4.73%|0.87ms| 100%        | 0%  | 0%  | 0ms   |
| 9-DEGG_low(15) |94.68%|1.06%|4.26%|0.35ms| 100%        | 0%  | 0%  | 0ms   |
| 9-EGG_high(15) |93.48%|1.17%|5.35%|0.73ms| 100%        | 0%  | 0%  | 0ms   |
| 9-EGG_low(15)  |93.69%|1.16%|5.15%|0.46ms| 100%        | 0%  | 0%  | 0ms   |
|10-DEGG_high(15)|93.69%|1.82%|4.48%|0.89ms| 100%        | 0%  | 0%  | 0ms   |
|10-DEGG_low(15) |94.66%|1.34%|4.00%|0.37ms| 100%        | 0%  | 0%  | 0ms   |
|10-EGG_high(15) |93.52%|1.50%|4.98%|0.73ms| 100%        | 0%  | 0%  | 0ms   |
|10-EGG_low(15)  |93.79%|1.48%|4.73%|0.49ms| 100%        | 0%  | 0%  | 0ms   |
|9-DEGG_high(3)  |97.55%|0.44%|2.01%|0.56ms| 100%        | 0%  | 0%  | 0ms   |
| 9-DEGG_low(3)  |98.16%|0.15%|1.70%|0.21ms| 100%        | 0%  | 0%  | 0ms   |
| 9-EGG_high(3)  |96.83%|1.33%|1.83%|0.53ms| 100%        | 0%  | 0%  | 0ms   |
| 9-EGG_low(3)   |96.47%|1.90%|1.63%|0.25ms| 100%        | 0%  | 0%  | 0ms   |
|11-DEGG_high(3) |97.38%|0.45%|2.17%|0.57ms| 100%        | 0%  | 0%  | 0ms   |
| 11-DEGG_low(3) |97.84%|0.23%|1.93%|0.22ms| 100%        | 0%  | 0%  | 0ms   |
| 11-EGG_high(3) |96.20%|1.91%|1.89%|0.56ms| 100%        | 0%  | 0%  | 0ms   |
| 11-EGG_low(3)  |95.68%|2.62%|1.70%|0.28ms| 100%        | 0%  | 0%  | 0ms   |