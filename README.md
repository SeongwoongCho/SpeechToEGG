# SpeechToEGG
speech to egg transformation via deep neural network

|Model Num|Model name    |Methodology         |Validation Loss| Hyper Parameters                |
|---------|--------------|--------------------|---------------|---------------------------------|
|1        |Wave Unet 4,10|Cosine Distance Loss|0.244799       |70,40000,192,1e-2,StepLr(100,0.1)|
|2        |AAI           |Cosine Distance Loss|0.290648       |70,40000,192,2e-3,StepLr(10,0.9) |
|3        |Wave Unet 4,10|MSE Loss            |               |70,40000,192,1e-2,StepLr(100,0.1)|

# Test Results

| Test Results |     |    |     |     |             |     |     |       | 
|--------------|-----|----|-----|-----|-------------|-----|-----|-------|
|              | CMU |  < |  <  |  <  | saarbrucken |  <  | <   |  <    |
|              | IDR | MR | FAR | IDA | IDR         | MR  | FAR | IDA   |
|==============|-----|----|-----|-----|-------------|-----|-----|-------|
|      1       | 100%| 0% | 0%  | 0ms | 100%        | 0%  | 0%  | 0ms   |
|      2       | 100%| 0% | 0%  | 0ms | 100%        | 0%  | 0%  | 0ms   |
|      3       | 100%| 0% | 0%  | 0ms | 100%        | 0%  | 0%  | 0ms   |
