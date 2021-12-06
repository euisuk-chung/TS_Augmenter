# Time Series Generation
Github Repo for timeseries generation (시계열 생성)

## Purpose
Time Series Generation can be used for multiple purposes

For example :
- Time Series Data Augmentations
- Generating Simulations


## How to use
1. TimeGAN을 이용한 시계열 생성

```python
python run_timegan.py

```

2. VRAE를 이용한 시계열 생성

```python
python run_vrae.py

```
## Models Used
### TimeGAN

<img src = 'https://github.com/euisuk-chung/timeseries-generation/blob/main/image/TimeGAN_architecture.PNG?raw=true' width="650" height="400">

- 코드 작성자 : 박경찬
- pyTorch implementation for `TimeGAN`
- Code Reference : https://github.com/d9n13lt4n/timegan-pytorch

### Variational Recurrent AutoEncoder (VRAE)

<img src = 'https://github.com/euisuk-chung/timeseries-generation/blob/main/image/LSTM_VAE_architecture.png?raw=true' width="650" height="400">

- 코드 작성자 : 정의석
- pyTorch implementation for `VRAE`
- Code Reference : https://github.com/tejaslodaya/timeseries-clustering-vae

## CAUTIONS!
The outputs for each model are different! See below for more detail.

### TimeGAN

### Variational Recurrent AutoEncoder (VRAE)
