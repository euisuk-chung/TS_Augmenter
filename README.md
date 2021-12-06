# Time Series Generation
Github Repo for timeseries generation (시계열 생성)

## Purpose
Time Series Generation can be used for multiple purposes

For example :
- Time Series Data Augmentations
- Generating Simulations

---

## Model Used
### TimeGAN
코드 작성자 : 박경찬
Code Reference : https://github.com/d9n13lt4n/timegan-pytorch
<img src = 'https://github.com/euisuk-chung/timeseries-generation/blob/main/image/TimeGAN_architecture.PNG?raw=true'>

### VRAE
코드 작성자 : 정의석
Code Reference : https://github.com/tejaslodaya/timeseries-clustering-vae
<img src = 'https://github.com/euisuk-chung/timeseries-generation/blob/main/image/LSTM_VAE_architecture.png?raw=true'>

---

## How to use
1. TimeGAN을 이용한 시계열 생성

```python
python run_timegan.py
```

2. VRAE를 이용한 시계열 생성

```python
python run_vrae.py
```
