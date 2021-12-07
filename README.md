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

Training method for each model are the same, which uses dataset that is loaded by moving sliding window(default=30) with certain stride(default=1).

<img src = 'https://github.com/euisuk-chung/timeseries-generation/blob/main/image/train_test_image.png?raw=true' width="650" height="400">

However, the generation method for each model are different! See below for more detail.

### TimeGAN
TimeGAN has 2 Modes, which is used to decide whether to train or generate :
1. is_train (default = True) : train model with loaded train data (window_size=30, stride=1)
2. is_generate (default = True) : generate multiple(num_generation) sequences of window (window_size=30)

```
# Mode 1 : Train mode
--is_train # train timeGAN

# Mode 2 : Generation mode
--is_generate # generate window size sequences
--num_generation # number of sequences to make

```

### Variational Recurrent AutoEncoder (VRAE)
VRAE has 3 Modes, which is used to decide whether to train or generate(train) or generate(test) :
1. is_train (default = True) : train model with loaded train data (window_size=30, stride=1)
2. is_generate_train (default = True) : generate train dataset loaded sequentially (window_size=stride)
3. is_generate_test (default = True) : generate test dataset loaded sequentially (window_size=stride)

```
# Mode 1 : Train mode
--is_train # train VRAE

# Mode 2 : Train Generation mode
--is_generate_train # generate train dataset

# Mode 3 : Test Generation mode
--is_generate_test # generate test dataset

```
