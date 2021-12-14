# Time Series Generation
Github Repo for timeseries generation (시계열 생성)

## 1. Purpose
Time Series Generation can be used for multiple purposes

For example :
- Time Series Data Augmentations
- Generating Simulations


## 2. How to use

- You can check detail about the argument at `4. Model Parameters`

### 2.1. Time Series Generation using TimeGAN

```python
# Example
python run_timegan.py --file_name test_data --cols_to_remove Time MNG_NO --time_gap 500 --emb_epochs 10 --sup_epochs 10 --gan_epochs 10 --window_size 5
```

### 2.2. VRAE를 이용한 시계열 생성

```python

# Example
python run_vrae.py --file_name test_data --cols_to_remove Time MNG_NO --time_gap 500 --n_epochs 10 --window_size 5

```

## 3. Models Used
### 3.1. TimeGAN

<img src = 'https://github.com/euisuk-chung/timeseries-generation/blob/main/image/TimeGAN_architecture.PNG?raw=true' width="800" height="400">

- pyTorch implementation for `TimeGAN`
- Code Reference : https://github.com/d9n13lt4n/timegan-pytorch

### 3.2. Variational Recurrent AutoEncoder (VRAE)

<img src = 'https://github.com/euisuk-chung/timeseries-generation/blob/main/image/LSTM_VAE_architecture.png?raw=true' width="800" height="400">

- pyTorch implementation for `VRAE`
- Code Reference : https://github.com/tejaslodaya/timeseries-clustering-vae

## 4. Model Arguments

- Indeed, TimeGAN and VRAE have shared parameters, however there are also lot of parameters that are not shared.
- Therefore, Arguments for each model is in `config_timegan.py` and `config_vrae.py`

### 4.1. Shared Arguments

Training method for each model are the same, which uses dataset that is loaded by moving sliding window(default=30) with certain stride(default=1).

<img src = 'https://github.com/euisuk-chung/timeseries-generation/blob/main/image/train_test_image.png?raw=true' width="800" height="400">

There are few things you need to know before implementing our code : 

- The generation method for each model are different :
    - TimeGAN generates window sized timeseries from `random noise` (without any input)
    - VRAE generates window sized timeseries from `given timeseries` and `trained latent space` (with input)
    
- The query for train/test split in my code is currently used for my side-project.
    - If you want to use train/test you need to go to `utils.custom_dataset` and change the query.
    - For generation purpose, you don't have to worry about train/test `split (defalut = False)`
    
Here are the following arguments:
``` 
--file_name file_name # 분석에 사용할 파일이름

--cols_to_remove Var1 Var2 # 분석에서 제외할 변수이름 (Ex. time var, idx var) 

--time_gap 500 # 데이터 수집 GAP(텀)

--window_size 10 # 학습에 사용할 윈도우 크기

```

### 4.2. TimeGAN Arguments

TimeGAN has following modes (for more check `config_timegan.py`) :

1. is_train (default = True) : train model with loaded train data (window_size=30, stride=1)
2. is_generate (default = True) : generate multiple(num_generation) sequences of window (window_size=30)

```
--is_train `True` or `False` # Train 데이터를 이용한 학습

--num_generation 100 # 생성할 데이터 윈도우의 개수

--is_generate `True` or `False` # 학습된 latent space를 바탕으로 데이터 생성

--emb_epochs 3000 # AutoEncoder 학습 Epoch 수

--sup_epochs 3000 # Supervisor 학습 Epoch 수

--gan_epochs 3000 # GAN 모델 학습 Epoch 수

```

### 4.3. Variational Recurrent AutoEncoder (VRAE) Arguments

VRAE has following modes (for more check `config_vrae.py`) :

1. is_train (default = True) : train model with loaded train data (window_size=30, stride=1)
2. is_generate_train (default = True) : generate train dataset loaded sequentially (stride=window_size)
3. is_generate_test (default = False) : generate test dataset loaded sequentially (stride=window_size)

```
--is_train `True` or `False` # Train 데이터를 이용한 학습

--is_generate_train `True` or `False` # 학습된 latent space와 Train Data를 바탕으로 데이터 생성 

--is_generate_test `True` or `False` # 학습된 latent space와 Test Data를 바탕으로 데이터 생성 (실험용 : 데이터 생성 시 사용할 필요 X)

--n_epochs 2000 # 모델 학습할 Epoch 수

```

## Repository Structure
```
├── data
│   ├── Data You Want to Use (in pkl)
├── gen_data_gan
│   └── where GAN Generated Data are saved
├── gen_data_vae
│   └── where VAE Generated Data are saved
├── models
│   ├── TimeGAN.py
│   └── vrae.py
├── save_model
│   └── where model parameters get saved 
├── utils
│   ├── TSTR.py # TSTR(TRTS) code
│   ├── custom_dataset.py # dataloading code
│   ├── utils_timegan.py # util function for timegan
│   ├── utils_vrae.py # util function for vrae
│   ├── visual_timegan.py # visualization function for timegan
│   └── visual_vrae.py # visualization function for vrae
├── run_timegan.py
├── run_vrae.py
├── config_timegan.py
├── config_vrae.py
```
