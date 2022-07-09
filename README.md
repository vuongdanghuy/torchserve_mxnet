# torchserve_mxnet
A simple repository to debug deploy MXNet model with TorchServe

How to run:
1. Clone this repository
2. Create some folders
```
mkdir model_store
mkdir src
```
3. Download [weight](https://drive.google.com/file/d/1mjSiw68UEYAVTEc-0qsEeLLa_WXwHCyh/view?usp=sharing) and extract to `src` folder
4. Run `make` to create .mar file
5. Run `make run` to start TorchServe
6. Run `make stop` or `make clean` to stop TorchServe
