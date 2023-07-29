Time: 2023-7

EasyRL

```cmd
# python: 3.7
conda create -n envrl python=3.7
conda activate envrl
# PyTorch v1.12.1
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 torchaudio==0.10.0 --extra-index-url https://download.pytorch.org/whl/cu113
# gym
pip install gym==0.25.2
// 2.5.0
pip install pygame
```

bili

```cmd
# python: 3.9
conda create -n rlbili python=3.9
conda activate rlbili
# PyTorch v1.12.1
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 torchaudio==0.10.0 --extra-index-url https://download.pytorch.org/whl/cu113
# gym 0.26.2
pip install gym==0.26.2
// 2.1.0
pip install pygame

# 下面这两个是使用 rgb_array 模式创建gym环境时使用的，用来显示游戏画面，在 Jupyter Notebook 中显示的是动画形式，但是在 PyCharm 中是逐个图片显示的；如果一直在 PyCharm 中使用 human 模式创建gym环境，或许可能不需要这两个包。
// 3.7.2
pip install matplotlib
// 8.14.0
pip install IPython

# 先安装 swig，才可以正确安装 gym[box2d]
// 4.1.1
pip install swig
pip install gym[box2d]
```

