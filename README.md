Time: 2023-7

# HandsOnRL

[动手学强化学习](https://hrl.boyuai.com/chapter/intro)

[GitHub - Hands-on-RL](https://github.com/boyu-ai/Hands-on-RL)



```cmd
conda create -n hrl python=3.8
conda activate hrl
# 3.7.2
pip install matplotlib

pip install setuptools==63.2.0
pip install gym==0.18.3

# 1.0.0
conda install jupyter
# 2.2.1
conda install nb_conda

# 4.65.0
conda install tqdm

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

# 2023-9-23

# 下面两个是为了运行 Gym - Atari
# ale-py-0.8.1, importlib-metadata-6.8.0
pip install ale-py
# atari-py-0.2.9, opencv-python-4.8.0.76
pip install gym[atari]
# python -m atari_py.import_roms S:\YYYXUEBING\Project\PyCharm\EnvRL

## Gym - Box2D
# 4.1.1
pip install swig
# box2d-py-2.3.8
pip install gym[box2d]

# 2024-5-15 02:47:41
pip install tyro
```

# XtdPyTorch

【PyTorch深度学习快速入门教程（绝对通俗易懂！）【小土堆】】 https://www.bilibili.com/video/BV1hE411t7RN/



```cmd
conda create -n XtdPyTorch python=3.6
conda activate XtdPyTorch
# 用这个：CUDA 11.3
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install tensorboard
pip install scipy
```

# EasyRL

[蘑菇树EasyRL](https://datawhalechina.github.io/easy-rl/#/?id=%E8%98%91%E8%8F%87%E4%B9%A6easyrl)

[GitHub - easy-rl](https://github.com/datawhalechina/easy-rl)

> 2024-4-30 19:02:35
>
> 注意：在笔记本上新建的环境名称是 envrl

```cmd
# python: 3.7
conda create -n joyrl python=3.7
conda activate joyrl
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 torchaudio==0.10.0 --extra-index-url https://download.pytorch.org/whl/cu113

pip install gym==0.25.2
# 2.5.0
pip install pygame
---
# 跑 附书代码 - DQN 时需要用到的库
pip install matplotlib
pip install seaborn
```

官方给出的方案：

https://github.com/datawhalechina/easy-rl/tree/master/notebooks

```cmd
conda create -n joyrl python=3.7
conda activate joyrl
pip install -r requirements.txt

pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 torchaudio==0.10.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

requirements.txt

```cmd
pyyaml==6.0
ipykernel==6.15.1
jupyter==1.0.0
matplotlib==3.5.3
seaborn==0.12.1
dill==0.3.5.1
argparse==1.4.0
pandas==1.3.5
pyglet==1.5.26
importlib-metadata<5.0
setuptools==65.2.0
```

# Gym

[Gym Documentation](https://www.gymlibrary.dev/)



```cmd
conda create -n gym python=3.8	// 3.8.17
conda activate gym
pip install gym	//0.26.2
pip install swig	//4.1.1
pip install gym[box2d]	//box2d-py 2.3.5
pip install gym[other]	//Pong-v0
pip install ale-py	//Arcade Learning Environment 街机学习环境
pip install gym[accept-rom-license]
```

# PARLTutorials

* [飞桨AI Studio - 世界冠军带你从零实践强化学习](https://aistudio.baidu.com/aistudio/course/introduce/1335)
  * [bili - 世界冠军带你从零实践强化学习](https://www.bilibili.com/video/BV1yv411i7xd/)

* [GitHub - 强化学习算法框架库 - PARL](https://github.com/PaddlePaddle/PARL/)
  * [GitHub - 《PARL强化学习入门实践》课程示例](https://github.com/PaddlePaddle/PARL/tree/develop/examples/tutorials)



```cmd
conda create -n parltutorials python=3.7
conda activate parltutorials
pip install -r path\requirements.txt
```

requirements.txt 的内容如下：

```cmd
# requirements for tutorials (paddle fluid version)
paddlepaddle==1.8.5
parl==1.4
gym==0.18.0
atari-py==0.2.6
rlschool==0.3.1
```

```cmd
pip install protobuf==3.20
```

# BiliLSNT - OldRL

[bili - 蓝斯诺特 - 强化学习 简明教程 代码实战](https://www.bilibili.com/video/BV1Ge4y1i7L6/)



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
# BiliLSNT - MSRL
More Simple Reinforcement Learning

【更简单的强化学习,代码实战】 https://www.bilibili.com/video/BV1X94y1Y7hS/

GitHub: https://github.com/lansinuote/More_Simple_Reinforcement_Learning/tree/main



> python==3.9
>
> pytorch==1.12.1(cpu)
>
> gym==0.26.2
>
> pettingzoo==1.23.1

```cmd
conda create -n MSRL python=3.9
conda activate MSRL
pip install gym==0.26.2
pip install pettingzoo==1.23.1
pip install S:\PyTorch_whl\torch-1.12.1+cu113-cp39-cp39-win_amd64.whl
pip install matplotlib # 3.8.2
pip install pygame # 2.5.2
pip install IPython # 8.18.1
```


