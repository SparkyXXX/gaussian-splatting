## 配置步骤

VSCode安装Remote-SSH插件，编辑config文件如下，连接到服务器

```config
Host A100
    HostName 10.106.11.110
    User huangruixiang
    Port 100
```

安装anaconda

```bash
cd ../A_Common/
bash Anaconda3-2024.02-1-Linux-x86_64.sh
source ~/.bashrc # 一路yes安装完成后刷新环境变量
```

由于服务器不能进行git操作，故将代码克隆到本地，以下几步为本地操作

```bash
git clone https://github.com/graphdeco-inria/gaussian-splatting.git --recursive # 注意在自己电脑上运行
```

再到[glm仓库](https://github.com/g-truc/glm/tree/5c46b9c07008ae65cb81ab79cd677ecc1934b903)下载压缩包，替换本地./submodules/diff-gaussian-rasterization/third_party/glm这个空目录

同样由于不能git，在本地切换分支以便后续安装加速模块

```bash
cd submodules/diff-gaussian-rasterization
git checkout 3dgs_accel
```

现在将本地的gaussian-splatting目录上传到服务器（这里假设服务器上存放代码的根目录为gaussian-splatting，后面操作以这个目录为例），以下操作在服务器上执行

创建虚拟环境

```bash
conda create -n 3dgs python=3.10.6
conda activate 3dgs
```

安装依赖（其中setproctitle是为了在自己使用让别人能看到）

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 
pip install plyfile tqdm tensorboard setproctitle joblib pytorch_msssim
pip install opencv-python==4.10.0.84
cd /home/huangruixiang/gaussian-splatting/submodules/diff-gaussian-rasterization
python setup.py install
cd /home/huangruixiang/gaussian-splatting/submodules/simple-knn
python setup.py install
cd /home/huangruixiang/gaussian-splatting/submodules/fused-ssim
python setup.py install
```

setproctitle用法

```python
from setproctitle import setproctitle
setproctitle("Ruixiang's Work 😆") # 这句加在train.py主程序中
```

到这里环境就配置完了，理论上已经可以用了；可以运行train.py并启用 sparse_adam 来测试环境，注意根据自己数据存放位置更改路径

```bash
python train.py -s data/Hub -m data/Hub/output --test_iterations 5000 10000 --iterations 10000 --optimizer_type sparse_adam
```

## 用到的链接

[原版3DGS仓库](https://github.com/graphdeco-inria/gaussian-splatting?tab=readme-ov-file)

[补充下载的glm](https://github.com/g-truc/glm/tree/5c46b9c07008ae65cb81ab79cd677ecc1934b903)

[步骤参考](https://blog.csdn.net/weixin_64588173/article/details/138140240)