## 原版 3DGS 运行指令
```shell
# 将输入图像和 COLMAP 数据转换为 3D Gaussian Splatting 训练所需的格式
python convert.py -s data/Hub
python convert.py -s data/Hub --no_gpu # 让COLMAP使用cpu 运行特征提取和特征匹配
python convert.py -s data/Hub --skip_matching	# 跳过 COLMAP 的特征提取和匹配步骤（用于已存在 COLMAP 结果，即存在distorted和input目录，只进行去畸变）

# 训练
python train.py -s data/Hub -m data/Hub/output
python train.py -s data/Hub -m data/Hub/output --eval # 启用训练/测试集拆分
python train.py -s data/Hub -m data/Hub/output -r 1 # 不要缩放宽度大于1.6k的图像
python train.py -s data/Hub -m data/Hub/output --iterations	10000 # 指定训练迭代次数
python train.py -s data/Hub -m data/Hub/output --debug # 启用调试模式，光栅化失败生成 dump 文件排查问题
python train.py -s data/Hub -m data/Hub/output --debug_from 10000 # 从指定轮数开始启用调试模式 
python train.py -s data/Hub -m data/Hub/output --test_iterations 10000 20000 30000 # 计算测试集 L1和PSNR 的迭代次数
python train.py -s data/Hub -m data/Hub/output --save_iterations 10000 20000 30000 # 保存模型的迭代次数
python train.py -s data/Hub -m data/Hub/output --checkpoint_iterations # 保存 checkpoint 的迭代次数，用于断点续训
python train.py -s data/Hub -m data/Hub/output --start_checkpoint ./output/xxx/chkpnt10000.pth # 用于续训的 checkpoint 路径
python train.py -s data/Hub -m data/Hub/output --optimizer_type sparse_adam # 启用稀疏 Adam 优化器，训练速度提升2.7倍

# 渲染
python render.py -m ./data/Hub/output
python render.py -m ./data/Hub/output --eval # 启用训练/测试集拆分，需要与train命令对应
python render.py -m ./data/Hub/output --iteration 10000 # 指定渲染的模型迭代次数，不指定默认为最新
python render.py -m ./data/Hub/output --skip_train # 跳过训练集渲染
python render.py -m ./data/Hub/output --skip_test # 跳过测试集渲染

# 定量评估，需要先运行render
python metrics.py -m ./data/Hub/output
python metrics.py -m ./data/Hub/output ./data/another_scene/output # 批量计算

# 用viewer查看模型结果
 .\submodules\viewers\bin\SIBR_gaussianViewer_app -m data/Hub/output

```

## 实验测试
```shell
python train.py -s data/Hub -m data/Hub/output --iterations 100 --optimizer_type sparse_adam
python train.py -s data/Hub -m data/Hub/output --eval --iterations 30000 --optimizer_type sparse_adam --test_iterations 10000 20000 30000 --save_iterations 10000 20000 30000

python train.py -s data/City -m data/City/output --optimizer_type sparse_adam
python train.py -s data/Garden -m data/Garden/output --optimizer_type sparse_adam --iterations 10000
python render.py -m ./data/Garden/output
python choose_camera.py --small_ply ./data/Garden/output/point_cloud/small_pcd.ply --big_ply ./data/Garden/output/point_cloud/big_pcd.ply --threshold 0.6 --source_path ./data/Garden
```