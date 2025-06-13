# Hunyuan3D-2.1-Shape


# 训练

我们会展示小数据集上DiT的训练全流程

## 数据预处理

渲染和水密化参考[链接](tools/README.md)，最终得到如下结构

``` yaml
dataset/preprocessed/{uid}
├── geo_data
│   ├── {uid}_sdf.npz
│   ├── {uid}_surface.npz
│   └── {uid}_watertight.obj
└── render_cond
    ├── 000.png
    ├── ...
    ├── 023.png
    ├── mesh.ply
    └── transforms.json
```

我们提供了一个8个case(均来自Objaverse-XL)预处理后的结果在 tools/mini_trainset，可以直接用于过拟合训练



## 启动训练

我们提供了可供参考的训练配置文件和启动脚本（默认单机8卡deepspeed训练），用户根据需要自行修改。

配置文件
```
configs/dit-from-scratch-overfitting-flowmatching-dinog518-bf16-lr1e4-1024.yaml
```
启动脚本

```
export node_num=1
export node_rank=0
export master_ip=0.0.0.0 # set your master_ip
export config='configs/dit-from-scratch-overfitting-flowmatching-dinog518-bf16-lr1e4-1024.yaml'
export output_dir='output_folder/dit/overfitting'
bash scripts/train_deepspeed.sh $node_num $node_rank $master_ip $config $output_dir
```