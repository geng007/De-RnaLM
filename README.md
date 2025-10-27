# 基于Transformer的基因表达与药物响应预测模型

## 项目概述
本项目旨在利用Transformer架构构建一个高精度的基因表达与药物响应预测模型。通过整合基因表达数据、药物特征信息以及细胞系信息，实现对药物处理后基因表达变化的有效预测，为药物研发与个性化医疗提供技术支持。

## 核心功能
1. **数据预处理**：支持对基因表达数据（AnnData格式）进行加载和预处理，包括数据标准化、对数转换等操作，同时提供药物分子指纹（Morgan指纹）的预计算功能。
2. **模型构建**：基于Transformer架构设计了`PerturbationTransformer`模型，能够有效融合基因表达、药物特征、细胞系及剂量信息，进行精准预测。
3. **训练与优化**：支持多种损失函数（MSE、NB、Gaussian NLL）、优化器（Adam、AdamW）和学习率调度器（ReduceLROnPlateau、CosineAnnealingLR），并集成自动混合精度训练（AMP）和模型编译（torch.compile）功能，提升训练效率。
4. **评估与可视化**：训练过程中实时记录损失、MSE、R²等指标，并提供训练曲线可视化、预测结果散点图可视化功能，便于分析模型性能。

## 目录结构
```
.
├── main.py                # 主运行脚本
├── README.md              # 项目说明文档
├── visualization_transformer_v2  # 可视化结果保存目录
│   └── ...
├── ckpts_tf_v2            # 模型检查点保存目录
│   └── ...
├── res_tf_v2              # 结果保存目录
│   └── ...
└── dataset                # 数据集目录
    └── xxxx.h5ad  # 示例数据集
```

## 安装依赖
```bash
pip install torch scanpy rdkit pandas matplotlib scikit-learn tqdm
```

## 使用方法
1. **参数配置**：通过命令行参数或修改`main.py`中的`parse_args`函数配置训练参数，包括数据路径、模型超参数、训练策略等。
2. **运行训练**：执行以下命令启动训练：
```bash
python main.py --data_path ./dataset/Lincs_L1000.h5ad --epochs 10 --batch_size 128
```
3. **模型评估**：训练完成后，模型会自动在测试集上进行评估，并生成可视化结果和评估报告，保存在`res_tf_v2`和`visualization_transformer_v2`目录下。

## 参数说明
| 参数名称 | 默认值 | 说明 |
|----------|--------|------|
| `data_path` | `./dataset/xxxx.h5ad` | 基因表达数据（AnnData格式）路径 |
| `split_key` | `random_split_0` | 数据集划分标识列名 |
| `save_dir` | `./ckpts_tf_v2/` | 模型保存目录 |
| `results_dir` | `./res_tf_v2/` | 结果保存目录 |
| `epochs` | 10 | 训练轮数 |
| `batch_size` | 128 | 批次大小 |
| `lr` | 1e-4 | 学习率 |
| `weight_decay` | 1e-5 | 权重衰减 |
| `optimizer` | `AdamW` | 优化器类型（Adam, AdamW） |
| `scheduler` | `CosineAnnealingLR` | 学习率调度器类型（ReduceLROnPlateau, CosineAnnealingLR, None） |
| `drug_dimension` | 1024 | 药物特征维度 |
| `d_model` | 256 | Transformer模型维度 |
| `nhead` | 4 | Transformer多头注意力头数 |
| `num_encoder_layers` | 3 | Transformer编码器层数 |
| `dim_feedforward` | 512 | Transformer前馈神经网络维度 |
| `dropout_transformer` | 0.1 | Transformer dropout率 |
| `use_conditions_individually` | True | 是否单独处理条件输入 |
| `use_cls_token` | True | 是否使用CLS token |
| `loss` | `['MSE']` | 损失函数类型（NB, GUSS, MSE） |


## 贡献与反馈
欢迎提交PR或在Issues中反馈问题、建议。如果本项目对您有帮助，欢迎Star支持！

## 许可证
本项目采用[MIT License](LICENSE)。 
