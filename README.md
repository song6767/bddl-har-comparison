# CNN 与 RNN 的人体活动识别比较

本项目比较一维卷积网络（CNN）、普通循环网络（RNN）和长短期记忆网络（LSTM）在 UCI HAR 人体活动识别任务中的分类表现、训练效率与稳定性，用于 BDDL 课程第三项作业。

公开仓库：[song6767/bddl-har-comparison](https://github.com/song6767/bddl-har-comparison)。仓库保存本项目的 PyTorch 实现、运行脚本、实验协议和轻量结果汇总；数据从 UCI 官方来源单独下载，数据集作者信息与许可见下文。

当前阶段的目标是下载数据、建立可复现的人员划分，并跑通三个模型的训练与验证。默认 smoke 实验各训练 5 轮、使用一个随机种子；它验证整个流程能够运行，不能作为正式性能排名或最终测试结果。

三个模型均已完成 5 轮 smoke 训练，验证记录与图表见 [本次运行结果](reports/smoke/README.md)。官方测试集尚未用于预测或评分。

## 数据与实验范围

- 数据集：[Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)。
- 30 名受试者，6 类活动，10,299 个窗口；官方训练部分为 7,352 个窗口，测试部分为 2,947 个窗口。
- 每个样本为 `128 × 9` 的预处理时序窗口，采样率为 50 Hz，窗口覆盖 2.56 秒，相邻窗口重叠 50%。这里的输入不是未处理的连续原始信号。
- 通道依次为三轴身体加速度、三轴角速度、三轴总加速度，具体排列顺序记录在处理后的元数据中。本项目读取官方已经滤波、分窗的 `Inertial Signals` 序列，并进一步使用训练统计量做标准化。**不要把数据包的 561 维人工特征当作时间轴，也不要把这些窗口称为完全原始的传感器波形。**
- 标签对应 walking、walking upstairs、walking downstairs、sitting、standing、laying。
- 保留官方 9 人测试集；在官方 21 人训练部分内部，以固定划分种子 `2026` 选择 17 人训练、4 人验证。模型训练种子与人员划分种子相互独立。
- 仅使用训练部分拟合逐通道标准化参数。训练、验证和测试中的受试者互不重叠。

当前固定划分的实际大小如下：

| 部分 | 受试者数量 | 窗口数量 | 用途 |
| --- | ---: | ---: | --- |
| 训练 | 17 | 5,844 | 拟合标准化参数和模型权重 |
| 验证 | 4 | 1,508 | 选择模型、训练轮数和超参数 |
| 官方测试 | 9 | 2,947 | 配置冻结后的最终评价 |

数据使用 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 许可。下载脚本保留来源和文件校验信息，原始数据、处理后的数组和模型权重不放入 Git 仓库。

## 环境

参考环境为 Python 3.10、PyTorch 2.6.0、NumPy 1.23.5 和 Matplotlib 3.7.2。项目支持 CPU；带 CUDA 的 NVIDIA 显卡可加速训练。不需要安装 scikit-learn。

已有可用环境时，可以直接激活后检查：

```powershell
conda activate d2l_torch
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

新环境示例：

```powershell
conda create -n bddl_har python=3.10 -y
conda activate bddl_har
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

上面的 PyTorch 命令用于 CUDA 12.4 版本；只使用 CPU 时，将索引地址改为 `https://download.pytorch.org/whl/cpu`。依赖安装命令应在项目根目录执行。安装方式依据 [PyTorch 官方历史版本说明](https://pytorch.org/get-started/previous-versions/)。

## 下载和准备数据

```powershell
python -m har_compare.prepare --data-dir data
```

首次执行从 UCI 官方站点下载数据，并生成处理后的数据文件和 `data/processed/metadata.json`。重复执行会复用本地数据，不覆盖已有训练结果。元数据保存实际人员名单、通道顺序、标准化参数和数据来源校验信息，作为所有模型共用的数据依据。检查点的数据指纹绑定这些实验内容；单纯改变生成时间或重新封装同一数据，不会让检查点失效。

## 一键跑通三个模型

在项目根目录执行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_smoke.ps1 -Python python -Device cuda
```

如果没有 CUDA，使用 `-Device cpu`；默认 `-Device auto` 自动选择。`-Python` 也可以传入 Python 可执行文件的完整路径。脚本按 CNN、RNN、LSTM 的顺序运行，任何一步失败都会立即停止。

为了保留已有实验，训练默认拒绝覆盖已有输出。重复运行时优先使用新的 `-OutputRoot`；确需替换原有输出时，为批量脚本添加 `-Overwrite`，或为单模型训练添加 `--overwrite`。

单个模型的等价命令：

```powershell
python -m har_compare.train --model cnn --data-dir data/processed --output-dir runs/smoke/cnn --epochs 5 --seed 42 --batch-size 64 --device cuda --lr 0.001 --patience 8 --stage smoke
```

将 `--model` 改为 `rnn` 或 `lstm`，并同时更换输出目录即可运行另外两个模型。默认训练命令只读取训练和验证部分，**不会用官方测试集选模型**。

## 模型与公平比较

| 模型 | 起步结构 | 可训练参数量 |
| --- | --- | ---: |
| CNN | 三层一维卷积，通道 32、64、64；核大小 5；前两层后接池化；全局平均池化 | 32,710 |
| RNN | 单层普通 RNN，隐藏维度 176；对输出序列做平均池化 | 33,974 |
| LSTM | 单层 LSTM，隐藏维度 80；对输出序列做平均池化 | 29,606 |

三个模型均使用 0.2 的分类前 dropout 和 6 类线性分类头。参数量处于相近规模，但参数量相近并不意味着表达能力或计算量完全相同，报告中仍需单独比较效率。

训练共用 batch size 64、Adam 优化器、梯度裁剪和验证集 Macro-F1 选优规则。每轮记录训练损失、验证损失、验证准确率和 Macro-F1。提前停止只基于验证表现。不要因为某个模型在 smoke 阶段暂时较低，就只对它增加训练预算而保留其他模型的短训练结果用于比较。

## 输出文件

每次训练的输出目录包含：

- `history.csv`：逐轮训练与验证记录。
- `config.json`：训练配置、数据来源、人员划分和标准化信息。
- `metrics.json`：最优验证轮次的指标、参数量、设备、运行时间和人员划分等信息。
- `best.pt`：按验证集 Macro-F1 选出的检查点。
- `learning_curves.png`：训练过程曲线。
- `confusion_matrix.png`：验证集混淆矩阵。

图表和初步结果应注明“验证集”“smoke”“训练轮数”和“随机种子”。正式报告需要补充多种子结果；不能把重复随机种子的标准差解释为对所有受试者总体的置信区间。

完成训练后，可以导出适合提交 GitHub 的轻量结果：

```powershell
python -m har_compare.summarize --runs-root runs/smoke --output-dir reports/smoke
```

导出内容包括指标汇总、逐轮记录、图表、数据说明和结果 README，不包含数据数组或模型权重。不同实验阶段和配置使用独立的导出目录，避免混合比较。

## 正式实验的运行入口

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_formal.ps1 -Python python -Device cuda
```

默认对三个模型分别使用 `42、43、44` 三个训练种子，最多训练 50 轮，并在验证表现停止改善时提前停止。这一入口仍然只进行训练和验证；运行前应完成等预算的验证集调参。如果不同模型选出了不同学习率，请分别使用训练命令记录各自冻结后的配置，不要直接把统一默认学习率的九次训练当成已经完成公平调参。

可以单独重载已保存的模型，在验证集复核结果：

```powershell
python -m har_compare.evaluate --checkpoint runs/smoke/cnn/best.pt --data-dir data/processed --split val --device cuda --output runs/smoke/cnn/recheck_val.json
```

评估检查点默认使用验证集，官方测试评估须显式指定 `--split test --allow-test`。只有在结构、超参数和评价方案确定后，才进行官方测试评估。完整约定见 [实验协议](docs/EXPERIMENT_PROTOCOL.md)。

## 检查项目

```powershell
python -m unittest discover -s tests -v
```

检查覆盖分类指标的手算例、三个模型的输出与反向传播、数据指纹的稳定性，以及真实数据准备完成后的人员隔离、标准化和缓存完整性。尚未下载数据时，数据检查会跳过；数据准备完成后应重新运行，确认不再跳过。

## 项目结构

```text
har_compare/        数据准备、模型、训练和评估代码
scripts/            Windows 批量运行入口
tests/              关键正确性检查
docs/               实验协议
data/               本地数据与处理缓存，不提交 Git
runs/               本地训练输出和权重，不提交 Git
reports/            可提交的轻量实验汇总
```

## 数据引用

Reyes-Ortiz, J., Anguita, D., Ghio, A., Oneto, L., & Parra, X. (2013). *Human Activity Recognition Using Smartphones* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C54S4K.

Anguita, D., Ghio, A., Oneto, L., Parra, X., & Reyes-Ortiz, J. L. (2013). *A Public Domain Dataset for Human Activity Recognition Using Smartphones*. ESANN 2013.
