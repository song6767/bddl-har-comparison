# CNN 与 RNN 的人体活动识别比较

本项目比较一维卷积网络（CNN）、普通循环网络（RNN）和长短期记忆网络（LSTM）在 UCI HAR 人体活动识别任务中的分类表现、训练效率与稳定性，用于 BDDL 课程第三项作业。

公开仓库：[song6767/bddl-har-comparison](https://github.com/song6767/bddl-har-comparison)。仓库保存本项目的 PyTorch 实现、运行脚本、实验协议和轻量结果汇总；数据从 UCI 官方来源单独下载，数据集作者信息与许可见下文。

三个模型各在验证集上尝试 3 个学习率，再用选定的学习率分别训练 3 个随机种子。冻结 9 个检查点后，在同一官方测试集上统一评分。[正式对比结果与图表](reports/formal/README.md)

在同一官方测试集上，使用这 9 个检查点比较 20 dB 和 10 dB 人工噪声造成的分数下降。[噪声鲁棒性结果与图表](reports/noise/README.md)

各模型训练 5 轮、使用单个种子的流程试跑另有记录。[试跑结果与图表](reports/smoke/README.md)

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
| 验证 | 4 | 1,508 | 为每个架构选择学习率与最佳检查点 |
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

训练共用 batch size 64、Adam 优化器、梯度裁剪和验证集 Macro-F1 选优规则。每轮记录训练损失、验证损失、验证准确率和 Macro-F1。早停依据验证表现，各模型共用相同的最大轮数和耐心值。

## 输出文件

每次训练的输出目录包含：

- `history.csv`：逐轮训练与验证记录。
- `config.json`：训练配置、数据来源、人员划分和标准化信息。
- `metrics.json`：最优验证轮次的指标、参数量、设备、运行时间和人员划分等信息。
- `best.pt`：按验证集 Macro-F1 选出的检查点。
- `learning_curves.png`：训练过程曲线。
- `confusion_matrix.png`：验证集混淆矩阵。

试跑图表对应单种子的验证集结果。正式报告按模型汇总三个种子的官方测试分数及样本标准差。

完成训练后，可以导出适合提交 GitHub 的轻量结果：

```powershell
python -m har_compare.summarize --runs-root runs/smoke --output-dir reports/smoke
```

导出内容包括指标汇总、逐轮记录、图表、数据说明和结果 README，不包含数据数组或模型权重。不同实验阶段和配置使用独立的导出目录，避免混合比较。

## 复现正式实验

在项目根目录准备数据后，依次运行以下命令。已有同名实验目录时，为新实验指定不同的 `--root` 和 `--output`，以便保留原始记录。没有 CUDA 时，可在带 `--device` 的步骤中统一使用 `cpu`。

```powershell
python -m har_compare.formal_train tune --root runs/formal_v1 --device cuda
python -m har_compare.formal_train train --root runs/formal_v1 --device cuda
python -m har_compare.formal_test freeze --root runs/formal_v1
python -m har_compare.formal_test evaluate --root runs/formal_v1 --device cuda
python -m har_compare.formal_report --root runs/formal_v1 --output reports/formal
```

1. **调参**。三个模型分别用相同的种子 `41` 试验 `0.0003`、`0.001`、`0.003`，合计 9 次训练。每次最多 50 轮，验证集 Macro-F1 连续 8 轮未改善则早停。对每个模型比较三个学习率各自的最佳验证集 Macro-F1，使用未四舍五入的分数选最高者，同分取较小的学习率。
2. **正式训练**。三个模型用各自选定的学习率分别从头训练 `42`、`43`、`44` 三个种子，合计 9 次。按验证集 Macro-F1 保留每次的最佳检查点。
3. **冻结与测试**。先记录全部配置和检查点的校验值，再验证清单并在同一官方测试集上评分。每个种子均使用相同的 9 位测试受试者。
4. **汇总**。导出调参记录、逐种子测试指标、对比图表和冻结清单到 `reports/formal/`。数据与权重保存在本地的 `data/` 和 `runs/`。

测试结果中的均值和样本标准差描述三个训练种子的表现与波动。三次评估共用同一组受试者，因此种子波动不能当成跨受试者置信区间。评价范围为 UCI HAR 的官方测试条件。完整设置见 [实验协议](docs/EXPERIMENT_PROTOCOL.md)。

可以单独重载已保存的模型，在验证集复核结果：

```powershell
python -m har_compare.evaluate --checkpoint runs/smoke/cnn/best.pt --data-dir data/processed --split val --device cuda --output runs/smoke/cnn/recheck_val.json
```

单检查点复核默认使用验证集。复现正式测试时，使用上面的冻结和批量评估入口。

## 复现噪声鲁棒性实验

完成正式冻结和干净测试后，在同一个本地实验目录中运行以下命令。评估使用与正式测试相同的设备类型。若已生成前两步的记录，可直接运行汇总命令。

```powershell
python -m har_compare.noise_eval plan --root runs/formal_v1
python -m har_compare.noise_eval evaluate --root runs/formal_v1 --device cuda
python -m har_compare.noise_report --input runs/formal_v1/noise_v1/results.json --output reports/noise
```

第一步在评分前写入噪声协议和原冻结清单的校验值。第二步核对 9 个检查点，复核干净测试的分数和混淆矩阵，然后对两档带噪输入评分。同一档的所有检查点使用同一份输入。每次先按检查点计算干净分数减带噪分数，再按模型汇总三个训练种子的均值和样本标准差。下降量的单位为百分点。加噪方法见[实验协议](docs/EXPERIMENT_PROTOCOL.md#人工噪声敏感性)。

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
