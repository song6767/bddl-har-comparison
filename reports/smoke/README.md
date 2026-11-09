# 三个时序模型的验证集运行记录

阶段：`smoke`。所有指标均来自验证集；官方测试集未进行预测或评分。

smoke 阶段仅证明数据、训练、保存和评估流程能够运行。单种子短训练结果不能作为最终模型排名。
验证集同时用于选择最佳轮次，因此这里不是独立测试结果。

| 模型 | 种子 | 完成轮数 | 最佳轮次 | 参数量 | 验证准确率 | 验证 Macro-F1 | 训练流程秒数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CNN | 42 | 5 | 3 | 32,710 | 97.21% | 0.9736 | 1.93 |
| LSTM | 42 | 5 | 5 | 29,606 | 95.42% | 0.9547 | 2.79 |
| RNN | 42 | 5 | 4 | 33,974 | 91.05% | 0.9109 | 2.91 |

设备：NVIDIA GeForce RTX 3050。
训练集 5,844 窗，验证集 1,508 窗。
标准化统计量只使用训练集。不同集合的受试者互不重叠。

训练流程秒数包含每轮验证和最佳检查点写入，不包含数据加载、绘图和启动进程。
显存记录为 PyTorch 分配器的峰值，不包括驱动、上下文与其他程序；推理时间为驻留设备批次的前向计算，
不包含数据传输，按批次均摊的时间不等于单样本延迟。短训练计时仅作流程记录。

## 结果文件

- [汇总 CSV](summary.csv)
- [汇总 JSON](summary.json)
- [数据来源和人员划分](data_manifest.json)
- CNN，种子 42：[完整指标](cnn/seed_42/metrics.json)、[逐轮记录](cnn/seed_42/history.csv)、[训练曲线](cnn/seed_42/learning_curves.png)、[混淆矩阵](cnn/seed_42/confusion_matrix.png)
- LSTM，种子 42：[完整指标](lstm/seed_42/metrics.json)、[逐轮记录](lstm/seed_42/history.csv)、[训练曲线](lstm/seed_42/learning_curves.png)、[混淆矩阵](lstm/seed_42/confusion_matrix.png)
- RNN，种子 42：[完整指标](rnn/seed_42/metrics.json)、[逐轮记录](rnn/seed_42/history.csv)、[训练曲线](rnn/seed_42/learning_curves.png)、[混淆矩阵](rnn/seed_42/confusion_matrix.png)

## 后续实验

先完成相同调参预算下的验证实验，再冻结各模型配置，进行三个随机种子的正式训练。
最终统一评估官方测试集，并按预先确定的方案分析效率和噪声敏感性。
