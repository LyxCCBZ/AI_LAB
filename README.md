# 当代人工智能 实验一：文本分类

20 Newsgroups 10 分类。训练集 7368 条带标签文本，测试集 2457 条无标签文本，训练分类器并预测测试集每条文本的类别。

## 做法

1. 在训练集上按类别分层划出 20% 作验证集（随机种子 42），测试集在选型阶段不参与。
2. 在训练集上拟合 TF-IDF，比较不同特征配置、不同模型与超参数在验证集上的准确率和 macro-F1。
3. 选出验证集最优方案，用全部 7368 条有标签数据重训，预测测试集，写出 prediction.csv。

## 环境

Python 3.10，numpy、pandas、scikit-learn、matplotlib。

## 文件

| 文件 | 说明 |
| --- | --- |
| main.py | 主程序 |
| plot_results.py | 画报告里的图 |
| check_predictions.py | 检查 prediction.csv 的格式 |
| prediction.csv | 预测结果，2457 行、单列、无表头 |
| results/ | 验证集成绩、混淆矩阵、分类报告 |
| figures/ | 报告用图 |
| 实验报告.md | 实验报告的文字稿 |

数据文件 train_data.csv、test_data_unlabeled.csv 与代码放在同一目录（课程附件里已有，仓库不再重复上传）。

## 运行

```
python main.py --data . --out results
python plot_results.py --results results --figures figures
python check_predictions.py prediction.csv
```

main.py 跑完大约 7 分钟，输出 prediction.csv 和 results/ 下的成绩表。

## 结果

验证集 1474 条上，线性 SVM（C=1）最好，0.9417；逻辑回归 C=10 为 0.9396，单隐层 MLP 0.9349，RBF 核 SVM 0.9328，朴素贝叶斯 alpha=0.1 为 0.9166。

特征方面，min_df=2、20000 维的基准 TF-IDF 最好，词表扩到 50000 维没有提升，加 bigram 或去掉 sublinear TF 都会变差。

最终用基准 TF-IDF 加线性 SVM（C=1），在全部训练数据上重训后预测测试集。
