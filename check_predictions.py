# -*- coding: utf-8 -*-
"""检查 prediction.csv 格式：一列、2457 行、取值 0~9。

用法：python check_predictions.py [prediction.csv]
"""
import sys
import pandas as pd

path = sys.argv[1] if len(sys.argv) > 1 else "prediction.csv"
vals = pd.read_csv(path, header=None).iloc[:, 0].astype(int).values
print("行数: %d" % len(vals))
print("取值集合: %s" % sorted(set(vals.tolist())))
print("各类别数量: %s" % [int((vals == k).sum()) for k in range(10)])
if len(vals) == 2457 and vals.min() >= 0 and vals.max() <= 9:
    print("格式正常")
else:
    print("格式有问题，请重新生成")
