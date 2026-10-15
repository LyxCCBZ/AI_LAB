# -*- coding: utf-8 -*-
"""根据 results/ 下的实验结果生成报告用图表（figures/*.png）。"""
import os
import re
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 中文字体：优先使用微软雅黑
for cand in ["Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC"]:
    try:
        font_manager.findfont(font_manager.FontProperties(family=cand), fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [cand]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"


def parse_params(s):
    s = str(s)
    try:
        d = json.loads(s)
        if isinstance(d, dict):
            return {str(k): v for k, v in d.items()}
    except Exception:
        pass
    out = {}
    for kv in str(s).split(","):
        kv = kv.strip()
        if not kv or "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--figures", required=True)
    args = ap.parse_args()
    os.makedirs(args.figures, exist_ok=True)

    val = pd.read_csv(os.path.join(args.results, "val_results.csv"))
    feat = pd.read_csv(os.path.join(args.results, "feature_results.csv"))
    with open(os.path.join(args.results, "summary.json"), encoding="utf-8") as f:
        summary = json.load(f)

    # 1) 模型与超参数对比（阶段一），只画单模型，按算法分组着色
    val_single = val[val["model"] != "ENSEMBLE"].copy()
    val_single["label"] = val_single.apply(
        lambda r: r["model"] + " " + ", ".join("%s=%s" % (k, v) for k, v in sorted(parse_params(r["params"]).items())
                                               if k in ("alpha", "C", "kernel", "gamma", "hidden_layer_sizes")), axis=1)
    val_single = val_single.sort_values("val_accuracy", ascending=False).reset_index(drop=True)
    colors = {"NB": "#4C72B0", "LR": "#DD8452", "SVM": "#55A868", "MLP": "#C44E52"}
    fig, ax = plt.subplots(figsize=(9, 4.2))
    bars = ax.barh(val_single["label"][::-1], val_single["val_accuracy"][::-1],
                   color=[colors.get(m, "#8172B3") for m in val_single["model"][::-1]])
    ax.set_xlabel("验证集准确率")
    ax.set_xlim(0.6, 1.0)
    ax.set_title("阶段一：各算法与超参数的验证集准确率")
    for b, v in zip(bars, val_single["val_accuracy"][::-1]):
        ax.text(v + 0.003, b.get_y() + b.get_height() / 2, "%.4f" % v, va="center", fontsize=8)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()],
              labels=list(colors.keys()), loc="lower right", fontsize=8)
    fig.savefig(os.path.join(args.figures, "stage1_model_comparison.png"))
    plt.close(fig)

    # 2) 调参曲线：C / alpha 对验证集准确率的影响
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, (model, key, xlabel) in zip(axes, [("NB", "alpha", "alpha"), ("LR", "C", "C"), ("SVM", "C", "C")]):
        sub = val[val["model"] == model].copy()
        sub["x"] = sub["params"].apply(lambda s: float(parse_params(s).get(key, float("nan"))))
        sub = sub[sub["kernel"].isna()] if "kernel" in sub else sub
        sub = sub.dropna(subset=["x"]).sort_values("x")
        sub = sub[~sub["params"].str.contains("kernel=rbf")]
        if len(sub):
            ax.plot(sub["x"], sub["val_accuracy"], "o-", color=colors[model])
            for _, r in sub.iterrows():
                ax.annotate("%.4f" % r["val_accuracy"], (r["x"], r["val_accuracy"]),
                            textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)
        ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("验证集准确率")
        ax.set_title("%s 调参曲线" % model)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.figures, "stage1_tuning_curves.png"))
    plt.close(fig)

    # 3) 特征工程对比（阶段二）
    feat_sorted = feat.sort_values("val_accuracy", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    bars = ax.barh([s.replace("（", "(").replace("）", ")") for s in feat_sorted["features"][::-1]],
                   feat_sorted["val_accuracy"][::-1], color="#4C72B0")
    ax.set_xlabel("验证集准确率")
    ax.set_xlim(0.6, 1.0)
    ax.set_title("阶段二：TF-IDF 特征配置对比")
    for b, v in zip(bars, feat_sorted["val_accuracy"][::-1]):
        ax.text(v + 0.003, b.get_y() + b.get_height() / 2, "%.4f" % v, va="center", fontsize=8)
    fig.savefig(os.path.join(args.figures, "stage2_feature_comparison.png"))
    plt.close(fig)

    # 4) 验证集混淆矩阵（最优单模型）
    # summary.json 里存了矩阵；若缺失（例如换了别的实现），回退到 best_confusion.csv
    if summary.get("best_val_confusion_matrix"):
        cm = np.array(summary["best_val_confusion_matrix"])
    else:
        cm = np.loadtxt(os.path.join(args.results, "best_confusion.csv"), delimiter=",", dtype=int)
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("预测类别"); ax.set_ylabel("真实类别")
    ax.set_title("最优模型验证集混淆矩阵")
    ax.set_xticks(range(cm.shape[0])); ax.set_yticks(range(cm.shape[0]))
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if cm[i, j] > cm.max() * 0.6 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.savefig(os.path.join(args.figures, "confusion_matrix.png"))
    plt.close(fig)

    # 5) 每类 precision / recall / F1
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    x = np.arange(len(tp))
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.bar(x - 0.27, prec, 0.27, label="precision")
    ax.bar(x, rec, 0.27, label="recall")
    ax.bar(x + 0.27, f1, 0.27, label="F1")
    ax.set_xticks(x)
    ax.set_ylim(0.6, 1.02)
    ax.set_xlabel("类别"); ax.set_ylabel("指标值")
    ax.set_title("最优模型各类别指标（验证集）")
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(args.figures, "per_class_metrics.png"))
    plt.close(fig)

    print("figures written to", args.figures)


if __name__ == "__main__":
    main()
