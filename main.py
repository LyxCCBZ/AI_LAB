# -*- coding: utf-8 -*-
"""当代人工智能 实验一：文本分类（20 Newsgroups 10 分类）。

固定随机种子，把训练集按类别分层划出验证集，比较不同特征配置与模型/超参数，
选验证集最优的方案用全部有标签数据重训，预测无标签测试集并写出 prediction.csv。

用法：python main.py --data . --out results
"""
import os
import re
import json
import time
import argparse

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# 固定随机种子，保证结果可复现
SEED = 42
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

N_CLASS = 10  # 数据集共有 10 个类别（0~9）

# 可选的文本清洗：去掉邮件/新闻组头、引用行与邮箱地址，只保留正文
HEADER_RE = re.compile(
    r"^(from|subject|organization|lines|nntp-posting-host|reply-to|distribution|"
    r"x-newsreader|keywords|summary|in-reply-to|article-i\.d\.|sender|newsgroups|"
    r"followup-to|expires|xref|references|message-id|path|return-path):",
    re.I,
)
EMAIL_RE = re.compile(r"\S+@\S+")


def clean_text(text):
    """去掉邮件/新闻组头、引用行与邮箱地址，只保留正文（可选的数据清洗）。"""
    keep = []
    for line in str(text).split("\n"):
        if HEADER_RE.match(line.strip()):
            continue
        if line.lstrip().startswith(">"):
            continue
        keep.append(line)
    return EMAIL_RE.sub(" ", "\n".join(keep))


def load_data(data_dir="."):
    """读取原始 csv，返回文本与标签（保持实验提供的数据加载方式）。"""
    train_df = pd.read_csv(os.path.join(data_dir, "train_data.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test_data_unlabeled.csv"))
    X_train = train_df["text"].astype(str).tolist()
    y_train = train_df["target"].values.astype(int)
    X_test_unlabeled = test_df["text"].astype(str).tolist()
    return X_train, y_train, X_test_unlabeled


def split_train_val(X_train, y_train, val_size=0.2, seed=SEED):
    """分层划分验证集：每个类别都按比例抽取，保证各类分布一致。"""
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_train, y_train, test_size=val_size, random_state=seed, stratify=y_train, shuffle=True
    )
    return X_tr, X_va, y_tr, y_va


def get_vectorizer(cfg):
    """根据配置创建 TF-IDF 向量化器（默认参数与课堂示例一致，便于对比）。"""
    return TfidfVectorizer(
        lowercase=True,
        preprocessor=(clean_text if cfg.get("clean", False) else None),
        ngram_range=cfg.get("ngram_range", (1, 1)),
        min_df=cfg.get("min_df", 2),
        max_features=cfg.get("max_features", 20000),
        sublinear_tf=cfg.get("sublinear_tf", True),
        strip_accents=cfg.get("strip_accents", None),
        stop_words=cfg.get("stop_words", None),
        token_pattern=r"(?u)\b\w\w+\b",
        dtype=np.float64,
    )


def build_model(name, params, seed=SEED):
    """根据名字与超参数创建模型。"""
    p = dict(params)
    if name == "NB":
        return MultinomialNB(**p)
    if name == "LR":
        p.setdefault("max_iter", 2000)
        p.setdefault("random_state", seed)
        return LogisticRegression(**p)
    if name == "SVM":
        p.setdefault("random_state", seed)
        if p.get("kernel", "linear") == "linear":
            # 线性核改用 liblinear 的 LinearSVC（一对多平方铰链损失）：
            # 高维稀疏 TF-IDF 上比 libsvm 的 SVC(kernel="linear") 快一到两个数量级
            p.pop("kernel", None)
            p.setdefault("dual", "auto")
            p.setdefault("max_iter", 5000)
            return LinearSVC(**p)
        return SVC(**p)
    if name == "MLP":
        p.setdefault("random_state", seed)
        return MLPClassifier(**p)
    raise ValueError("unknown model: %s" % name)


def model_proba(model, X):
    """返回 (n_samples, n_classes) 的类别概率。

    LogisticRegression / MultinomialNB / MLP 自带 predict_proba；
    SVC 默认 probability=False，没有概率输出，这里对 decision_function
    的决策值做 softmax，得到可比较的概率，用于软投票集成。
    """
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X), dtype=np.float64)
    scores = np.asarray(model.decision_function(X), dtype=np.float64)
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    scores = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def fmt_params(params):
    return ", ".join("%s=%s" % (k, v) for k, v in sorted(params.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=".", help="数据目录（含 train_data.csv / test_data_unlabeled.csv）")
    ap.add_argument("--out", default="results", help="结果输出目录")
    ap.add_argument("--val-size", type=float, default=0.2, help="验证集比例")
    ap.add_argument("--seed", type=int, default=SEED, help="随机种子")
    ap.add_argument("--quick", action="store_true", help="快速模式（更少的参数组合）")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    np.random.seed(args.seed)
    t_start = time.time()
    log_lines = []

    def log(msg=""):
        print(msg, flush=True)
        log_lines.append(str(msg))

    log("=" * 84)
    log("实验一 文本分类：20 Newsgroups 10 分类  (seed=%d, val_size=%.2f)" % (args.seed, args.val_size))
    log("=" * 84)

    # 1. 读取数据
    X_train_all, y_train_all, X_test_unlabeled = load_data(args.data)
    log("训练集(有标签): %d 条    测试集(无标签): %d 条    类别数: %d"
        % (len(X_train_all), len(X_test_unlabeled), N_CLASS))
    log("训练集类别分布: %s" % np.bincount(y_train_all, minlength=N_CLASS).tolist())

    # 2. 划出验证集
    X_tr, X_va, y_tr, y_va = split_train_val(X_train_all, y_train_all, args.val_size, args.seed)
    log("分层划分结果: 训练 %d 条 / 验证 %d 条" % (len(X_tr), len(X_va)))
    log("验证集类别分布: %s" % np.bincount(y_va, minlength=N_CLASS).tolist())

    # 3. 阶段一：算法与超参数对比
    base_cfg = {"clean": False, "ngram_range": (1, 1), "min_df": 2, "max_features": 20000, "sublinear_tf": True}
    vec = get_vectorizer(base_cfg)
    X_tr_tfidf = vec.fit_transform(X_tr)          # 只在训练集上拟合词表与 IDF
    X_va_tfidf = vec.transform(X_va)
    log("")
    log("阶段一：基准 TF-IDF 词表大小 = %d，训练/验证非零特征 = %d / %d"
        % (len(vec.vocabulary_), X_tr_tfidf.nnz, X_va_tfidf.nnz))

    if args.quick:
        grid = [
            ("NB", {"alpha": 1.0}),
            ("LR", {"C": 10.0}),
            ("SVM", {"kernel": "linear", "C": 1.0}),
        ]
    else:
        grid = [
            ("NB", {"alpha": 0.1}), ("NB", {"alpha": 1.0}), ("NB", {"alpha": 5.0}),
            ("LR", {"C": 1.0}), ("LR", {"C": 10.0}), ("LR", {"C": 50.0}),
            ("SVM", {"kernel": "linear", "C": 0.1}),
            ("SVM", {"kernel": "linear", "C": 1.0}),
            ("SVM", {"kernel": "linear", "C": 10.0}),
            ("SVM", {"kernel": "rbf", "C": 10.0, "gamma": "scale"}),
            ("MLP", {"hidden_layer_sizes": (100,), "alpha": 1e-4, "max_iter": 300}),
        ]

    log("")
    log("-" * 84)
    log("阶段一：不同算法 / 超参数在验证集上的表现（特征固定）")
    log("-" * 84)
    rows, fitted, params_of = [], {}, {}
    for name, params in grid:
        t0 = time.time()
        model = build_model(name, params, args.seed)
        model.fit(X_tr_tfidf, y_tr)
        pred_va = model.predict(X_va_tfidf)
        acc = accuracy_score(y_va, pred_va)
        f1m = f1_score(y_va, pred_va, average="macro")
        dur = time.time() - t0
        rows.append({"model": name, "params": fmt_params(params),
                     "val_accuracy": round(acc, 4), "val_macro_f1": round(f1m, 4),
                     "seconds": round(dur, 1)})
        fitted[(name, fmt_params(params))] = (model, pred_va)
        params_of[(name, fmt_params(params))] = params
        log("  %-5s %-52s val_acc=%.4f  macroF1=%.4f  (%.1fs)" % (name, fmt_params(params), acc, f1m, dur))

    # 软投票集成：取每个家族中验证集最好的一个模型，概率平均
    best_per_family = {}
    for (name, pstr), (model, pred) in fitted.items():
        acc = accuracy_score(y_va, pred)
        if name not in best_per_family or acc > best_per_family[name][0]:
            best_per_family[name] = (acc, pstr, model)
    if len(best_per_family) >= 2:
        members = [(name, v[2]) for name, v in sorted(best_per_family.items())]
        proba = np.zeros((len(y_va), N_CLASS))
        for name, model in members:
            proba += model_proba(model, X_va_tfidf)
        proba /= len(members)
        pred_ens = proba.argmax(axis=1)
        rows.append({"model": "ENSEMBLE", "params": "members=" + "+".join(n for n, _ in members),
                     "val_accuracy": round(accuracy_score(y_va, pred_ens), 4),
                     "val_macro_f1": round(f1_score(y_va, pred_ens, average="macro"), 4),
                     "seconds": 0.0})
        log("  %-5s %-52s val_acc=%.4f  macroF1=%.4f"
            % ("ENS", "members=" + "+".join(n for n, _ in members),
               accuracy_score(y_va, pred_ens), f1_score(y_va, pred_ens, average="macro")))

    df_val = pd.DataFrame(rows).sort_values("val_accuracy", ascending=False).reset_index(drop=True)
    df_val.to_csv(os.path.join(args.out, "val_results.csv"), index=False, encoding="utf-8-sig")
    log("")
    log("阶段一排名前 5：")
    for _, r in df_val.head(5).iterrows():
        log("  %-8s %-52s %.4f" % (r["model"], r["params"], r["val_accuracy"]))

    best_row = df_val.iloc[0]
    best_model_name = best_row["model"]
    if best_model_name == "ENSEMBLE":
        probe_name = max(best_per_family, key=lambda k: best_per_family[k][0])
        probe_params = params_of[(probe_name, best_per_family[probe_name][1])]
    else:
        probe_name = best_row["model"]
        probe_params = params_of[(probe_name, best_row["params"])]

    # 4. 阶段二：特征工程对比
    log("")
    log("-" * 84)
    log("阶段二：特征工程配置对性能的影响（模型固定为阶段一最优的单模型 %s）" % probe_name)
    log("-" * 84)
    feature_cfgs = [
        ("基准 unigram + max_features=20000", {"ngram_range": (1, 1), "min_df": 2, "max_features": 20000, "sublinear_tf": True}),
        ("max_features=5000", {"ngram_range": (1, 1), "min_df": 2, "max_features": 5000, "sublinear_tf": True}),
        ("min_df=1, max_features=50000", {"ngram_range": (1, 1), "min_df": 1, "max_features": 50000, "sublinear_tf": True}),
        ("unigram+bigram (ngram_range=(1,2))", {"ngram_range": (1, 2), "min_df": 3, "max_features": 50000, "sublinear_tf": True}),
        ("sublinear_tf=False（原始词频）", {"ngram_range": (1, 1), "min_df": 2, "max_features": 20000, "sublinear_tf": False}),
        ("去掉邮件头与引用行", {"clean": True, "ngram_range": (1, 1), "min_df": 2, "max_features": 20000, "sublinear_tf": True}),
    ]
    feat_rows = []
    for label, cfg in feature_cfgs:
        t0 = time.time()
        v = get_vectorizer(cfg)
        A = v.fit_transform(X_tr)
        B = v.transform(X_va)
        m = build_model(probe_name, probe_params, args.seed)
        m.fit(A, y_tr)
        pred = m.predict(B)
        acc = accuracy_score(y_va, pred)
        f1m = f1_score(y_va, pred, average="macro")
        feat_rows.append({"features": label, "vocab_size": len(v.vocabulary_),
                          "val_accuracy": round(acc, 4), "val_macro_f1": round(f1m, 4),
                          "seconds": round(time.time() - t0, 1)})
        log("  %-36s vocab=%-7d val_acc=%.4f  macroF1=%.4f  (%.1fs)"
            % (label, len(v.vocabulary_), acc, f1m, time.time() - t0))
    df_feat = pd.DataFrame(feat_rows)
    df_feat.to_csv(os.path.join(args.out, "feature_results.csv"), index=False, encoding="utf-8-sig")
    best_feat = df_feat.sort_values("val_accuracy", ascending=False).iloc[0]
    best_feat_cfg = dict(feature_cfgs[[c[0] for c in feature_cfgs].index(best_feat["features"])][1])
    log("阶段二最优特征配置: %s (val_acc=%.4f)" % (best_feat["features"], best_feat["val_accuracy"]))

    # 5. 阶段三：用最优方案在全部有标签数据上重训并预测
    log("")
    log("-" * 84)
    log("阶段三：用最优配置在 训练+验证 全部有标签数据上重训，并预测无标签测试集")
    log("-" * 84)

    # 保存最优单模型在验证集上的详细评估（混淆矩阵 + 分类报告）
    cm = None
    if probe_name in best_per_family:
        _, pstr, model_best = best_per_family[probe_name]
        pred_best_va = model_best.predict(X_va_tfidf)
        cm = confusion_matrix(y_va, pred_best_va, labels=list(range(N_CLASS)))
        np.savetxt(os.path.join(args.out, "best_confusion.csv"), cm, fmt="%d", delimiter=",")
        report = classification_report(y_va, pred_best_va, digits=4)
        with open(os.path.join(args.out, "classification_report.txt"), "w", encoding="utf-8") as f:
            f.write("最优单模型: %s (%s)\n验证集准确率: %.4f, macro-F1: %.4f\n\n%s"
                    % (probe_name, pstr, accuracy_score(y_va, pred_best_va),
                       f1_score(y_va, pred_best_va, average="macro"), report))
        log("最优单模型 %s 的验证集分类报告已保存到 %s" % (probe_name, args.out))

    final_vec = get_vectorizer(best_feat_cfg)
    X_all = final_vec.fit_transform(X_train_all)          # 使用全部有标签数据拟合词表与 IDF
    X_test = final_vec.transform(X_test_unlabeled)
    final_model = build_model(probe_name, probe_params, args.seed)
    t0 = time.time()
    final_model.fit(X_all, y_train_all)
    predictions = final_model.predict(X_test)
    log("最终模型 %s(%s) 使用 %d 条有标签样本重训完成，用时 %.1fs"
        % (probe_name, fmt_params(probe_params), len(X_train_all), time.time() - t0))
    log("测试集预测类别分布: %s" % np.bincount(predictions, minlength=N_CLASS).tolist())

    pred_path = os.path.join(".", "prediction.csv")
    pd.DataFrame(predictions).to_csv(pred_path, index=False, header=False)
    log("已保存提交文件: %s （%d 行，无表头）" % (os.path.abspath(pred_path), len(predictions)))

    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "seed": args.seed, "val_size": args.val_size,
            "n_train": len(X_train_all), "n_test_unlabeled": len(X_test_unlabeled),
            "n_class": N_CLASS,
            "split": {"train": len(X_tr), "val": len(X_va)},
            "stage1_best": {"model": best_model_name, "params": best_row["params"],
                            "val_accuracy": float(best_row["val_accuracy"]),
                            "val_macro_f1": float(best_row["val_macro_f1"])},
            "stage2_best": {"features": best_feat["features"],
                            "val_accuracy": float(best_feat["val_accuracy"]),
                            "vocab_size": int(best_feat["vocab_size"]),
                            "config": {k: (list(v) if isinstance(v, tuple) else v)
                                       for k, v in best_feat_cfg.items()}},
            "final_model": {"name": probe_name, "params": probe_params},
            "best_val_confusion_matrix": cm.tolist() if cm is not None else [],
            "test_prediction_distribution": np.bincount(predictions, minlength=N_CLASS).tolist(),
            "val_results": rows, "feature_results": feat_rows,
            "total_seconds": round(time.time() - t_start, 1),
        }, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out, "run_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    log("")
    log("全部完成，总耗时 %.1fs" % (time.time() - t_start))


if __name__ == "__main__":
    main()
