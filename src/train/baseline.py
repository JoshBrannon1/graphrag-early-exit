"""Logistic regression on hand-built prefix features.

Trains on train, evaluates on valid. Never reads test.

Usage: python -m src.train.baseline [--label label_looping] [--early-t 10]
"""

import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data.build_dataset import DEFAULT_OUT_DIR
from src.features import featurize

FEATURIZER = "hand"
TRIVIAL_FEATURES = ("duplicate_action_rate", "t")
MIN_CLASS_COUNT = 20
SEED = 0


def load_split(out_dir, split, label, featurizer):
    """Return (feature_names, X, y, t). Labels are read from the row separately."""
    feats, labels = [], []
    with (out_dir / f"instances_{split}.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            feats.append(featurize(featurizer, row))
            labels.append(bool(row[label]))
    names = list(feats[0])
    X = np.array([[d[n] for n in names] for d in feats], dtype=float)
    return names, X, np.array(labels), X[:, names.index("t")]


def fit(X, y, seed):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed),
    )
    return model.fit(X, y)


def scores(y, score):
    return average_precision_score(y, score), roc_auc_score(y, score)


def subset_header(trivial_names):
    trivial = " | ".join(f"{n} PR-AUC | {n} ROC-AUC" for n in trivial_names)
    return f"subset | n | pos_rate | model PR-AUC | model ROC-AUC | {trivial}"


def subset_line(name, y, model_score, trivial_scores):
    cols = [f"{v:.3f}" for s in (model_score, *trivial_scores) for v in scores(y, s)]
    return " | ".join([name, str(len(y)), f"{y.mean():.3f}", *cols])


def per_t_lines(y, t, model_score, dup_score, max_t, min_class):
    """One line per t in 1..max_t; skips t where either class has < min_class."""
    lines = ["t | n | pos_rate | model ROC-AUC | duplicate_action_rate ROC-AUC"]
    for k in range(1, max_t + 1):
        mask = t == k
        n_pos, n_neg = int(y[mask].sum()), int((~y[mask]).sum())
        if min(n_pos, n_neg) < min_class:
            lines.append(f"{k} | skipped (pos={n_pos}, neg={n_neg})")
            continue
        lines.append(
            f"{k} | {mask.sum()} | {y[mask].mean():.3f} | "
            f"{roc_auc_score(y[mask], model_score[mask]):.3f} | "
            f"{roc_auc_score(y[mask], dup_score[mask]):.3f}"
        )
    return lines


def coefficients(model, names):
    coef = model[-1].coef_[0]
    return sorted(zip(names, coef), key=lambda nc: -abs(nc[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="label_looping")
    parser.add_argument("--early-t", type=int, default=10)
    parser.add_argument("--data-dir", type=type(DEFAULT_OUT_DIR), default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    names, X_tr, y_tr, _ = load_split(args.data_dir, "train", args.label, FEATURIZER)
    names_va, X_va, y_va, t_va = load_split(args.data_dir, "valid", args.label, FEATURIZER)
    assert names == names_va

    model = fit(X_tr, y_tr, SEED)
    model_score = model.predict_proba(X_va)[:, 1]
    trivial = [X_va[:, names.index(n)] for n in TRIVIAL_FEATURES]
    early = t_va <= args.early_t

    print(f"label={args.label}  train n={len(y_tr)} pos_rate={y_tr.mean():.3f}")
    print(subset_header(TRIVIAL_FEATURES))
    print(subset_line("valid_all", y_va, model_score, trivial))
    print(subset_line(f"valid_t<={args.early_t}", y_va[early], model_score[early], [s[early] for s in trivial]))
    print()
    dup = X_va[:, names.index("duplicate_action_rate")]
    print("\n".join(per_t_lines(y_va, t_va, model_score, dup, args.early_t, MIN_CLASS_COUNT)))
    print("\ncoefficients (standardized), by |coef|:")
    for name, c in coefficients(model, names):
        print(f"  {name:<26s} {c:+.3f}")
    print(f"  {'intercept':<26s} {model[-1].intercept_[0]:+.3f}")


if __name__ == "__main__":
    main()
