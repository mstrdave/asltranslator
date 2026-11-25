"""
train.py - trains the letter classifier on data/landmarks.csv.

Validation is done on whole held-out *takes* (recording sessions), not random
frames - neighbouring frames are near-duplicates, so a random split would
report inflated accuracy.

Outputs:
  models/asl_mlp.npz            weights used by translate.py
  results/confusion_matrix.png  which letters get mixed up
  results/training_curve.png    loss / accuracy per epoch
"""

import argparse
import csv
import os

import numpy as np

from common import DATA_CSV, HERE, MLP, WEIGHTS, augment, features

RESULTS = os.path.join(HERE, "results")


def load(path=DATA_CSV):
    labels, takes, pts = [], [], []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            labels.append(row[0])
            takes.append(int(row[1]))
            pts.append(np.array(row[2:], np.float32).reshape(21, 3))
    return np.array(labels), np.array(takes), np.stack(pts)


def split_by_take(labels, takes, val_frac, rng):
    """Hold out ~val_frac of each letter's takes (at least one take if the letter has 2+)."""
    val = np.zeros(len(labels), bool)
    gap = np.zeros(len(labels), bool)
    for lab in np.unique(labels):
        t = np.unique(takes[labels == lab])
        rng.shuffle(t)
        if len(t) > 1:
            k = max(1, int(round(len(t) * val_frac)))
            val |= np.isin(takes, t[:k]) & (labels == lab)
        else:
            # only one take: hold out the LAST part of it in time (with a 10-frame gap so
            # the frames right at the boundary - near-duplicates - aren't on both sides)
            idx = np.flatnonzero(labels == lab)
            cut = int(len(idx) * (1 - val_frac * 2))  # last ~40% of the recording
            val[idx[cut:]] = True
            gap[idx[max(0, cut - 10):cut]] = True
    return ~val & ~gap, val


def adam_step(params, grads, state, lr, t, b1=0.9, b2=0.999, eps=1e-8):
    for i, (p, g) in enumerate(zip(params, grads)):
        m, v = state.setdefault(i, (np.zeros_like(p), np.zeros_like(p)))
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        state[i] = (m, v)
        p -= lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--reg", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--val", type=float, default=0.2, help="fraction of takes held out")
    ap.add_argument("--data", default=DATA_CSV)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    labels, takes, pts = load(args.data)
    classes = sorted(set(labels))
    y_all = np.array([classes.index(l) for l in labels])
    tr, va = split_by_take(labels, takes, args.val, rng)
    print(f"{len(labels)} samples, {len(classes)} letters, {len(np.unique(takes))} takes "
          f"-> train {tr.sum()} / val {va.sum()}")

    Xva = np.stack([features(p) for p in pts[va]]) if va.any() else None
    model = MLP(73, classes)
    X0 = np.stack([features(p) for p in pts[tr]])
    model.mu, model.sd = X0.mean(axis=0), np.maximum(X0.std(axis=0), 1e-2)  # floor: constant features stay ~0

    params = model.W + model.b
    state, step, hist = {}, 0, []
    idx_tr = np.flatnonzero(tr)
    for ep in range(1, args.epochs + 1):
        rng.shuffle(idx_tr)
        tot = 0.0
        for s in range(0, len(idx_tr), args.batch):
            bi = idx_tr[s:s + args.batch]
            X = np.stack([features(augment(pts[i], rng)) for i in bi])
            loss, gW, gb = model.loss(X, y_all[bi], args.reg)
            step += 1
            adam_step(params, gW + gb, state, args.lr, step)
            tot += loss * len(bi)
        tr_acc = (model.predict_proba(X0).argmax(1) == y_all[tr]).mean()
        va_acc = (model.predict_proba(Xva).argmax(1) == y_all[va]).mean() if Xva is not None else float("nan")
        hist.append((tot / len(idx_tr), tr_acc, va_acc))
        if ep % 5 == 0 or ep == 1:
            print(f"epoch {ep:3d}  loss {hist[-1][0]:.3f}  train {tr_acc * 100:5.1f}%  val {va_acc * 100:5.1f}%")

    os.makedirs(os.path.dirname(WEIGHTS), exist_ok=True)
    model.save()
    print(f"saved {WEIGHTS}")
    report(model, classes, Xva, y_all[va] if va.any() else None, hist)


def report(model, classes, Xva, yva, hist):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(RESULTS, exist_ok=True)
    h = np.array(hist)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(h[:, 0], color="#0f766e")
    ax[0].set_title("training loss")
    ax[1].plot(h[:, 1] * 100, label="train", color="#0f766e")
    ax[1].plot(h[:, 2] * 100, label="val (held-out takes)", color="#c2410c")
    ax[1].set_title("accuracy %")
    ax[1].legend()
    for a in ax:
        a.set_xlabel("epoch")
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "training_curve.png"), dpi=150)

    if Xva is None:
        return
    pred = model.predict_proba(Xva).argmax(1)
    C = len(classes)
    cm = np.zeros((C, C))
    for t, p in zip(yva, pred):
        cm[t, p] += 1
    cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cm, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(C), [c.upper() for c in classes])
    ax.set_yticks(range(C), [c.upper() for c in classes])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Confusion matrix (val accuracy {(pred == yva).mean() * 100:.1f}%)")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "confusion_matrix.png"), dpi=150)
    worst = sorted(((cm[i, i], classes[i]) for i in range(C)))[:5]
    print("weakest letters:", ", ".join(f"{c.upper()} {a * 100:.0f}%" for a, c in worst))


if __name__ == "__main__":
    main()
