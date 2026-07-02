"""
diagnose.py
-----------
Run AFTER train.py. Two modes:

    python diagnose.py            (memorized view)
        Scores every image with the FINAL trained model, i.e. a model
        that has already seen every one of these images during training.
        This will look better than reality - use it to sanity-check
        obvious mislabels, not to judge real accuracy.

    python diagnose.py --honest   (genuine held-out view, recommended)
        Uses the out-of-fold predictions saved by train.py - for each
        image, the prediction comes from a model copy that did NOT see
        that image during training. This is what "real" accuracy on
        unseen photos actually looks like, and is the right one to trust.

Either way, this is the fastest way to tell whether low accuracy is a
MODEL problem (features genuinely don't separate the classes) or a DATA
problem (mislabeled file, weird outlier photo, blurry accidental shot).
"""

import os
import sys

import joblib
import numpy as np

from features import extract_features

MODEL_PATH = "model.pkl"
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def load_paths(folder):
    """List image files in a folder exactly once each (see the note in
    train.py's load_paths - avoids double-counting on case-insensitive
    filesystems like Windows/macOS)."""
    if not os.path.isdir(folder):
        return []
    paths = []
    for fname in sorted(os.listdir(folder)):
        ext = os.path.splitext(fname)[1].lower()
        if ext in IMG_EXTS:
            paths.append(os.path.join(folder, fname))
    return paths


def print_rows(rows, footer_note):
    rows.sort(reverse=True)  # worst mistakes first
    print(f"{'file':40s} {'true':6s} {'pred':6s} {'score':>7s}")
    print("-" * 65)
    n_wrong = 0
    for error, p, label, predicted, prob_screen in rows:
        true_str = "screen" if label == 1 else "real"
        pred_str = "screen" if predicted == 1 else "real"
        flag = "  <-- WRONG" if predicted != label else ""
        if flag:
            n_wrong += 1
        print(f"{p:40s} {true_str:6s} {pred_str:6s} {prob_screen:7.3f}{flag}")

    print("-" * 65)
    acc = (1 - n_wrong / len(rows)) * 100
    print(f"{n_wrong}/{len(rows)} misclassified ({acc:.1f}% accuracy) - {footer_note}")
    print("\nGo look at the WRONG files above first - check for:")
    print("  - mislabeled photos (saved in the wrong folder)")
    print("  - real photos taken of a screen-like glossy/reflective surface")
    print("  - screen photos with no glare and a very matte/anti-glare screen")
    print("  - blurry/dark outlier shots")
    print("  - genuinely hard/ambiguous cases (near-perfect recaptures, real")
    print("    objects with heavy specular glare) - these may be worth")
    print("    swapping out for cleaner examples rather than 'fixing'")


def honest_mode():
    if not (os.path.exists("oof_probs.npy") and os.path.exists("oof_paths.npy")
            and os.path.exists("oof_labels.npy")):
        print("No out-of-fold predictions found - run train.py first "
              "(it now saves oof_probs.npy / oof_paths.npy / oof_labels.npy).")
        return

    probs = np.load("oof_probs.npy")
    paths = np.load("oof_paths.npy")
    labels = np.load("oof_labels.npy")

    rows = []
    for p, label, prob_screen in zip(paths, labels, probs):
        predicted = 1 if prob_screen >= 0.5 else 0
        error = abs(prob_screen - label)
        rows.append((error, str(p), int(label), predicted, float(prob_screen)))

    print_rows(rows, "HONEST out-of-fold accuracy - trust this number over "
                      "the memorized one from `python diagnose.py`")


def memorized_mode():
    if not os.path.exists(MODEL_PATH):
        print("No model.pkl found - run train.py first.")
        return

    bundle = joblib.load(MODEL_PATH)
    clf = bundle["model"]

    rows = []
    for label, folder in [(0, "real"), (1, "screen")]:
        for p in load_paths(folder):
            vec, _ = extract_features(p)
            prob_screen = float(clf.predict_proba(vec.reshape(1, -1))[0, 1])
            predicted = 1 if prob_screen >= 0.5 else 0
            error = abs(prob_screen - label)
            rows.append((error, p, label, predicted, prob_screen))

    print_rows(rows, "MEMORIZED training accuracy - optimistic, this model has "
                      "already seen these exact images. Run with --honest instead.")


if __name__ == "__main__":
    if "--honest" in sys.argv:
        honest_mode()
    else:
        memorized_mode()
