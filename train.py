

import glob
import os
import sys
import time

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

from features import extract_features

REAL_DIR = "real"
SCREEN_DIR = "screen"
MODEL_PATH = "model.pkl"
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def load_paths(folder):
    
    if not os.path.isdir(folder):
        return []
    paths = []
    for fname in sorted(os.listdir(folder)):
        ext = os.path.splitext(fname)[1].lower()
        if ext in IMG_EXTS:
            paths.append(os.path.join(folder, fname))
    return paths


def build_dataset():
    real_paths = load_paths(REAL_DIR)
    screen_paths = load_paths(SCREEN_DIR)

    if len(real_paths) < 10 or len(screen_paths) < 10:
        print(f"Found {len(real_paths)} real/ images and {len(screen_paths)} "
              f"screen/ images. Add at least ~50 of each for a meaningful "
              f"result (see the assignment's data-collection step).")
        sys.exit(1)

    X, y, used_paths, bad = [], [], [], []
    for p in real_paths:
        try:
            vec, names = extract_features(p)
            X.append(vec)
            y.append(0)  # 0 = real
            used_paths.append(p)
        except Exception as e:
            bad.append((p, str(e)))
    for p in screen_paths:
        try:
            vec, names = extract_features(p)
            X.append(vec)
            y.append(1)  # 1 = screen
            used_paths.append(p)
        except Exception as e:
            bad.append((p, str(e)))

    if bad:
        print(f"Skipped {len(bad)} unreadable file(s): {bad[:3]}{'...' if len(bad) > 3 else ''}")

    return np.array(X), np.array(y), names, used_paths


def main():
    print("Extracting features...")
    t0 = time.time()
    X, y, feature_names, used_paths = build_dataset()
    print(f"  {X.shape[0]} images, {X.shape[1]} features each "
          f"({time.time() - t0:.1f}s)")

    
    candidates = {
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced")
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=3, min_samples_leaf=4,
            class_weight="balanced", random_state=42
        ),
        "svm_rbf": make_pipeline(
            StandardScaler(), SVC(kernel="rbf", C=1.0, probability=True, class_weight="balanced")
        ),
    }

    
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
    results = {}
    print(f"\nRunning {cv.get_n_splits()} total train/test splits per model "
          f"(5-fold x 10 repeats) for a stable estimate...")
    for name, clf in candidates.items():
        scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
        results[name] = scores
        print(f"{name:22s} {scores.mean()*100:5.1f}% (+/- {scores.std()*100:4.1f}%)  "
              f"[{scores.min()*100:.0f}%-{scores.max()*100:.0f}% range across {len(scores)} splits]")

   
    ranked = sorted(results.items(), key=lambda kv: -kv[1].mean())
    best_name, best_scores = ranked[0]
    for name, scores in ranked[1:]:
        if best_scores.mean() - scores.mean() < 0.03 and scores.std() < best_scores.std():
            best_name, best_scores = name, scores
    best_clf = candidates[best_name]

    print(f"\nSelected model: {best_name} "
          f"({best_scores.mean()*100:.1f}% avg accuracy, "
          f"+/- {best_scores.std()*100:.1f}% across {len(best_scores)} splits)")
    if best_scores.mean() < 0.95:
        print(f"NOTE: this is below the 95% target. See the honesty note printed below "
              f"before doing anything to force this number up artificially.")

    
    oof_probs = cross_val_predict(
        best_clf, X, y, cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1),
        method="predict_proba"
    )[:, 1]
    np.save("oof_probs.npy", oof_probs)
    np.save("oof_paths.npy", np.array(used_paths))
    np.save("oof_labels.npy", y)
    print("Saved out-of-fold predictions to oof_probs.npy (used by diagnose.py --honest)")

   
    # in predict.py
    best_clf.fit(X, y)
    joblib.dump({"model": best_clf, "feature_names": feature_names, "model_name": best_name}, MODEL_PATH)
    print(f"Saved trained model to {MODEL_PATH}")

    
    if best_name == "logistic_regression":
        lr = best_clf.named_steps["logisticregression"]
        print("\nLearned feature weights (positive = pushes toward SCREEN):")
        for name, w in sorted(zip(feature_names, lr.coef_[0]), key=lambda t: -abs(t[1])):
            print(f"  {name:24s} {w:+.3f}")
    elif best_name == "random_forest":
        print("\nFeature importances:")
        for name, imp in sorted(zip(feature_names, best_clf.feature_importances_), key=lambda t: -t[1]):
            print(f"  {name:24s} {imp:.3f}")

    print(
        "\n--- Note ---\n"
        "This number is measured on YOUR 65 photos, not the grader's. If it's below\n"
        "95%, the fix that actually helps is more/varied real photos (different\n"
        "screens, angles, lighting) or fixing genuinely ambiguous training images -\n"
        "not cranking model complexity to force this specific number up. A model\n"

    )


if __name__ == "__main__":
    main()