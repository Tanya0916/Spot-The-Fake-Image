#!/usr/bin/env python3


import argparse
import os
import sys
import time

import numpy as np

from features import extract_features

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def heuristic_score(vec, names):
    """Used only when no trained model.pkl is present.."""
    f = dict(zip(names, vec))

   
    z = 0.0
    z += 2.2 * np.clip((f["fft_peak_score"] - 2.0) / 3.0, 0, 1)
    z += 1.3 * np.clip((f["high_pass_energy"] - 8.0) / 25.0, 0, 1)
    z += 1.0 * np.clip((f["glare_blob_count"] - 1.0) / 6.0, 0, 1)
    z += 0.8 * np.clip((f["glare_ratio"] - 0.002) / 0.02, 0, 1)
    z += 0.6 * np.clip((f["blue_color_cast"] + 0.05) / 0.25, 0, 1)
    z += 0.5 * np.clip((f["rectangle_line_score"] - 3.0) / 10.0, 0, 1)
    z -= 0.8 * np.clip((f["laplacian_sharpness"] - 40.0) / 200.0, 0, 1) 
    z -= 2.5  

    return float(_sigmoid(z))


def score_image(path):
    vec, names = extract_features(path)

    if os.path.exists(MODEL_PATH):
        import joblib
        bundle = joblib.load(MODEL_PATH)
        clf = bundle["model"]
        prob_screen = float(clf.predict_proba(vec.reshape(1, -1))[0, 1])
        return prob_screen
    else:
        return heuristic_score(vec, names)


def main():
    parser = argparse.ArgumentParser(description="Score an image: 0 = real, 1 = screen recapture.")
    parser.add_argument("image", help="path to the image file")
    args = parser.parse_args()

    t0 = time.time()
    score = score_image(args.image)
    elapsed_ms = (time.time() - t0) * 1000

    print(f"{score:.4f}")
    print(f"# {elapsed_ms:.1f} ms", file=sys.stderr)


if __name__ == "__main__":
    main()
