# Spot the Fake Image

Given one image, decide REAL photo vs PHOTO OF A SCREEN (a recapture).

## Setup

```bash
pip install -r requirements.txt
```

## 1. Collect data

Take ~50 normal photos of real things and ~50 photos of a screen/printout
showing a picture. Drop them into:

```
real/*.jpg
screen/*.jpg
```

More variety (rooms, lighting, angles, different screens/printers) = a
better model.

## 2. Train

```bash
python train.py
```

Extracts 11 features per image, compares three small models (logistic
regression, random forest, RBF-SVM) via 5-fold cross-validation, keeps
whichever generalizes best, and saves it to `model.pkl`.

If accuracy comes back low, run:

```bash
python diagnose.py
```

It scores every training photo and prints the worst mistakes first, so
you can go look at the actual files - usually a mislabeled photo, an
outlier shot, or a screen with no glare/moire visible.

## 3. Predict

```bash
python predict.py path/to/some_image.jpg
# -> 0.0342          (stdout: the score, 0 = real, 1 = screen)
# -> # 27.5 ms        (stderr: how long it took)
```

If `model.pkl` doesn't exist yet, `predict.py` still works using a
hand-tuned heuristic fallback built from the same features.

## 4.  Live camera demo

```bash
pip install flask
python demo_app.py
# open http://localhost:5000
```

## Files

| File | Purpose |
|---|---|
| `features.py` | Extracts 11 physically-motivated features from one image (moire/FFT, autocorrelation periodicity, RGB channel decorrelation, sharpness, glare, color, bezel lines) |
| `train.py` | Extracts features from `real/` + `screen/`, compares 3 model types via CV, saves the best one |
| `diagnose.py` | Shows which specific training photos the model gets wrong, for debugging |
| `predict.py` | The required one-line predictor |
| `demo_app.py` / `demo.html` | Optional live webcam demo |
| `NOTE.md` | Write-up: approach, accuracy, latency, cost, next steps |

See `NOTE.md` for the full write-up.
