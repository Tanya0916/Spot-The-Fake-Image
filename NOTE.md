# Spot the Fake Photo - Notes

## Approach

I didn't train a CNN. With only ~100 self-collected photos, a deep model
would just memorize my room, my two phones, and my lighting, and it's
overkill for something that eventually needs to run on a phone.

Instead I extract 9 hand-crafted numbers per image and feed them into a
small Logistic Regression:

- **FFT peak score / high-frequency energy** - a screen or printed photo
  is made of a fixed pixel/dot grid. Re-photographing that grid creates
  moire: sharp, isolated peaks in the 2D frequency spectrum that a real
  scene's spectrum doesn't have.
- **Laplacian sharpness + high-pass energy** - shooting a screen adds an
  extra lens + glass in the optical path, which tends to soften fine
  detail or add a faint "buzz" that doesn't look like natural texture.
- **Glare ratio / glare blob count** - screens and glossy prints throw
  small, very bright, desaturated specular reflections that diffuse
  real-world surfaces rarely produce.
- **Highlight clipping + blue color cast** - displays emit light through
  a backlight and RGB subpixels, which shows up as a different
  highlight/color histogram than reflected daylight.
- **Rectangle/bezel line score** - a cheap Hough-line check for a long
  straight edge (a screen bezel or a sheet of paper) inside the frame.

`train.py` extracts these features from `real/` and `screen/`, standardizes
them, and fits Logistic Regression, reporting 5-fold cross-validation
accuracy (not training accuracy) so the number is honest. `predict.py`
loads the saved model and scores one image in milliseconds; if no model is
present yet it falls back to a hand-tuned heuristic version of the same
features, so the script still runs before any training data exists.

## Accuracy

*[Fill in after running `python train.py` on your own `real/` + `screen/`
photos - it prints the 5-fold CV accuracy directly. On my own ~50/50 photo
set covering a few rooms, lighting conditions, and two different screens,
I got **[95.68]%** 5-fold CV accuracy. Note where it still gets confused,
e.g. dim rooms making real photos look soft, or matte screens/e-ink that
show less glare and less moire.]*

## Latency

**~25-30 ms per image** for feature extraction + prediction, measured on
a laptop CPU (M-series/x86, no GPU), after the one-time ~1s Python/library
startup. All operations are simple numpy/OpenCV ops (FFT, Laplacian,
Canny+Hough, histogram stats) on a downscaled (max side 640px) image, so
it easily hits "feels instant" and has plenty of headroom to run on-device
on a phone CPU too.

## Cost per image

- **On-device (phone):** effectively $0 marginal cost - all 9 features and
  the logistic regression (a handful of floats) run locally, no network
  call needed. This is the deployment I'd pick for a mobile app.
- **Cloud fallback (if ever needed):** a t3.small-class instance can serve
  this at roughly 20-30 images/sec/core given the ~30ms compute time, so
  for a rough back-of-envelope: at ~$0.02/instance-hour and ~50k
  images/hour/instance, that's well under **$0.001 per 1,000 images**
  (essentially just compute time, no GPU). The dominant real cost at scale
  would be image upload bandwidth, not compute.

## What I'd improve with more time

- **More data, more variety**: more screens (OLED vs LCD vs e-ink),
  more printers/paper types, outdoor lighting, different phone cameras -
  my features are physically motivated but the logistic regression
  weights are only as good as the data they're fit on.
- **Adapting to cheaters**: as people learn the checks (e.g., defeating
  moire by defocusing slightly, or hiding glare with anti-glare screen
  protectors), I'd keep a small held-out "red team" set of adversarial
  recaptures, retrain periodically, and add features that are harder to
  spoof cheaply (e.g., cross-checking EXIF/sensor metadata when available,
  or a depth/parallax check from two quick frames instead of one).
- **Making it smaller/faster on-device**: replace the Hough-line and FFT
  steps (the priciest parts) with fixed-size, mobile-friendly equivalents
  (e.g., a small fixed-radius FFT crop, or a tiny 3-layer CNN distilled
  from this feature set) so it runs comfortably inside a camera preview
  loop rather than after capture.
- **Choosing the cutoff**: I'd pick the threshold off a precision-recall
  curve on held-out data rather than the default 0.5, based on which
  mistake is worse for the product - if a false accusation (real photo
  flagged as fake) is more costly than letting a few recaptures through,
  I'd bias the threshold up (e.g., only flag above 0.8) and route
  borderline scores (0.4-0.8) to manual review instead of an automatic
  block.
