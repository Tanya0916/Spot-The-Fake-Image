"""
check_batch.py
--------------
Checks for a specific, common data-collection trap: photos that got
resized/recompressed differently across two "batches" (e.g. some sent
through WhatsApp/an app that recompresses images, some copied straight
off the camera). Heavy recompression introduces block/grid artifacts
that look a lot like the pixel-grid signal this whole project uses to
detect screens - so if your real/ and screen/ folders were filled in
separate sessions with different compression, the model can end up
learning "which batch was this from" instead of "real vs screen", and
that shows up as a big, otherwise-inexplicable cluster of wrong
predictions concentrated in specific filename ranges.

Run:
    python check_batch.py

For every image, prints resolution, file size, and an estimated JPEG
quality level, grouped by folder. Look for a sub-group within real/ (or
screen/) whose numbers are clearly different from the rest - that's your
batch effect, and it's very likely why train.py's accuracy fell apart on
exactly those filenames.
"""

import os

import cv2
import numpy as np

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


def estimate_jpeg_quality(path):
    """Rough proxy for how hard an image has been compressed: re-encode
    at a few quality levels and see which one produces a file size
    closest to the original. Not exact, but good enough to spot 'this
    batch was compressed much harder than that batch'."""
    img = cv2.imread(path)
    if img is None:
        return None
    orig_size = os.path.getsize(path)

    best_q, best_diff = None, float("inf")
    for q in (30, 50, 70, 85, 95):
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        if not ok:
            continue
        diff = abs(len(enc) - orig_size)
        if diff < best_diff:
            best_diff, best_q = diff, q
    return best_q


def main():
    rows = []
    for folder in ("real", "screen"):
        for p in load_paths(folder):
            img = cv2.imread(p)
            if img is None:
                continue
            h, w = img.shape[:2]
            size_kb = os.path.getsize(p) / 1024
            q = estimate_jpeg_quality(p)
            rows.append((folder, p, w, h, size_kb, q))

    print(f"{'folder':7s} {'file':30s} {'resolution':12s} {'size_kb':>8s} {'~jpeg_q':>8s}")
    print("-" * 72)
    for folder, p, w, h, size_kb, q in rows:
        print(f"{folder:7s} {os.path.basename(p):30s} {f'{w}x{h}':12s} {size_kb:8.0f} {q!s:>8s}")

    # Simple summary stats per folder to make batch effects easy to spot.
    print("\nSummary (median values per folder):")
    for folder in ("real", "screen"):
        sizes = [size_kb for f, p, w, h, size_kb, q in rows if f == folder]
        quals = [q for f, p, w, h, size_kb, q in rows if f == folder and q is not None]
        if sizes:
            print(f"  {folder:7s}  median file size: {np.median(sizes):.0f} KB   "
                  f"median ~JPEG quality: {np.median(quals):.0f}")

    print(
        "\nIf you sort the table above by file size or resolution, and see a\n"
        "block of images from ONE folder with a clearly different size/quality\n"
        "than the rest of that same folder, that sub-group is a likely batch\n"
        "effect - it was probably captured/exported differently, and that\n"
        "difference (not real-vs-screen) is what the model may be learning.\n"
        "Fix: re-export/re-copy those photos the same way as the rest, or\n"
        "retake them, so real/ and screen/ are captured under one consistent\n"
        "pipeline (same phone, same app, same camera-roll export - no\n"
        "messaging-app recompression)."
    )


if __name__ == "__main__":
    main()
