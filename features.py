"""
features.py
------------
Turns one image into a small vector of numbers that tend to differ between
a REAL photo and a PHOTO OF A SCREEN (a "recapture").

Everything below uses only numpy + OpenCV so it stays tiny and fast enough
to eventually port to a phone (or run in a lightweight on-device model).
"""

import cv2
import numpy as np


def _load_gray_and_color(path, max_side=900):
    
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")

    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray, img


def _fft_features(gray):
    
    g = gray.astype(np.float32) / 255.0
    f = np.fft.fftshift(np.fft.fft2(g))
    mag = np.log1p(np.abs(f))

    h, w = mag.shape
    cy, cx = h // 2, w // 2

   
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    radius = min(h, w) * 0.06
    outer_mask = dist > radius

    high_freq_energy = float(mag[outer_mask].mean())

    
    band = mag[outer_mask]
    if band.size == 0:
        peak_score = 0.0
    else:
        top = np.sort(band)[-max(1, band.size // 500):]  # top 0.2%
        peak_score = float(top.mean() - band.mean())

    return high_freq_energy, peak_score


def _autocorr_periodicity(gray):
   
    hp = gray.astype(np.float32) - cv2.GaussianBlur(gray, (0, 0), sigmaX=1.5).astype(np.float32)
    hp -= hp.mean()

    f = np.fft.fft2(hp)
    power = f * np.conj(f)
    corr = np.fft.ifft2(power).real
    corr = np.fft.fftshift(corr)
    corr /= (corr.max() + 1e-6)  

    h, w = corr.shape
    cy, cx = h // 2, w // 2

   
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    ring = (dist > 2) & (dist < min(h, w) * 0.04)

    if not np.any(ring):
        return 0.0
    return float(corr[ring].max())


def _channel_decorrelation(color_img):
    
    b, g, r = cv2.split(color_img)
    b = b.astype(np.float32)
    r = r.astype(np.float32)

    def highpass(ch):
        blur = cv2.GaussianBlur(ch, (0, 0), sigmaX=1.5)
        return ch - blur

    hb, hr = highpass(b).flatten(), highpass(r).flatten()
    hb -= hb.mean()
    hr -= hr.mean()
    denom = (np.linalg.norm(hb) * np.linalg.norm(hr)) + 1e-6
    corr = float(np.dot(hb, hr) / denom)
   
    return 1.0 - corr


def _sharpness_features(gray):
    
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = float(lap.var())

    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=2)
    high_pass = gray.astype(np.float32) - blur.astype(np.float32)
    hp_energy = float(np.mean(high_pass ** 2))
    return sharpness, hp_energy


def _glare_features(color_img):
    
    hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]

    bright_desat = ((v > 245) & (s < 40)).astype(np.uint8)
    glare_ratio = float(bright_desat.mean())

    
    n_labels, _ = cv2.connectedComponents(bright_desat)
    n_blobs = max(0, n_labels - 1)
    return glare_ratio, float(n_blobs)


def _color_features(color_img):
    
    b, g, r = cv2.split(color_img.astype(np.float32))
    clip_ratio = float(np.mean((r > 250) | (g > 250) | (b > 250)))
    mean_b, mean_g, mean_r = b.mean(), g.mean(), r.mean()
    denom = mean_r + 1e-6
    blue_cast = float((mean_b - mean_r) / denom)
    return clip_ratio, blue_cast


def _rectangle_features(gray):
    
    edges = cv2.Canny(gray, 60, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                             minLineLength=min(gray.shape) * 0.4, maxLineGap=10)
    if lines is None:
        return 0.0
    return float(min(len(lines), 20))  


def _border_darkness_features(gray):
   
    h, w = gray.shape
    strip = max(2, int(min(h, w) * 0.04))

    top = gray[:strip, :]
    bottom = gray[-strip:, :]
    left = gray[:, :strip]
    right = gray[:, -strip:]

    darkness_scores = []
    for region in (top, bottom, left, right):
        mean_val = region.mean()
        std_val = region.std()
      
        is_dark_flat = (mean_val < 40) and (std_val < 15)
        darkness_scores.append(1.0 if is_dark_flat else 0.0)

    return float(sum(darkness_scores))  



def extract_features(path):
   
    gray, color = _load_gray_and_color(path)

    hf_energy, peak_score = _fft_features(gray)
    autocorr_score = _autocorr_periodicity(gray)
    channel_decorr = _channel_decorrelation(color)
    sharpness, hp_energy = _sharpness_features(gray)
    glare_ratio, n_blobs = _glare_features(color)
    clip_ratio, blue_cast = _color_features(color)
    rect_score = _rectangle_features(gray)
    border_dark = _border_darkness_features(gray)

    names = [
        "fft_high_freq_energy",
        "fft_peak_score",
        "autocorr_periodicity",
        "channel_decorrelation",
        "laplacian_sharpness",
        "high_pass_energy",
        "glare_ratio",
        "glare_blob_count",
        "highlight_clip_ratio",
        "blue_color_cast",
        "rectangle_line_score",
        "border_darkness_sides",
    ]
    vec = np.array([
        hf_energy, peak_score, autocorr_score, channel_decorr,
        sharpness, hp_energy,
        glare_ratio, n_blobs, clip_ratio, blue_cast, rect_score, border_dark,
    ], dtype=np.float32)

    return vec, names