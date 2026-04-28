"""
Live Webcam Background Blur
CG Concepts: Average Blur, Gaussian Blur, Median Blur, Bilateral Filter
Controls:
  1 - Average (Box) Blur
  2 - Gaussian Blur
  3 - Median Blur
  4 - Bilateral Filter
  +/- - Increase/Decrease blur intensity
  Q   - Quit
"""

import cv2
import mediapipe as mp
import numpy as np

# ── MediaPipe Selfie Segmentation ──────────────────────────────────────────────
mp_selfie = mp.solutions.selfie_segmentation
segmentor = mp_selfie.SelfieSegmentation(model_selection=1)

# ── Blur modes ─────────────────────────────────────────────────────────────────
BLUR_MODES = {
    1: "Average (Box) Blur",
    2: "Gaussian Blur",
    3: "Median Blur",
    4: "Bilateral Filter",
}

current_mode = 2   # default: Gaussian
blur_level   = 3   # kernel size multiplier (odd numbers: 3,5,7,9,...)

def get_kernel(level):
    """Return an odd kernel size from blur level."""
    k = max(3, level * 2 + 1)
    return k if k % 2 == 1 else k + 1

def apply_blur(frame, mode, level):
    k = get_kernel(level)
    if mode == 1:
        # Average / Box blur — simple mean of neighbourhood pixels
        return cv2.blur(frame, (k, k))
    elif mode == 2:
        # Gaussian blur — weighted mean using Gaussian kernel
        return cv2.GaussianBlur(frame, (k, k), 0)
    elif mode == 3:
        # Median blur — replaces pixel with median of neighbourhood
        return cv2.medianBlur(frame, k)
    elif mode == 4:
        # Bilateral filter — edge-preserving blur (keeps edges sharp)
        d = max(5, level * 3)
        sigma = level * 20
        return cv2.bilateralFilter(frame, d, sigma, sigma)
    return frame

def draw_hud(frame, mode, level):
    """Overlay mode info on the frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    mode_text  = f"Mode [{mode}]: {BLUR_MODES[mode]}"
    level_text = f"Intensity: {level}  (+/- to change)"
    keys_text  = "Keys: 1-4 mode | +/- intensity | Q quit"

    cv2.putText(frame, mode_text,  (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 120), 2)
    cv2.putText(frame, level_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, keys_text,  (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
    return frame

# ── Main loop ──────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam. Check your camera connection.")

print("Webcam Background Blur started.")
print("Keys: 1=Average  2=Gaussian  3=Median  4=Bilateral  +/-=intensity  Q=quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)                        # mirror view
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ── Segmentation ───────────────────────────────────────────────────────────
    result = segmentor.process(rgb)
    mask   = result.segmentation_mask                 # float32 [0..1], 1=person

    # Smooth mask edges to avoid hard cutouts
    mask_blur = cv2.GaussianBlur(mask, (15, 15), 0)
    mask_3ch  = np.stack([mask_blur] * 3, axis=-1)   # (H,W,3)

    # ── Apply selected blur to full frame, then composite ─────────────────────
    blurred_bg = apply_blur(frame, current_mode, blur_level)

    # Composite: person pixels from original, background from blurred
    output = (frame * mask_3ch + blurred_bg * (1 - mask_3ch)).astype(np.uint8)

    # ── HUD ───────────────────────────────────────────────────────────────────
    output = draw_hud(output, current_mode, blur_level)

    cv2.imshow("Background Blur - CG Project", output)

    # ── Key handling ──────────────────────────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('Q'):
        break
    elif key in (ord('1'), ord('2'), ord('3'), ord('4')):
        current_mode = int(chr(key))
        print(f"Switched to: {BLUR_MODES[current_mode]}")
    elif key in (ord('+'), ord('=')):
        blur_level = min(blur_level + 1, 10)
        print(f"Blur intensity: {blur_level}")
    elif key in (ord('-'), ord('_')):
        blur_level = max(blur_level - 1, 1)
        print(f"Blur intensity: {blur_level}")

cap.release()
cv2.destroyAllWindows()
print("Done.")
