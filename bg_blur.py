"""
Live Webcam Background Blur
CG Concepts: Average Blur, Gaussian Blur, Median Blur, Bilateral Filter

Controls:
  Click    - Click on a person box to focus on them
  C        - Cycle focus to next detected person
  A        - Auto mode (focus on center person)
  1-4      - Switch blur mode
  B        - Toggle blur on/off
  S        - Save snapshot
  Q        - Quit
  Sliders  - Blur Intensity (1-20) | Blur Mode (1-4)
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import os

# ── MediaPipe setup ────────────────────────────────────────────────────────────
mp_selfie  = mp.solutions.selfie_segmentation
mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

segmentor = mp_selfie.SelfieSegmentation(model_selection=1)

# Pose detector — used only for bounding-box detection, not drawing skeleton
pose_detector = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,          # 0 = fastest
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ── Blur modes ─────────────────────────────────────────────────────────────────
BLUR_MODES = {
    1: "Average (Box) Blur",
    2: "Gaussian Blur",
    3: "Median Blur",
    4: "Bilateral Filter",
}

current_mode = 2
blur_level   = 10
blur_enabled = True

# ── Person selection state ─────────────────────────────────────────────────────
# Each entry: (x1, y1, x2, y2) in pixel coords
detected_boxes  = []          # list of bounding boxes found this frame
selected_idx    = -1          # -1 = auto (center person)
AUTO_MODE       = True        # True = always pick the center-most person

# ── Temporal smoothing ─────────────────────────────────────────────────────────
TEMPORAL_ALPHA = 0.6
prev_mask      = None

# ── Segmentation input resolution ─────────────────────────────────────────────
SEG_WIDTH  = 256
SEG_HEIGHT = 144

# ── Morphological kernel ───────────────────────────────────────────────────────
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

SNAPSHOT_DIR = "snapshots"

# ── Helpers ────────────────────────────────────────────────────────────────────

def get_kernel(level):
    k = max(3, level * 2 + 1)
    return k if k % 2 == 1 else k + 1


def apply_blur(frame, mode, level):
    k = get_kernel(level)
    if mode == 1:
        return cv2.blur(frame, (k, k))
    elif mode == 2:
        return cv2.GaussianBlur(frame, (k, k), 0)
    elif mode == 3:
        return cv2.medianBlur(frame, k)
    elif mode == 4:
        d     = max(5, level * 3)
        sigma = level * 25
        return cv2.bilateralFilter(frame, d, sigma, sigma)
    return frame


def get_pose_boxes(rgb_frame, h, w):
    """
    Run pose detection and return two boxes per person:
      display_box  — tight box shown as the selection outline (moderate padding)
      seg_box      — generously expanded box fed to segmentation so the full
                     silhouette (hair, arms, feet) is always inside the crop
    Returns list of (display_box, seg_box) tuples.
    """
    result = pose_detector.process(rgb_frame)
    if not result.pose_landmarks:
        return []

    lms = result.pose_landmarks.landmark
    xs  = [lm.x * w for lm in lms if lm.visibility > 0.3]
    ys  = [lm.y * h for lm in lms if lm.visibility > 0.3]
    if not xs or not ys:
        return []

    raw_x1, raw_y1 = int(min(xs)), int(min(ys))
    raw_x2, raw_y2 = int(max(xs)), int(max(ys))
    bw = raw_x2 - raw_x1
    bh = raw_y2 - raw_y1

    # Display box: fixed 40 px padding on all sides
    DISP_PAD = 40
    disp = (
        max(0,     raw_x1 - DISP_PAD),
        max(0,     raw_y1 - DISP_PAD),
        min(w - 1, raw_x2 + DISP_PAD),
        min(h - 1, raw_y2 + DISP_PAD),
    )

    # Segmentation box: 30% of bbox width/height extra on every side so
    # hair, wide arms, and feet are never clipped before segmentation runs
    SEG_PAD_X = max(60, int(bw * 0.30))
    SEG_PAD_Y = max(60, int(bh * 0.30))
    seg = (
        max(0,     raw_x1 - SEG_PAD_X),
        max(0,     raw_y1 - SEG_PAD_Y),
        min(w - 1, raw_x2 + SEG_PAD_X),
        min(h - 1, raw_y2 + SEG_PAD_Y),
    )

    return [(disp, seg)]


def center_most_box(boxes, w, h):
    """Return the index of the box whose display-box center is closest to the frame center."""
    if not boxes:
        return -1
    cx, cy = w / 2, h / 2
    dists  = [((x1 + x2) / 2 - cx) ** 2 + ((y1 + y2) / 2 - cy) ** 2
              for (x1, y1, x2, y2), _ in boxes]
    return int(np.argmin(dists))


def box_for_segmentation(boxes, idx, w, h):
    """
    Return the segmentation box for the selected person.
    Falls back to an 80×90% center crop when no detection is available.
    """
    if boxes and 0 <= idx < len(boxes):
        _, seg_box = boxes[idx]
        return seg_box
    # fallback: generous center crop
    cx, cy = w // 2, h // 2
    cw, ch = int(w * 0.85), int(h * 0.95)
    return (cx - cw // 2, cy - ch // 2, cx + cw // 2, cy + ch // 2)


def get_mask(rgb_frame, h, w, seg_box):
    """
    Segment only within seg_box, embed result in a full-frame mask,
    apply morphological cleanup and temporal smoothing.
    """
    global prev_mask

    x1, y1, x2, y2 = seg_box
    crop_w = max(x2 - x1, 1)
    crop_h = max(y2 - y1, 1)
    crop   = rgb_frame[y1:y2, x1:x2]

    # Downscale for faster inference
    small      = cv2.resize(crop, (SEG_WIDTH, SEG_HEIGHT), interpolation=cv2.INTER_LINEAR)
    result     = segmentor.process(small)
    small_mask = result.segmentation_mask

    # Scale mask back to crop size, embed in full frame
    crop_mask = cv2.resize(small_mask, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
    mask      = np.zeros((h, w), dtype=np.float32)
    mask[y1:y2, x1:x2] = crop_mask

    # Threshold — lower value keeps more of the person's soft edges
    mask = np.where(mask > 0.40, 1.0, 0.0).astype(np.uint8)

    # Morphological cleanup:
    #   CLOSE (dilate then erode) — fills holes inside the silhouette
    #   NO OPEN — open() erodes first which clips thin parts like arms/hair
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL, iterations=3)
    mask = mask.astype(np.float32)

    # Temporal smoothing
    if prev_mask is not None:
        mask = TEMPORAL_ALPHA * mask + (1.0 - TEMPORAL_ALPHA) * prev_mask
    prev_mask = mask.copy()

    # Feather edges
    mask_feathered = cv2.GaussianBlur(mask, (21, 21), 0)
    return np.stack([mask_feathered] * 3, axis=-1)


def draw_person_boxes(frame, boxes, selected, auto_mode):
    """Draw bounding boxes around detected persons with selection highlight."""
    for i, (disp_box, _) in enumerate(boxes):
        x1, y1, x2, y2 = disp_box
        is_selected = (i == selected)
        color     = (0, 255, 80)  if is_selected else (100, 100, 255)
        thickness = 3             if is_selected else 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        label = f"Person {i + 1}"
        if is_selected:
            label += " [FOCUSED]"
        cv2.putText(frame, label, (x1 + 4, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return frame


def draw_hud(frame, mode, level, fps, blur_on, auto_mode, n_persons):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 85), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    focus_str   = "AUTO (center)" if auto_mode else f"Manual (click or C to cycle)"
    blur_status = "ON" if blur_on else "OFF"

    cv2.putText(frame, f"Mode [{mode}]: {BLUR_MODES[mode]}  |  Blur: {blur_status}",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 120), 2)
    cv2.putText(frame, f"Intensity: {level}   FPS: {fps:.1f}   Persons: {n_persons}   Focus: {focus_str}",
                (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1)
    cv2.putText(frame, "Click person | C cycle | A auto | 1-4 mode | B blur | S snap | Q quit",
                (10, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1)
    return frame


def save_snapshot(frame):
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    ts       = time.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(SNAPSHOT_DIR, f"snapshot_{ts}.png")
    cv2.imwrite(filename, frame)
    print(f"Snapshot saved: {filename}")


# ── Mouse callback — click to select a person ──────────────────────────────────
def on_mouse(event, x, y, flags, param):
    global selected_idx, AUTO_MODE
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    for i, (disp_box, _) in enumerate(detected_boxes):
        x1, y1, x2, y2 = disp_box
        if x1 <= x <= x2 and y1 <= y <= y2:
            selected_idx = i
            AUTO_MODE    = False
            print(f"Selected Person {i + 1}")
            return
    # Clicked outside all boxes → revert to auto
    AUTO_MODE    = True
    selected_idx = -1
    print("Auto mode (no person clicked)")


# ── Window setup ───────────────────────────────────────────────────────────────
WINDOW_NAME = "Background Blur - CG Project"
cv2.namedWindow(WINDOW_NAME)
cv2.setMouseCallback(WINDOW_NAME, on_mouse)
cv2.createTrackbar("Blur Intensity",  WINDOW_NAME, blur_level,   20, lambda v: None)
cv2.createTrackbar("Blur Mode (1-4)", WINDOW_NAME, current_mode,  4, lambda v: None)

# ── Camera ─────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam.")
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Background Blur started.")
print("Click on a person box to focus on them, or press A for auto-center mode.")
print("Keys: 1-4 mode | C cycle person | A auto | B toggle blur | S snapshot | Q quit")

fps        = 0.0
frame_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w  = frame.shape[:2]

    # ── FPS ───────────────────────────────────────────────────────────────────
    now        = time.time()
    fps        = 0.9 * fps + 0.1 * (1.0 / max(now - frame_time, 1e-6))
    frame_time = now

    # ── Trackbars ─────────────────────────────────────────────────────────────
    blur_level   = max(1, cv2.getTrackbarPos("Blur Intensity",  WINDOW_NAME))
    current_mode = max(1, cv2.getTrackbarPos("Blur Mode (1-4)", WINDOW_NAME))

    # ── Pose detection → bounding boxes ───────────────────────────────────────
    detected_boxes = get_pose_boxes(rgb, h, w)

    # In auto mode, always track the center-most person
    if AUTO_MODE:
        selected_idx = center_most_box(detected_boxes, w, h)

    # Clamp selected_idx in case person count changed
    if detected_boxes and selected_idx >= len(detected_boxes):
        selected_idx = len(detected_boxes) - 1

    # ── Determine segmentation region ─────────────────────────────────────────
    seg_box  = box_for_segmentation(detected_boxes, selected_idx, w, h)
    mask_3ch = get_mask(rgb, h, w, seg_box)

    # ── Composite ─────────────────────────────────────────────────────────────
    if blur_enabled:
        blurred_bg = apply_blur(frame, current_mode, blur_level)
        output     = (frame * mask_3ch + blurred_bg * (1.0 - mask_3ch)).astype(np.uint8)
    else:
        output = frame.copy()

    # ── Draw person boxes on output ───────────────────────────────────────────
    output = draw_person_boxes(output, detected_boxes, selected_idx, AUTO_MODE)

    # ── HUD ───────────────────────────────────────────────────────────────────
    output = draw_hud(output, current_mode, blur_level, fps,
                      blur_enabled, AUTO_MODE, len(detected_boxes))

    cv2.imshow(WINDOW_NAME, output)

    # ── Keys ──────────────────────────────────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), ord('Q')):
        break
    elif key in (ord('1'), ord('2'), ord('3'), ord('4')):
        current_mode = int(chr(key))
        cv2.setTrackbarPos("Blur Mode (1-4)", WINDOW_NAME, current_mode)
        print(f"Blur mode: {BLUR_MODES[current_mode]}")
    elif key in (ord('b'), ord('B')):
        blur_enabled = not blur_enabled
        print(f"Blur {'enabled' if blur_enabled else 'disabled'}")
    elif key in (ord('s'), ord('S')):
        save_snapshot(output)
    elif key in (ord('a'), ord('A')):
        AUTO_MODE    = True
        selected_idx = -1
        prev_mask    = None   # reset temporal mask on mode switch
        print("Auto mode: tracking center person")
    elif key in (ord('c'), ord('C')):
        if detected_boxes:
            AUTO_MODE    = False
            selected_idx = (selected_idx + 1) % len(detected_boxes)
            prev_mask    = None
            print(f"Cycled to Person {selected_idx + 1}")

cap.release()
cv2.destroyAllWindows()
print("Done.")
