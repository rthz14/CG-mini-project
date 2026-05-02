"""
Live Webcam Background Blur - Web Version
Deployable on Vercel
"""

import cv2
import mediapipe as mp
import numpy as np
import base64
from flask import Flask, render_template, Response, request

app = Flask(__name__)

mp_selfie = mp.solutions.selfie_segmentation
segmentor = mp_selfie.SelfieSegmentation(model_selection=1)

BLUR_MODES = {
    1: "Average",
    2: "Gaussian",
    3: "Median",
    4: "Bilateral"
}

current_mode = 2
blur_level = 10
blur_enabled = True

SEG_WIDTH = 256
SEG_HEIGHT = 144
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

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
        d = max(5, level * 3)
        sigma = level * 25
        return cv2.bilateralFilter(frame, d, sigma, sigma)
    return frame

def get_mask(rgb_frame, h, w):
    small = cv2.resize(rgb_frame, (SEG_WIDTH, SEG_HEIGHT), interpolation=cv2.INTER_LINEAR)
    result = segmentor.process(small)
    small_mask = result.segmentation_mask
    
    if small_mask is None:
        return np.ones((h, w, 3), dtype=np.float32)
    
    crop_mask = cv2.resize(small_mask, (w, h), interpolation=cv2.INTER_LINEAR)
    mask = np.where(crop_mask > 0.40, 1.0, 0.0).astype(np.float32)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL, iterations=3)
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    return np.stack([mask] * 3, axis=-1)

def generate_frames():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        
        mask_3ch = get_mask(rgb, h, w)
        
        if blur_enabled:
            blurred = apply_blur(frame, current_mode, blur_level)
            output = (frame * mask_3ch + blurred * (1.0 - mask_3ch)).astype(np.uint8)
        else:
            output = frame
        
        _, buffer = cv2.imencode('.jpg', output)
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        
        yield f"data:image/jpeg;base64,{frame_b64}\n"

@app.route('/')
def index():
    return render_template('index.html', modes=BLUR_MODES, current_mode=current_mode, blur_level=blur_level, blur_enabled=blur_enabled)

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/update', methods=['POST'])
def update():
    global current_mode, blur_level, blur_enabled
    data = request.json
    if 'mode' in data:
        current_mode = int(data['mode'])
    if 'level' in data:
        blur_level = int(data['level'])
    if 'enabled' in data:
        blur_enabled = data['enabled'] == 'true'
    return {'status': 'ok'}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)