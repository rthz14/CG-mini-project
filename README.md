# 🎥 Live Webcam Background Blur

A real-time computer graphics project that blurs webcam backgrounds using person segmentation and 4 different blur techniques.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/opencv-4.x-green.svg)
![MediaPipe](https://img.shields.io/badge/mediapipe-latest-orange.svg)

## 🎯 Features

- **Real-time person segmentation** using MediaPipe
- **4 CG blur techniques** you can switch between live:
  - Average (Box) Blur
  - Gaussian Blur
  - Median Blur
  - Bilateral Filter
- **Adjustable blur intensity** (1-10 levels)
- **Smooth edge blending** for natural results

## 🖼️ Blur Techniques Explained

| Technique | Description | Best For |
|-----------|-------------|----------|
| **Average Blur** | Simple mean of neighbourhood pixels | Fast, basic effects |
| **Gaussian Blur** | Weighted mean using Gaussian kernel | Natural-looking blur |
| **Median Blur** | Non-linear filter using median value | Noise removal, sharp edges |
| **Bilateral Filter** | Edge-preserving blur | Photo-realistic results |

## 📋 Requirements

- Python 3.8 or higher
- Webcam
- Windows/Linux/macOS

## 🚀 Installation & Setup

### Step 1: Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/webcam-background-blur.git
cd webcam-background-blur
```

### Step 2: Install dependencies

```bash
pip install opencv-python mediapipe numpy
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

### Step 3: Run the program

```bash
python bg_blur.py
```

## 🎮 Controls

| Key | Action |
|-----|--------|
| `1` | Switch to Average (Box) Blur |
| `2` | Switch to Gaussian Blur |
| `3` | Switch to Median Blur |
| `4` | Switch to Bilateral Filter |
| `+` or `=` | Increase blur intensity |
| `-` or `_` | Decrease blur intensity |
| `Q` | Quit the application |

## 🛠️ How It Works

1. **Capture** — Reads frames from your webcam
2. **Segment** — MediaPipe detects the person and creates a mask
3. **Blur** — Applies selected blur technique to the entire frame
4. **Composite** — Combines original person with blurred background
5. **Display** — Shows the result in real-time

## 📊 Technical Details

### Blur Algorithms

**Average Blur:**
```
Each pixel = mean of all neighbours
Kernel: uniform weights (1/9 for 3x3)
```

**Gaussian Blur:**
```
Weighted mean with Gaussian distribution
Center pixels weighted more than edges
```

**Median Blur:**
```
Non-linear: replaces pixel with median value
Excellent for salt-and-pepper noise
```

**Bilateral Filter:**
```
Considers spatial distance + color similarity
Preserves edges while blurring smooth areas
```

## 🐛 Troubleshooting

**Camera not opening?**
- Check if another app is using the webcam
- Try changing camera index: `cv2.VideoCapture(1)` instead of `0`

**Slow performance?**
- Use Gaussian blur (mode 2) for best speed
- Reduce blur intensity
- Lower webcam resolution

**Import errors?**
```bash
pip install --upgrade opencv-python mediapipe numpy
```

## 📝 License

MIT License - feel free to use this project for learning and experimentation.

## 🤝 Contributing

Pull requests welcome! Feel free to:
- Add new blur techniques
- Improve performance
- Add background replacement features
- Enhance the UI

## 📧 Contact

Created as a computer graphics learning project.

---

⭐ Star this repo if you found it helpful!
