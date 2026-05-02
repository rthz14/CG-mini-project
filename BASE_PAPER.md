# BASE PAPER

---

## Real-Time Background Blur in Live Video Streams Using Spatial Image Filtering and Deep Learning Segmentation

---

**Abstract** — This paper presents a real-time background blur system for live webcam video using a combination of deep learning-based human segmentation and classical spatial image filtering techniques from Computer Graphics. The system employs Google MediaPipe's Selfie Segmentation model to generate per-pixel foreground probability masks, which are then used to selectively apply one of four blur filters — Average (Box) Blur, Gaussian Blur, Median Blur, and Bilateral Filter — to the background region of each video frame. Alpha compositing is used to blend the original foreground with the blurred background. The paper discusses the mathematical foundations of each filter, their computational complexity, and their visual characteristics in the context of real-time video processing. Experimental results demonstrate that the Bilateral Filter produces the most visually realistic background blur while Gaussian Blur offers the best trade-off between quality and performance for real-time applications.

**Keywords** — Background Blur, Image Filtering, Gaussian Blur, Bilateral Filter, Median Filter, Person Segmentation, MediaPipe, OpenCV, Real-Time Video Processing, Computer Graphics, Alpha Compositing

---

## I. INTRODUCTION

The proliferation of video communication platforms such as Zoom, Microsoft Teams, and Google Meet has created significant demand for real-time background manipulation in video streams. Background blur, in particular, serves dual purposes: it protects user privacy by obscuring the environment and reduces visual distractions for other participants. Traditionally, such effects required specialized hardware (green screens) or significant computational resources. Recent advances in deep learning and efficient neural network architectures have made it possible to achieve high-quality person segmentation in real-time on commodity hardware.

Image blurring is a classical operation in Computer Graphics and Image Processing, rooted in the mathematical concept of spatial filtering through convolution. Different blur algorithms exhibit distinct characteristics in terms of edge preservation, noise reduction, and computational cost. Understanding these differences is essential for selecting the appropriate technique for a given application.

This paper makes the following contributions:

1. A complete real-time pipeline combining deep learning segmentation with classical CG blur filters.
2. A systematic comparison of four spatial blur techniques applied to background blurring.
3. An analysis of the mathematical foundations and practical trade-offs of each technique.
4. An open-source implementation using Python, OpenCV, and MediaPipe.

---

## II. RELATED WORK

### A. Image Spatial Filtering

Spatial filtering is one of the oldest and most studied areas of image processing. Gonzalez and Woods [1] provide a comprehensive treatment of linear and non-linear spatial filters. The convolution theorem establishes that linear spatial filtering is equivalent to multiplication in the frequency domain, providing both theoretical insight and computational efficiency through Fast Fourier Transform (FFT)-based implementations.

### B. Bilateral Filtering

Tomasi and Manduchi [2] introduced the bilateral filter in 1998 as an edge-preserving smoothing technique. Unlike linear filters, the bilateral filter weights pixel contributions by both spatial proximity and photometric similarity, preventing blurring across edges. Paris et al. [3] later provided a comprehensive survey of bilateral filtering and its applications.

### C. Background Segmentation

Background subtraction has been studied extensively for video surveillance applications. Stauffer and Grimson [4] proposed Gaussian Mixture Models (GMM) for adaptive background modeling. However, these methods require a static background and a training period. Deep learning approaches, such as those based on Fully Convolutional Networks (FCN) [5] and encoder-decoder architectures, have largely superseded traditional methods for person segmentation.

### D. MediaPipe Selfie Segmentation

Lugaresi et al. [6] introduced MediaPipe as a framework for building cross-platform perception pipelines. The Selfie Segmentation component uses a MobileNetV3-based architecture optimized for real-time inference on mobile and desktop CPUs. It produces a continuous probability mask rather than a binary segmentation, enabling smooth alpha blending at boundaries.

---

## III. SYSTEM DESIGN

### A. Overall Architecture

The proposed system follows a sequential pipeline:

```
Input Frame → Pre-processing → Segmentation → Mask Refinement
    → Blur Application → Alpha Compositing → Output Frame
```

Each stage is designed to minimize latency while maintaining visual quality.

### B. Pre-processing

Each captured frame undergoes two pre-processing steps:

1. **Horizontal flip** — Mirrors the frame to create a natural mirror-view experience for the user.
2. **Color space conversion** — Converts from OpenCV's default BGR format to RGB, as required by MediaPipe.

### C. Segmentation

The pre-processed RGB frame is passed to MediaPipe's SelfieSegmentation model (model_selection=1, landscape model). The model returns a single-channel float32 mask M of the same spatial dimensions as the input:

```
M : H × W → [0.0, 1.0]
```

where M(x,y) ≈ 1.0 indicates a foreground (person) pixel and M(x,y) ≈ 0.0 indicates a background pixel.

### D. Mask Refinement

The raw segmentation mask contains high-frequency transitions at object boundaries. These produce visually unpleasant hard edges in the composited output. A Gaussian blur with a 15×15 kernel is applied to the mask to create smooth spatial transitions:

```
M_smooth = GaussianBlur(M, kernel=(15,15), σ=0)
```

This step is critical for achieving natural-looking blending between the sharp foreground and the blurred background.

### E. Blur Application

The selected spatial filter is applied to the entire input frame to produce a fully blurred version. The four supported filters are described in detail in Section IV.

### F. Alpha Compositing

The final output frame is computed using standard alpha compositing [7]:

```
O(x,y) = I(x,y) · M_smooth(x,y) + B(x,y) · (1 - M_smooth(x,y))
```

Where:
- O = output frame
- I = original input frame (sharp foreground)
- B = blurred frame (blurred background)
- M_smooth = refined segmentation mask

This formula produces a weighted blend at each pixel, with the mask value determining the contribution of the original versus blurred frame.

---

## IV. SPATIAL FILTERING TECHNIQUES

### A. Average (Box) Blur

The average blur is the simplest linear spatial filter. Each output pixel is the arithmetic mean of all pixels within a rectangular neighbourhood of size k×k.

**Convolution kernel:**
```
         1    | 1  1  ... 1 |
H_avg = ——— × | 1  1  ... 1 |   (k×k matrix)
        k²    | 1  1  ... 1 |
```

**Output computation:**
```
         1    k-1 k-1
O(x,y) = —— × Σ   Σ   I(x+i, y+j)
         k²  i=0 j=0
```

The average blur is a low-pass filter that attenuates high-frequency components (edges, fine details) uniformly. Its frequency response is a sinc function, which introduces ringing artifacts at sharp edges. Despite this limitation, it is the fastest blur operation and is suitable for applications where speed is prioritized over quality.

**Computational complexity:** O(k²) per pixel, or O(1) per pixel using integral images (summed area tables).

### B. Gaussian Blur

The Gaussian blur uses a kernel derived from the 2D isotropic Gaussian function:

```
              1          x² + y²
G(x,y,σ) = ——————— × e^(- ———————)
            2πσ²           2σ²
```

The parameter σ (standard deviation) controls the spread of the blur. Larger σ values produce stronger blurring.

**Key property — separability:** The 2D Gaussian kernel is separable into two 1D kernels:

```
G(x,y,σ) = G(x,σ) × G(y,σ)
```

This allows the 2D convolution to be computed as two sequential 1D convolutions, reducing complexity from O(k²) to O(2k) per pixel.

The Gaussian blur has a Gaussian frequency response, which means it smoothly attenuates high frequencies without the ringing artifacts of the box filter. This produces a more natural-looking blur that closely resembles optical defocus.

**Computational complexity:** O(k) per pixel (separable implementation).

### C. Median Blur

The median blur is a non-linear filter that replaces each pixel with the median value of its neighbourhood. It cannot be expressed as a convolution.

**Algorithm:**
```
For each pixel (x,y):
  1. Collect all pixel values in k×k neighbourhood: N = {I(x+i, y+j)}
  2. Sort N
  3. O(x,y) = N[⌊k²/2⌋]  (middle element)
```

The median filter is particularly effective at removing impulse noise (salt-and-pepper noise) because the extreme values (the "salt" and "pepper") are never selected as the median. It also preserves edges better than linear filters because it does not average across discontinuities.

**Computational complexity:** O(k² log k) per pixel (due to sorting). Optimized implementations using histograms achieve O(k) per pixel.

### D. Bilateral Filter

The bilateral filter [2] extends the Gaussian blur by incorporating a second weighting term based on pixel intensity similarity:

```
              Σ I(xi) · fr(||I(xi) - I(x)||) · gs(||xi - x||)
BF[I](x) = ——————————————————————————————————————————————————
                   Σ fr(||I(xi) - I(x)||) · gs(||xi - x||)
```

Where:
- `gs` is the spatial Gaussian kernel (penalizes spatial distance)
- `fr` is the range Gaussian kernel (penalizes intensity difference)

**Spatial weight:**
```
ws(i,j) = e^(-(i² + j²) / 2σs²)
```

**Range (intensity) weight:**
```
wr(i,j) = e^(-(I(i,j) - I(x,y))² / 2σr²)
```

**Combined weight:**
```
w(i,j) = ws(i,j) × wr(i,j)
```

The range kernel ensures that pixels with significantly different intensities (i.e., pixels on the other side of an edge) receive very low weights. This prevents blurring across edges while still smoothing within homogeneous regions.

The bilateral filter is not separable and cannot be implemented as a simple convolution, making it significantly more computationally expensive than the other three filters.

**Computational complexity:** O(k²) per pixel with a high constant factor due to the per-pixel weight computation.

---

## V. COMPARATIVE ANALYSIS

### A. Edge Preservation

Edge preservation is quantified by the ability of a filter to maintain sharp transitions between regions of different intensity. The bilateral filter achieves the best edge preservation due to its range kernel. The median filter provides moderate edge preservation. Linear filters (average and Gaussian) do not preserve edges.

### B. Noise Reduction

For Gaussian noise, linear filters (especially Gaussian blur) are optimal in the mean-squared-error sense. For impulse noise, the median filter is superior as it completely rejects outlier values.

### C. Computational Performance

In real-time video processing at 30 fps with 720p resolution (1280×720 pixels), the relative performance of the four filters is:

| Filter | Relative Speed | Suitable for Real-Time |
|--------|---------------|----------------------|
| Average Blur | 1.0× (baseline) | Yes |
| Gaussian Blur | 0.9× | Yes |
| Median Blur | 0.3× | Yes (moderate kernel) |
| Bilateral Filter | 0.1× | Yes (small kernel only) |

### D. Visual Quality for Background Blur

For the specific application of background blur in video conferencing:

- **Average Blur** produces a flat, artificial-looking blur. Suitable for low-end hardware.
- **Gaussian Blur** produces a natural, smooth blur that closely resembles optical defocus. Best general-purpose choice.
- **Median Blur** preserves background texture edges, which can look unnatural for background blur but is useful for noisy backgrounds.
- **Bilateral Filter** produces the most realistic result, with sharp person boundaries and smooth background blur, closely mimicking camera bokeh.

---

## VI. IMPLEMENTATION

The system is implemented in Python using the following libraries:

- **OpenCV 4.x** — Frame capture, image processing, display
- **MediaPipe 0.9+** — Real-time person segmentation
- **NumPy** — Efficient array operations for compositing

The application provides an interactive interface allowing users to switch between blur modes (keys 1–4) and adjust blur intensity (keys +/-) in real-time without interrupting the video stream.

```python
# Core compositing operation
output = (frame * mask_3ch + blurred_bg * (1 - mask_3ch)).astype(np.uint8)
```

The mask is expanded to 3 channels to enable element-wise multiplication with the BGR frame arrays.

---

## VII. RESULTS AND DISCUSSION

The system was tested on a standard laptop (Intel Core i5, 8GB RAM, 720p webcam) under various lighting conditions.

**Segmentation quality** was found to be highest under uniform, frontal lighting. Backlighting and low-light conditions reduced mask accuracy, occasionally including background regions in the foreground mask.

**Frame rate** remained above 25 fps for Average, Gaussian, and Median blur modes at intensity level 5 (kernel size 11×11). The Bilateral filter reduced frame rate to approximately 15–20 fps at the same intensity level.

**Visual quality** was subjectively evaluated. The Bilateral filter consistently produced the most professional-looking result, with the Gaussian blur being preferred for its combination of quality and performance.

---

## VIII. CONCLUSION

This paper presented a real-time background blur system combining MediaPipe deep learning segmentation with four classical Computer Graphics spatial filters. The system demonstrates that effective background blur can be achieved on commodity hardware without GPU acceleration.

The comparative analysis reveals that each filter has distinct strengths: Average blur for speed, Gaussian blur for balanced quality, Median blur for noise robustness, and Bilateral filter for edge-preserving realism. The choice of filter depends on the specific requirements of the application in terms of visual quality, computational resources, and background characteristics.

Future work will explore GPU-accelerated implementations, adaptive filter selection based on scene content, and integration with background replacement for more versatile video manipulation.

---

## REFERENCES

[1] R. C. Gonzalez and R. E. Woods, *Digital Image Processing*, 4th ed. Pearson, 2018.

[2] C. Tomasi and R. Manduchi, "Bilateral filtering for gray and color images," in *Proc. IEEE International Conference on Computer Vision (ICCV)*, 1998, pp. 839–846.

[3] S. Paris, P. Kornprobst, J. Tumblin, and F. Durand, "Bilateral filtering: Theory and applications," *Foundations and Trends in Computer Graphics and Vision*, vol. 4, no. 1, pp. 1–73, 2009.

[4] C. Stauffer and W. E. L. Grimson, "Adaptive background mixture models for real-time tracking," in *Proc. IEEE CVPR*, 1999, pp. 246–252.

[5] J. Long, E. Shelhamer, and T. Darrell, "Fully convolutional networks for semantic segmentation," in *Proc. IEEE CVPR*, 2015, pp. 3431–3440.

[6] C. Lugaresi et al., "MediaPipe: A framework for building perception pipelines," *arXiv preprint arXiv:1906.08172*, 2019.

[7] T. Porter and T. Duff, "Compositing digital images," *ACM SIGGRAPH Computer Graphics*, vol. 18, no. 3, pp. 253–259, 1984.

[8] R. Szeliski, *Computer Vision: Algorithms and Applications*, 2nd ed. Springer, 2022.

[9] G. Bradski, "The OpenCV library," *Dr. Dobb's Journal of Software Tools*, 2000.

[10] A. Howard et al., "Searching for MobileNetV3," in *Proc. IEEE ICCV*, 2019, pp. 1314–1324.

---

*This paper was prepared as part of a Computer Graphics course project.*
