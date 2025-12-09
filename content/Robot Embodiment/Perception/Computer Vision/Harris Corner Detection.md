![[Pasted image 20251117090747.png]]

A corner is detected if it witnesses a gradient in multiple directions.

The pipeline consists of:
1. Compute gradients: Ix = ∂I/∂x, Iy = ∂I/∂y  (using Sobel or similar)
2. Compute products: Ix², Iy², Ix·Iy
3. Apply Gaussian weighting: w * Ix², w * Iy², w * Ix·Iy
4. Compute R for each pixel
5. Non-maximum suppression (keep local maxima)
6. Threshold R > threshold → corners