import imageio
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# Membaca gambar grayscale
def read_and_normalize_image(path):
    image = imageio.v2.imread(path, mode='F')  # Menggunakan mode 'F' untuk float
    return image / 255.0  # Normalisasi ke rentang 0-1

def roberts_edge_detection(image):
    """Implementasi Deteksi Tepi dengan Operator Robert."""
    kernel_x = np.array([[1, 0], [0, -1]])
    kernel_y = np.array([[0, 1], [-1, 0]])

    Gx = convolve2d(image, kernel_x, mode='same', boundary='symm')
    Gy = convolve2d(image, kernel_y, mode='same', boundary='symm')

    return np.sqrt(Gx**2 + Gy**2)

def sobel_edge_detection(image):
    """Implementasi Deteksi Tepi dengan Operator Sobel."""
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    Gx = convolve2d(image, kernel_x, mode='same', boundary='symm')
    Gy = convolve2d(image, kernel_y, mode='same', boundary='symm')

    return np.sqrt(Gx**2 + Gy**2)

# Path ke gambar input
image_path = "C:\\Dokumen\\prabowo.jpg"
image = read_and_normalize_image(image_path)

# Deteksi tepi dengan Robert dan Sobel
edges_roberts = roberts_edge_detection(image)
edges_sobel = sobel_edge_detection(image)

# Visualisasi hasil
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Roberts Edge Detection")
plt.imshow(edges_roberts, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Sobel Edge Detection")
plt.imshow(edges_sobel, cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()
