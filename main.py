
# !pip install pyswarms
# !pip install matplotlib
# !pip install scikit-image





import cv2
import numpy as np
import pyswarms as ps
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from skimage import img_as_float, io
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.util import random_noise
import requests
from io import BytesIO

image_url = "https://th.bing.com/th/id/OIP.yBiiLgLnSteBE0u0ELga6gHaLG?rs=1&pid=ImgDetMain"
image_path = "sample_image.png"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
}
response = requests.get(image_url, headers=headers, stream=True)
response.raise_for_status()

with open(image_path, 'wb') as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)

# Load the image here
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # Load the image in grayscale

# Optional Noise Reduction
# sigma_est = estimate_sigma(image, multichannel=False)  # Estimate noise standard deviation
# image = denoise_nl_means(image, h=1.15 * sigma_est, fast_mode=True, patch_size=5, patch_distance=3, multichannel=False)


# Enhancement Function
def enhance_image(values, image, method='clahe'):
    alpha, beta = values  # Contrast (alpha) and Brightness (beta)
    enhanced = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    if method == 'clahe':
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(enhanced)
    # Add other enhancement methods here if needed (e.g., unsharp masking)

    return enhanced

# Fitness Functions
def fitness_entropy(values):
    values = np.array(values).reshape(-1, 2)
    fitness_values = []
    for p in values:
        enhanced = enhance_image(p, image) # image is now accessible
        hist = cv2.calcHist([enhanced], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        fitness_values.append(-entropy)  # Negative entropy for maximization
    return np.array(fitness_values)

def fitness_variance(values):
    values = np.array(values).reshape(-1, 2)
    fitness_values = []
    for p in values:
        enhanced = enhance_image(p, image)
        variance = np.var(enhanced)
        fitness_values.append(-variance)  # Negative variance for maximization
    return np.array(fitness_values)

def fitness_ssim(values):
    values = np.array(values).reshape(-1, 2)
    fitness_values = []
    for p in values:
        enhanced = enhance_image(p, image)
        ssim_score = ssim(img_as_float(image), img_as_float(enhanced), data_range=255)
        fitness_values.append(-ssim_score)  # Negative SSIM for maximization
    return np.array(fitness_values)

def fitness_psnr(values):  # New fitness function for PSNR
    values = np.array(values).reshape(-1, 2)
    fitness_values = []
    for p in values:
        enhanced = enhance_image(p, image)
        psnr_score = psnr(img_as_float(image), img_as_float(enhanced), data_range=255)
        fitness_values.append(-psnr_score)  # Negative PSNR for maximization
    return np.array(fitness_values)

#PSO Optimization
fitness_function = fitness_ssim  # Select desired fitness function
options = {'c1': 2.0, 'c2': 2.0, 'w': 0.7}  # Adjust PSO parameters
bounds = ([0.5, -50], [3.0, 50])
n_particles = 50  # Increase number of particles
iters = 100  # Increase iterations

optimizer = ps.single.GlobalBestPSO(
    n_particles=n_particles, dimensions=2, options=options, bounds=bounds
)
best_cost, best_pos = optimizer.optimize(fitness_function, iters=iters)
best_values = best_pos  # Use the position as best_values

#Enhancement and Evaluation
final_image = enhance_image(best_values, image)

ssim_score = ssim(img_as_float(image), img_as_float(final_image), data_range=255)
psnr_score = psnr(img_as_float(image), img_as_float(final_image), data_range=255)
print(f"SSIM Score: {ssim_score}") # Structural Similarity Index Measure
print(f"PSNR Score: {psnr_score}") # Peak Signal-to-Noise Ratio

# Display and Save Results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.subplot(1, 2, 2)
plt.imshow(final_image, cmap='gray')
plt.title('Enhanced Image')
plt.show()

io.imsave("enhanced_image_pso.png", final_image)  # Save using scikit-image