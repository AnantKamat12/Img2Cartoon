import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

def Cartoon(img_path, resize=0, size=(512, 512), result_folder="result", show=0):
    # Load the image
    img = cv.imread(img_path)
    filename = os.path.splitext(os.path.basename(img_path))[0]

    # Create the result folder if it doesn't exist
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # Resize the image if resize flag is set
    if resize:
        img = cv.resize(img, size)

    # Convert the image to RGB
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Convert the image to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray, 5)  # Apply median blur

    # Apply adaptive threshold to detect edges
    edges = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 9, 9)

    # Apply bilateral filter to smooth the image while preserving edges
    color = cv.bilateralFilter(img, 9, 75, 75)  # Adjust parameters for better performance

    # Color quantization using K-means clustering
    def color_quantization(img, k):
        z = img.reshape((-1, 3))
        z = np.float32(z)
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, labels, centers = cv.kmeans(z, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        res = centers[labels.flatten()]
        return res.reshape(img.shape)

    out1 = color_quantization(img, k=4)
    out2 = color_quantization(img, k=8)

    # Combine images to create cartoon effect
    result0 = combine_images(color, color, edges)
    result0 = combine_images(result0, out1, edges)
    result1 = combine_images(out1, out2, edges)
    result2 = combine_images(out2, color, edges)
    result8 = cv.cvtColor(result0, cv.COLOR_BGR2GRAY)

    # Adjust brightness and contrast of the results
    result0 = adjust_brightness_contrast(result0)
    result1 = adjust_brightness_contrast(result1)

    # Save the results
    cv.imwrite(os.path.join(result_folder, f"{filename}_cartoon1.jpg"), result0)
    cv.imwrite(os.path.join(result_folder, f"{filename}_cartoon2.jpg"), result1)
    cv.imwrite(os.path.join(result_folder, f"{filename}_cartoon3.jpg"), result2)
    cv.imwrite(os.path.join(result_folder, f"{filename}_gray.jpg"), result8)
    cv.imwrite(os.path.join(result_folder, f"{filename}_edges.jpg"), edges)

    # Show images if show=1
    if show == 1:
        plot_images(result0, result1, result2, result8, gray)

def combine_images(image1, image2, edges):
    combined_image = cv.bitwise_and(image1, image2, mask=edges)
    imag1 = cv.bitwise_or(image1, image2, mask=edges)
    combined_image = cv.bitwise_and(combined_image, imag1, mask=edges)
    return combined_image

def adjust_brightness_contrast(image, brightness=30, contrast=30):
    new_image = np.clip((1 + contrast / 127.0) * image - contrast + brightness, 0, 255).astype(np.uint8)
    return new_image

def plot_images(result0, result1, result2, result8, gray):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 5, 1)
    plt.imshow(result0, cmap='gray') if result0.ndim == 2 else plt.imshow(cv.cvtColor(result0, cv.COLOR_BGR2RGB))
    plt.title("Cartoon 1")
    plt.axis('off')

    plt.subplot(1, 5, 2)
    plt.imshow(result1, cmap='gray') if result1.ndim == 2 else plt.imshow(cv.cvtColor(result1, cv.COLOR_BGR2RGB))
    plt.title("Cartoon 2")
    plt.axis('off')

    plt.subplot(1, 5, 3)
    plt.imshow(result2, cmap='gray') if result2.ndim == 2 else plt.imshow(cv.cvtColor(result2, cv.COLOR_BGR2RGB))
    plt.title("Cartoon 3")
    plt.axis('off')

    plt.subplot(1, 5, 4)
    plt.imshow(result8, cmap='gray')
    plt.title("Gray Img Cartoon 1")
    plt.axis('off')

    plt.subplot(1, 5, 5)
    plt.imshow(gray, cmap='gray')
    plt.title("Gray Image")
    plt.axis('off')

    plt.show()

# Example usage
if __name__ == "__main__":
    img_path = r"YOUR_IMAGE_PATH"
    Cartoon(img_path, resize=0, show=1)

