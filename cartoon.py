    # Apply adaptive threshold to detect edges
    edges = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 9, 9)

    # Apply bilateral filter to smooth the image while preserving edges
    color = cv.bilateralFilter(img, 9, 75, 75)  # Adjust parameters for better performance

    # Color quantization using K-means clustering
    k = 8  # Number of colors
    z = img.reshape((-1, 3))  # Correct the reshape to handle color channels
    z = np.float32(z)
    criteria2 = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1)
    ret2, labels2, centers2 = cv.kmeans(z, k, None, criteria2, 10, cv.KMEANS_RANDOM_CENTERS)
    centers2 = np.uint8(centers2)
    res2 = centers2[labels2.flatten()]
    out2 = res2.reshape(img.shape)

    # Combine images to create cartoon effect
    result0 = combine_images(color, color, edges)
    result1 = combine_images(result0, out2, edges)
    result2 = combine_images(out2, color, edges)

    # Adjust brightness and contrast of the results
    result0 = adjust_brightness_contrast(result0)
    result1 = adjust_brightness_contrast(result1)

    # Save the results
    cv.imwrite(os.path.join(result_folder, f"{filename}_cartoon1.jpg"), result0)
    cv.imwrite(os.path.join(result_folder, f"{filename}_cartoon2.jpg"), result1)
    cv.imwrite(os.path.join(result_folder, f"{filename}_cartoon3.jpg"), result2)
    #cv.imwrite(os.path.join(result_folder, f"{filename}_gray.jpg"), gray)
    cv.imwrite(os.path.join(result_folder, f"{filename}_edges.jpg"), edges)

    # Show images if show=1
    if show == 1:
        plot_images(result0, result1, result2, gray)

def combine_images(image1, image2, edges):
    combined_image = cv.bitwise_and(image1, image2, mask=edges)
    return combined_image

def adjust_brightness_contrast(image, brightness=30, contrast=30):
    new_image = np.clip((1 + contrast / 127.0) * image - contrast + brightness, 0, 255).astype(np.uint8)
    return new_image

def plot_images(result0, result1, result2, gray):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 4, 1)
    plt.imshow(result0, cmap='gray') if result0.ndim == 2 else plt.imshow(cv.cvtColor(result0, cv.COLOR_BGR2RGB))
    plt.title("Cartoon 1")
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(result1, cmap='gray') if result1.ndim == 2 else plt.imshow(cv.cvtColor(result1, cv.COLOR_BGR2RGB))
    plt.title("Cartoon 2")
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(result2, cmap='gray') if result2.ndim == 2 else plt.imshow(cv.cvtColor(result2, cv.COLOR_BGR2RGB))
    plt.title("Cartoon 3")
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(gray, cmap='gray')
    plt.title("Gray Img Cartoon 2")
    plt.axis('off')

    plt.show()

# Example usage
if __name__ == "__main__":
    img_path = r"ANY_IMG_PATH"
    Cartoon(img_path, resize=0, show=1)
