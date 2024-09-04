import cv2 as cv
import numpy as np
import os
import math

# Define the kernels globally
gaussian_kernel = np.array([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4], [6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]]) / 256
sobelx_kernel = np.array([[-1, 0, 1],
                          [-2, 0, 2],
                          [-1, 0, 1]])

sobely_kernel = np.array([[-1, -2, -1], 
                          [0, 0, 0], 
                          [1, 2, 1]])

def gaussian_blur(img, kernel):
    img_blur = np.zeros_like(img, dtype=np.float64)
    for y in range(2, img.shape[0]-2):
        for x in range(2, img.shape[1]-2):
            img_blur[y, x] = np.sum(kernel * img[y-2:y+3, x-2:x+3])
    return img_blur

def sobel_operator(img_blur, sobelx_kernel, sobely_kernel):
    sobelx = np.zeros_like(img_blur)
    sobely = np.zeros_like(img_blur)
    for y in range(1, img_blur.shape[0]-1):
        for x in range(1, img_blur.shape[1]-1):
            sobelx[y, x] = np.sum(sobelx_kernel * img_blur[y-1:y+2, x-1:x+2])
            sobely[y, x] = np.sum(sobely_kernel * img_blur[y-1:y+2, x-1:x+2])
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    gradient_direction = np.arctan2(sobely, sobelx)
    return gradient_magnitude, gradient_direction

def non_max_suppression(gradient_magnitude, gradient_direction):
    output = np.zeros_like(gradient_magnitude)
    PI = 180
    for i in range(1, gradient_magnitude.shape[0]-1):
        for j in range(1, gradient_magnitude.shape[1]-1):
            direction = gradient_direction[i, j]
            if (0 <= direction < PI/8) or (15*PI/8 <= direction <= 2*PI):
                if gradient_magnitude[i, j] >= max(gradient_magnitude[i, j-1], gradient_magnitude[i, j+1]):
                    output[i, j] = gradient_magnitude[i, j]
            elif PI/8 <= direction < 3*PI/8:
                if gradient_magnitude[i, j] >= max(gradient_magnitude[i-1, j+1], gradient_magnitude[i+1, j-1]):
                    output[i, j] = gradient_magnitude[i, j]
            elif 3*PI/8 <= direction < 5*PI/8:
                if gradient_magnitude[i, j] >= max(gradient_magnitude[i-1, j], gradient_magnitude[i+1, j]):
                    output[i, j] = gradient_magnitude[i, j]
            else:
                if gradient_magnitude[i, j] >= max(gradient_magnitude[i-1, j-1], gradient_magnitude[i+1, j+1]):
                    output[i, j] = gradient_magnitude[i, j]
    return output

def threshold(img, low, high):
    strong = 255
    weak = 50
    output = np.zeros_like(img)
    strong_i, strong_j = np.where(img >= high)
    weak_i, weak_j = np.where((img <= high) & (img >= low))
    output[strong_i, strong_j] = strong
    output[weak_i, weak_j] = weak
    return output

def hysteresis(img, weak=50, strong=255):
    for i in range(1, img.shape[0]-1):
        for j in range(1, img.shape[1]-1):
            if img[i, j] == weak:
                if np.any(img[i-1:i+2, j-1:j+2] == strong):
                    img[i, j] = strong
                else:
                    img[i, j] = 0
    return img

def canny(img, low, high):
    img_blur = gaussian_blur(img, gaussian_kernel)
    gradient_magnitude, gradient_direction = sobel_operator(img_blur, sobelx_kernel, sobely_kernel)
    img_nms = non_max_suppression(gradient_magnitude, gradient_direction)
    img_threshold = threshold(img_nms, low, high)
    img_edge = hysteresis(img_threshold)
    return img_edge

def apply_canny_to_all_images(folder_path, low, high):
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img = cv.imread(os.path.join(folder_path, filename), cv.IMREAD_GRAYSCALE)
            edges = canny(img, low, high)
            cv.imshow('Edges', edges)
            cv.waitKey(0)
            cv.destroyAllWindows()

def compare_sigma_effect(img_path, sigmas, low, high):
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    for sigma in sigmas:
        # Create a Gaussian kernel with the given sigma
        size = int(6*sigma + 1)
        step = 6*sigma / (size - 1)
        x = [-3*sigma + i*step for i in range(size)]
        y = [math.exp(-xi**2 / (2*sigma**2)) / math.sqrt(2*math.pi*sigma**2) for xi in x]
        gaussian_kernel = np.outer(y, y)

        # Apply the Canny edge detection algorithm
        edges = canny(img, low, high)

        # Display the result
        cv.imshow(f'Edges (sigma={sigma})', edges)
        cv.waitKey(0)
        cv.destroyAllWindows()

def study_threshold_effect(img_path, low, high):
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    # Apply the Canny edge detection algorithm
    edges = canny(img, low, high)

    # Display the result
    cv.imshow(f'Edges (low={low}, high={high})', edges)
    cv.waitKey(0)
    cv.destroyAllWindows()


def main():
    print("Choose from following\n")
    print("1. Apply Canny Edge Detector on all the images in the folder\n")
    print("2. Compare the effect of sigma value on gausian smoothing on an image\n")
    choice = int(input(print("3. Compare the effect of variations on variations in Threshold High and Threshold low\n")))
    if choice == 1:
        folder_path = input("Enter the folder path: ")
        low = float(input("Enter the low threshold value: "))
        high = float(input("Enter the high threshold value: "))
        apply_canny_to_all_images(folder_path, low, high)

    elif choice == 2:
        img_path = input("Enter the image path: ")
        sigma_values = input("Enter the sigma values (separated by spaces): ")
        sigmas = [float(value) for value in sigma_values.split()]
        low = float(input("Enter the low threshold value: "))
        high = float(input("Enter the high threshold value: "))
        compare_sigma_effect(img_path, sigmas, low, high)
    elif choice == 3:
        img_path = input("Enter the image path: ")
        low = float(input("Enter the low threshold value: "))
        high = float(input("Enter the high threshold value: "))
        study_threshold_effect(img_path, low, high)

if __name__ == "__main__":
    main()