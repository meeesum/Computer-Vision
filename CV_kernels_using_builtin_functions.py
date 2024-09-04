import cv2 as cv
import numpy as np

fname = input("Enter the name of Image you want to read\n")
img = cv.imread(fname)
# Define kernels
# Define kernels
kernels = {
    'Sharpen': np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
    'Blur': np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9,
    'Edge Detection': np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
    'Box Blur': np.ones((3, 3)) / 9,
    'Sobel X': np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # Sobel kernel for edge detection in x direction
}

# Apply each kernel to the image and display the result
for kernel_name, kernel in kernels.items():
    output = cv.filter2D(img, -1, kernel)
    cv.imshow(kernel_name, output)

cv.waitKey(0)
cv.destroyAllWindows()