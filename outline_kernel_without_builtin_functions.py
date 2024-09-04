import cv2 as cv
import numpy as np

def apply_outline(image):
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    outlined_image = np.zeros_like(image)
    for y in range(1, image.shape[1]-1):
        for x in range(1, image.shape[0]-1):
            for c in range(image.shape[2]):
                outlined_image[x, y, c] = np.sum(kernel * image[x-1:x+2, y-1:y+2, c])

    return outlined_image

fname = input("Enter the file name you want to read")
img = np.array(cv.imread(fname))

cv.imshow("image",img)
cv.waitKey(0)
# Add two black rows at the top and bottom, and two black columns on the left and right
img_padded = np.pad(img, ((2, 2), (2, 2), (0, 0)), mode='constant', constant_values=0)

new_image = apply_outline(img_padded)

cv.imshow("Padded Image",new_image)
cv.waitKey(0)