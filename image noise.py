import cv2 as cv
import numpy as np

img = cv.imread("a.jpg")
cv.imshow("original_image",img)
img2 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("img2",img2)
cv.waitKey(0)

noise = np.random.randint(low=0,high=25,size = (img.shape[0], img.shape[1]), dtype= img.dtype)

noisy_img = cv.add(img2, noise)
cv.imshow("noisey image",noisy_img)
cv.waitKey(0)