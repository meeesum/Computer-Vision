import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread("x.png")
img1 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

img2 = cv.equalizeHist(img1)
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(cv.cvtColor(img1,cv.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('OFF')
plt.subplot(1,2,2)
plt.imshow(img2, cmap='gray')
plt.title("Equalized Histogram Image")
plt.axis('OFF')

plt.show()