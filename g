import cv2
from matplotlib import pyplot as plt

img = cv2.imread("garden.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
g = cv2.GaussianBlur(img, (5, 5), 0)
m = cv2.medianBlur(img, 5)

plt.subplot(1, 2, 1), plt.imshow(g), plt.title('Gaussian')
plt.subplot(1, 2, 2), plt.imshow(m), plt.title('Median')
plt.show()
