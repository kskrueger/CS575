import cv2
import numpy as np

image = cv2.imread("Letter_Cutouts/A.png")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)[1]

im2, contours, hierarchy = cv2.findContours(binary.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

area = 0
ct = None
for c in contours:
    if cv2.contourArea(c) > area:
        area = cv2.contourArea(c)
        ct = c

rect = cv2.boundingRect(ct)

binary = binary[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
cv2.imshow("Image1",binary)

x, y = binary.shape[:2]
image2 = np.ones((x, y, 3), np.uint8)*255

for col in range(x):
    for row in range(y):
        if binary[col][row] > 100:
            image2[col][row] = (0, 0, 255)

cv2.imshow("Image", image2)
cv2.waitKey(0)
