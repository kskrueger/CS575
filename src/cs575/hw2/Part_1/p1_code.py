import cv2
import numpy as np

# Part 1
# Write a program that uses basic morphological operations to identify and crop the grid in the image shown below.
# The result image should contain only the pixels inside the grid, including the outer border. Edit the web template
# to display your result image and the computer code that was used to generate it for this and all subsequent parts.
# You must solve part1 using Matlab and OpenCV.

path = 'Part_1/p1_search.png'
image = cv2.imread(path)
cv2.imshow("Image", image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

binary = ~cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)[1]
cv2.imshow("Binary", binary)

se_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
e1 = cv2.erode(binary, se_v)
vert = cv2.dilate(e1, se_v)
# cv2.imshow("Verticals", vert)

se_h = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
e2 = cv2.erode(binary, se_h)
horz = cv2.dilate(e2, se_h)
# cv2.imshow("Horizontal", horz)

grid = cv2.bitwise_or(vert, horz)

cv2.imshow("Grid", grid)

im2, contours, hierarchy = cv2.findContours(grid, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

area = 0
ct = None
for c in contours:
    if cv2.contourArea(c) > area:
        area = cv2.contourArea(c)
        ct = c

rect = cv2.boundingRect(ct)

output = image[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
cv2.imwrite("p1_output.png", output)
cv2.imshow("Out", output)

cv2.waitKey(0)
cv2.destroyAllWindows()
