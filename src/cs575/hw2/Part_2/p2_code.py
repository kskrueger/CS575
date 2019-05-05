import cv2
import numpy as np
import random

# Part 2
# Write a program that starts by cropping the grid (as was done in part1) and then separates the grid from the
# letters. Output two images: one that contains only the grid without any letters and another that contains only the
# letters without the grid (note that all letters outside the grid were already cropped and should not appear in the
# result image). Solve this and all subsequent problems using either MATLAB or OpenCV.

path = 'p2_search.png'
image = cv2.imread(path)
cv2.imshow("Image", image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

binary = ~cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)[1]

se_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
e1 = cv2.erode(binary, se_v)
vert = cv2.dilate(e1, se_v)

se_h = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
e2 = cv2.erode(binary, se_h)
horz = cv2.dilate(e2, se_h)

grid = cv2.bitwise_or(vert, horz)

im2, contours, hierarchy = cv2.findContours(grid, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

area = 0
ct = None
for c in contours:
    if cv2.contourArea(c) > area:
        area = cv2.contourArea(c)
        ct = c

rect = cv2.boundingRect(ct)

cropped = ~image[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
grid = ~grid[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
grid = cv2.cvtColor(grid, cv2.COLOR_GRAY2BGR)
letters_only = ~cv2.bitwise_and(cropped, grid)
cv2.imwrite("letters_only.png", letters_only)
cv2.imshow("Letters_only", letters_only)

grid_only = ~cv2.bitwise_and(cropped, ~grid)
cv2.imwrite("grid_only.png", grid_only)
cv2.imshow("Grid_only", grid_only)

cv2.waitKey(0)
cv2.destroyAllWindows()