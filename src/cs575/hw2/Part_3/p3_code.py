import cv2
import random
import numpy as np

image = cv2.imread('p3_search.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)[1]

se_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
e1 = cv2.erode(binary, se_v)
vert = cv2.dilate(e1, se_v)

img = cv2.threshold(e1, 200, 255, cv2.THRESH_BINARY)[1]
v_image, contours, _ = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
rows = len(contours) - 1
print("Rows:", rows)

se_h = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
e2 = cv2.erode(binary, se_h)
horz = cv2.dilate(e2, se_h)

img = cv2.threshold(e1, 200, 255, cv2.THRESH_BINARY)[1]
h_image, contours, _ = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cols = len(contours) - 1
print("Cols:", cols)

grid = ~cv2.bitwise_or(horz, vert)
grid = cv2.cvtColor(grid, cv2.COLOR_GRAY2BGR)
colors = ~cv2.bitwise_and(~image, ~grid)

height, width = image.shape[:2]
mask = np.zeros((height + 2, width + 2), np.uint8)

print("Height:", height, "Width:", width)
row_size = height // rows
col_size = width // cols

for x in range(col_size//2, width, col_size):
    for y in range(row_size//2, height, row_size):
        b = random.randint(1, 255)
        g = random.randint(1, 255)
        r = random.randint(1, 255)

        colors = cv2.floodFill(colors, mask, (x, y), (b, g, r))[1]

cv2.imwrite("p3_colors.png", colors)
cv2.imshow("Output", colors)

cv2.waitKey(0)
