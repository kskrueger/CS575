import os
import cv2
import numpy as np
import random


# Part 6
# Find all letters within the grid. Color each letter in a different random color. Identical letters must have
# the same color.

def color_change(modify_image, search_image, corner, rect, color):
    gray = cv2.cvtColor(search_image, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]
    x, y, width, height = rect

    for row in range(x, x + width):
        for col in range(y, y + height):
            if binary[col][row] > 100:
                modify_image[col + corner[0] - 1][row + corner[1] - 1] = color


def rand_color():
    b = random.randint(5, 245)
    g = random.randint(5, 245)
    r = random.randint(5, 245)

    return b, g, r


#image = cv2.imread('Part_6/p6_search.png')
image = cv2.imread('Part_10/p10_search1.png')
cv2.imshow("Image", image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
binary = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY_INV)[1]

se_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
e1 = cv2.erode(binary, se_v)
vert = cv2.dilate(e1, se_v)
img = cv2.threshold(e1, 200, 255, cv2.THRESH_BINARY)[1]
v_image, contours, _ = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
rows = len(contours) - 1

se_h = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
e2 = cv2.erode(binary, se_h)
horz = cv2.dilate(e2, se_h)
img = cv2.threshold(e1, 200, 255, cv2.THRESH_BINARY)[1]
h_image, contours, _ = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cols = len(contours) - 1

grid = cv2.bitwise_or(vert, horz)
im2, contours, _ = cv2.findContours(grid, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

area = 0
ct = None
for c in contours:
    if cv2.contourArea(c) > area:
        area = cv2.contourArea(c)
        ct = c

rect = cv2.boundingRect(ct)

corner = rect[1], rect[0]
all_letters = image[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
mask = ~grid[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

all_letters = cv2.dilate(all_letters, (5, 5))

height, width = mask.shape[:2]
print("Height:", height, "Width:", width)
row_size = height // rows
col_size = width // cols

word_search_array = [['-' for i in range(cols)] for j in range(rows)]

for filename in os.listdir('Letter_Cutouts'):
    current_letter = str(filename[0])
    letter_image = cv2.imread("Letter_Cutouts/" + str(filename))
    gray = cv2.cvtColor(letter_image, cv2.COLOR_BGR2GRAY)
    letter = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)[1]
    se_letter = np.array(letter, np.uint8)

    out = cv2.erode(all_letters, se_letter)
    letter = ~letter
    se_letter = cv2.flip(letter, -1)
    out = cv2.dilate(out, se_letter)

    img = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)[1]
    _, contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    color = rand_color()
    for c in contours:
        x1, y1, w, h = cv2.boundingRect(c)
        x, y = (x1 + w // 2, y1 + h // 2)
        col = x // col_size
        row = y // row_size
        if word_search_array[row][col] is '-':
            color_change(image, img, corner, (x1, y1, w, h), color)
            word_search_array[row][col] = current_letter
        # else: print("Skipped", row, col, current_letter)


# cv2.imwrite("colored.png", image)
cv2.imshow("All Colored", image)

cv2.waitKey(0)
cv2.destroyAllWindows()
