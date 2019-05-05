import os
import cv2
import numpy as np
import random


# Part 5
# Find and color all vowels within the grid (A, E, I, O, U, Y). Color each vowel in a different color.

def color_change(modify_image, search_image, corner, rect, color):
    gray = cv2.cvtColor(search_image, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]
    x, y, width, height = rect

    for row in range(x, x + width):
        for col in range(y, y + height):
            if binary[col][row] > 100:
                modify_image[col + corner[0] - 1][row + corner[1] - 1] = color


image = cv2.imread('p5_search.png')
cv2.imshow("Image", image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
binary = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY_INV)[1]

se_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
e1 = cv2.erode(binary, se_v)
vert = cv2.dilate(e1, se_v)

se_h = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
e2 = cv2.erode(binary, se_h)
horz = cv2.dilate(e2, se_h)

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

colors = [(50, 0, 200), (0, 0, 255), (50, 150, 255), (0, 255, 0), (255, 0, 0), (255, 0, 150)]
color_num = 0

for filename in os.listdir('Letter_Cutouts'):
    # print(str(filename))
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
    for c in contours:
        x1, y1, w, h = cv2.boundingRect(c)
        color_change(image, img, corner, (x1, y1, w, h), colors[color_num])

    color_num += 1

cv2.imwrite("colored_vowels.png", image)
cv2.imshow("Colored Vowels", image)

cv2.waitKey(0)
cv2.destroyAllWindows()
