import os
import cv2
import numpy as np
import random

# Part 9
# Find all words and highlight them in differnt colors.

delta_x = [-1, -1, -1, 0, 0, 1, 1, 1]
delta_y = [-1, 0, 1, -1, 1, -1, 0, 1]


def color_change(modify_image, corner, rect, color):
    gray = cv2.cvtColor(modify_image, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)[1]
    x, y, width, height = rect

    for row in range(x - 3, x + width + 3):
        for col in range(y - 3, y + height + 3):
            if binary[col + corner[0] - 1][row + corner[1] - 1] > 225:
                modify_image[col + corner[0] - 1][row + corner[1] - 1] = color


def search(grid, word):
    rows = len(grid)
    cols = len(grid[0])
    length = len(word)
    for row in range(rows):
        for col in range(cols):
            if grid[row][col] != word[0]:
                continue

            for dir in range(8):
                rd = row + delta_y[dir]
                cd = col + delta_x[dir]

                for l in range(1, length):
                    if rd >= rows or rd < 0 or cd >= cols or cd < 0:
                        break

                    if grid[rd][cd] != word[l]:
                        break

                    rd += delta_y[dir]
                    cd += delta_x[dir]

                if l == length - 1:
                    return row, col, dir, l


def color_word(modify_image, corner, rect_array, position, color):
    row, col, dir, length = position
    for l in range(length + 1):
        color_change(modify_image, corner, rect_array[row][col], color)
        row += delta_y[dir]
        col += delta_x[dir]


def rand_color():
    b = random.randint(5, 245)
    g = random.randint(5, 245)
    r = random.randint(5, 245)

    return b, g, r


image = cv2.imread('Part_9/p9_search.png')
# cv2.imshow("Input Image", image)

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
letter_position_array = [[(0, 0, 0, 0) for i in range(cols)] for j in range(rows)]

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
    for c in contours:
        x1, y1, w, h = cv2.boundingRect(c)
        x, y = (x1 + w // 2, y1 + h // 2)
        col = x // col_size
        row = y // row_size
        if word_search_array[row][col] is '-':
            word_search_array[row][col] = current_letter
            letter_position_array[row][col] = x1, y1, w, h
        # else: print("Skipped", row, col, current_letter)

# color_change(image, img, corner, (x1, y1, w, h), color)

colors = [[(0, 0, 0) for i in range(14)] for j in range(14)]

words_to_find = ["CUBIC", "VARIABLE", "QUADRATIC", "ROOT", "TRINOMIAL", "BINOMIAL", "ZERO", "DEGREE", "MONOMIAL",
                 "CONSTANT", "LINEAR", "ALGEBRA", "EQUATION"]

for word in words_to_find:
    color = rand_color()
    word_pos = search(word_search_array, word)
    color_word(image, corner, letter_position_array, word_pos, color)

cv2.imwrite("colored_all_words.png", image)
cv2.imshow("Colored Words", image)

cv2.waitKey(0)
cv2.destroyAllWindows()
