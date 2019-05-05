import os

import cv2
import numpy as np


def color_change(modify_image, search_image, corner, rect, color):
    gray = cv2.cvtColor(search_image, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]
    x, y, width, height = rect

    for row in range(x, x + width):
        for col in range(y, y + height):
            if binary[col][row] > 100:
                modify_image[col + corner[0] - 1][row + corner[1] - 1] = color


image = cv2.imread("Part_10/p10_noisy_search1.png")
cv2.imshow("input", image)

med = cv2.medianBlur(image, 3)
gray = cv2.cvtColor(med, cv2.COLOR_BGR2GRAY)
bw = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY)[1]

cv2.imshow("med", bw)

binary = ~bw

modify_im = med

for filename in os.listdir('Letter_Cutouts'):
    current_letter = str(filename[0])
    letter_image = cv2.imread("Letter_Cutouts/" + str(filename))
    gray = cv2.cvtColor(letter_image, cv2.COLOR_BGR2GRAY)
    letter = ~cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
    letter = cv2.erode(letter, (3, 3))
    cv2.imshow("letter", letter)
    se_letter = np.array(letter, np.uint8)

    out = cv2.erode(binary, se_letter)
    cv2.imshow("erode", out)
    letter = ~letter
    se_letter = cv2.flip(letter, -1)
    out = cv2.dilate(out, se_letter)

    img = cv2.threshold(out, 200, 255, cv2.THRESH_BINARY)[1]
    _, contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for c in contours:
        x1, y1, w, h = cv2.boundingRect(c)
        color_change(modify_im, ~img, (0, -1), (x1, y1, w, h), (0, 0, 255))

cv2.imwrite("colored_p10_noisy_search1.png", modify_im)
cv2.imshow("Colored A", modify_im)

cv2.waitKey(0)
