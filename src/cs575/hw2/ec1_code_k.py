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


image = cv2.imread('Part_EC_1/ec1_search.png')
cv2.imshow("Image", image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
binary = ~cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY)[1]

letter_image = cv2.imread("Part_EC_1/K0.png")
gray = cv2.cvtColor(letter_image, cv2.COLOR_BGR2GRAY)
letter = ~cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
cv2.imshow("letter", letter)
se_letter = np.array(letter, np.uint8)

out = cv2.erode(binary, se_letter)
cv2.imshow("erode", out)
letter = ~letter
se_letter = cv2.flip(letter, -1)
out = cv2.dilate(out, se_letter)

# img = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
img = out
img = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)[1]
_, contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
for c in contours:
    x1, y1, w, h = cv2.boundingRect(c)
    if x1 > 590:
        color_change(image, img, (0, 0), (x1, y1, w, h), (0, 0, 255))

cv2.imwrite("ec1_1_k.png", image)
cv2.imshow("Colored K", image)
cv2.waitKey(0)