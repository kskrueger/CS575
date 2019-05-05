import cv2
import numpy as np


"""def nothing(x):
    pass


cv2.namedWindow("Window")
cv2.createTrackbar("Size", "Window", 50, 100, nothing)"""

letter_image = cv2.imread("Letter_Cutouts/A.png")
cv2.imshow("A", letter_image)

image = cv2.imread("Part_EC_1/ec1_search.png")
h, w = image.shape[:2]

size = 165
resized = cv2.resize(image, (0, 0), fx=size/100, fy=size/100)
cv2.imshow("resized", resized)

all_letters = cv2.dilate(resized, (5, 5))

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

cv2.imshow("img", img)
cv2.imshow("out", out)

cv2.waitKey(0)

"""while(1):
    size = cv2.getTrackbarPos("Size", "Window")
    resized = cv2.resize(image, (0, 0), fx=size//50, fy=size//50)
    cv2.imshow("resized", resized)
    k = cv2.waitKey(1) & 0xFF
    if k == 32:
        break

cv2.destroyAllWindows()"""
