import cv2
import numpy as np

video_in = cv2.VideoCapture('input2a_small.mp4')
width = int(video_in.get(3))
height = int(video_in.get(4))
debug_mode = False

if not video_in.isOpened():
    print("Video not found!")

while True:
    ret, frame = video_in.read()

    if ret:
        cv2.imshow("Frame", frame)
        cv2.waitKey(1)
    else:
        break

video_in.release()
cv2.destroyAllWindows()
