import cv2
import numpy as np

video_in = cv2.VideoCapture('L1_clip1_12s.m4v')
width = int(video_in.get(3))
height = int(video_in.get(4))

if not video_in.isOpened():
    print("Video not found!")

width = int(video_in.get(3))
height = int(video_in.get(4))
fps = video_in.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'H264')
out = cv2.VideoWriter('L1_clip1_12s_output.mp4', fourcc, fps, (width, height))

while True:
    ret, frame = video_in.read()

    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        out.write(frame_out)
        # cv2.imshow('frame', frame)
        cv2.waitKey(1)
    else:
        break

video_in.release()
out.release()

cv2.destroyAllWindows()
