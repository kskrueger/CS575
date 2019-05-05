import cv2
import numpy as np

# L2_clip2_16s.m4v
video_in = cv2.VideoCapture('L1_clip1_12s.m4v')
width = int(video_in.get(3))
height = int(video_in.get(4))
debug_mode = False

if not video_in.isOpened():
    print("Video not found!")

width = int(video_in.get(3))
height = int(video_in.get(4))
fps = video_in.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'H264')
out = cv2.VideoWriter('L1_clip1_12s_output.mp4', fourcc, fps, (width, height))

lower_bound = (0, 0, 140)
upper_bound = (50, 60, 255)

while True:
    ret, frame = video_in.read()

    if ret:
        # ORANGE (57, 156, 242)
        if debug_mode: cv2.imshow('frame', frame)
        red_mask = cv2.inRange(frame, lower_bound, upper_bound)
        mask = red_mask
        im2, contours, hierarchy = cv2.findContours(red_mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if debug_mode: cv2.imshow('red', red_mask)
        new_contours = []
        for cont in contours:
            if cv2.contourArea(cont) > 1200:
                new_contours.append(cont)
                rect = cv2.boundingRect(cont)

                height, width = frame.shape[:2]
                mask = np.zeros((height + 2, width + 2), np.uint8)
                cv2.floodFill(frame, mask, (rect[0]+rect[2]//2, rect[1]+rect[3]//2), (57, 156, 242), loDiff=(5, 20, 40), upDiff=(10, 10, 30), flags=0)
        # frame_out = cv2.bitwise_and(frame, frame, mask=red_mask)
        out.write(frame)
        # frame = cv2.drawContours(frame, new_contours, -1, (0, 255, 0))
        if debug_mode: cv2.imshow("draw", frame)
        if debug_mode: cv2.waitKey(1)
    else:
        break

video_in.release()
out.release()

cv2.destroyAllWindows()