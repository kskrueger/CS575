import cv2
import numpy as np

video_in = cv2.VideoCapture('L5_clip2_12s.m4v')
width = int(video_in.get(3))
height = int(video_in.get(4))
debug_mode = False

if not video_in.isOpened():
    print("Video not found!")

width = int(video_in.get(3))
height = int(video_in.get(4))
fps = int(video_in.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'H264')
out = cv2.VideoWriter('L5_clip2_12s_output.m4v', fourcc, fps, (width, height))

lower_bound = (200, 236, 0)
upper_bound = (256, 256, 230)
color = (35, 50, 235)
history_size = fps*2
points_list = []
for i in range(0, history_size):
    points_list.append((None, None))
current_frame = 0
last_point = (None, None)

while True:
    ret, frame = video_in.read()

    if ret:
        current_frame += 1
        print(current_frame)
        if debug_mode: cv2.imshow('frame', frame)
        blur = cv2.GaussianBlur(frame, (3, 3), 0)
        blue_mask = cv2.inRange(blur, lower_bound, upper_bound)
        mask = blue_mask
        im2, contours, hierarchy = cv2.findContours(blue_mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if debug_mode: cv2.imshow('Blue', blue_mask)
        new_contours = []
        for cont in contours:
            rect = cv2.boundingRect(cont)
            if 50 < cv2.contourArea(cont) < 120 and .2 < rect[2] / rect[3] < 1.8:
                new_contours.append(cont)
                if debug_mode: print(cv2.contourArea(cont))

                height, width = frame.shape[:2]
                mask = np.zeros((height + 2, width + 2), np.uint8)
                x_coord = rect[0] + rect[2] // 2
                y_coord = rect[1] + rect[3] // 2
                cv2.floodFill(frame, mask, (x_coord, y_coord), color, loDiff=(20, 20, 50), upDiff=(30, 30, 30), flags=0)
                points_list[(current_frame % history_size)] = (x_coord, y_coord)
        if len(new_contours) < 1:
            points_list[(current_frame % history_size)] = points_list[(current_frame % history_size)-1]

        for i in range(0,len(points_list)):
            point = points_list[i]
            if point[0] is not None and point[1] is not None and last_point[0] is not None and last_point[1] is not None and last_point is not points_list[(current_frame % history_size)]:
                cv2.line(frame, (point[0], point[1]), (last_point[0], last_point[1]), (0, 0, 255), 2)
            last_point = point

        out.write(frame)
        if debug_mode: cv2.imshow("draw", frame)
        if debug_mode: cv2.waitKey(1)
    else:
        break

video_in.release()
out.release()

cv2.destroyAllWindows()
