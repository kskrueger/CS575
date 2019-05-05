import cv2
import numpy as np

video_in = cv2.VideoCapture('L2_clip1_16s.m4v')
width = int(video_in.get(3))
height = int(video_in.get(4))
print("W, H",width,height)

debug_mode = False

if not video_in.isOpened():
    print("Video not found!")

width = int(video_in.get(3))
height = int(video_in.get(4))
fps = video_in.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'H264')
if not debug_mode: out = cv2.VideoWriter('L2_clip1_16s_output.mp4', fourcc, fps, (width, height))

colors = ["red", "orange", "white", "cyan", "green", "blue", "magenta", "yellow", "gray"]
column_colors_lower = [(0, 0, 255), (0, 125, 225), (225, 225, 225), (225, 225, 0), (0, 225, 0), (225, 90, 0), (225, 0, 225), (0, 205, 205), (130, 130, 130)]
column_colors_upper = [(25, 25, 255), (25, 165, 255), (255, 255, 255), (255, 255, 25), (25, 255, 25), (255, 125, 25), (255, 25, 255), (25, 255, 255), (160, 160, 160)]
column_count = 12
column_zero = 66
column_width = 55
text_height = 565

block_counts = []
column_x_pos = []
for i in range(0, column_count):
    block_counts.append(0)
    column_x_pos.append(0)

while True:
    ret, frame = video_in.read()

    if ret:
        image = cv2.rectangle(frame.copy(), (0, height), (width, height - 450), (0, 0, 0), -1)
        if debug_mode: cv2.imshow('frame', image)

        for i in range(0, column_count):
            block_counts[i] = 0
        for color_num in range(0, len(colors)):
            color = colors[color_num]
            count = 0
            lower_bound = column_colors_lower[color_num]
            upper_bound = column_colors_upper[color_num]

            mask = cv2.inRange(image, lower_bound, upper_bound)
            im2, contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            if debug_mode: cv2.imshow('Color: '+str(color), mask)
            new_contours = []
            for cont in contours:
                if cv2.contourArea(cont) > 1000:
                    new_contours.append(cont)
                    rect = cv2.boundingRect(cont)
                    x_coord = rect[0]+rect[2]//2
                    column = (x_coord-column_zero)//column_width
                    if debug_mode: print("Color, W, X, Col",color, rect[2], rect[0]+rect[2]//2, column)
                    block_counts[column] += 1
                    column_x_pos[column] = x_coord
                    count += 1

            if debug_mode: print("Count "+str(color)+" "+str(count))

        for i in range(0, column_count):
            if i is 0:
                column_x_pos[i] -= 20
            frame = cv2.putText(frame, str(block_counts[i]), (column_x_pos[i]-10, text_height), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 4)
        if debug_mode:
            cv2.imshow("draw", frame)
            cv2.waitKey(1)
        else:
            out.write(frame)
    else:
        break

video_in.release()
if not debug_mode: out.release()

cv2.destroyAllWindows()
