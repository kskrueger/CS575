import cv2
import numpy as np

video_in = cv2.VideoCapture('input2b_small.mp4')
width = int(video_in.get(3))
height = int(video_in.get(4))
debug_mode = False

if not video_in.isOpened():
    print("Video not found!")

fps = video_in.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'H264')
out_writer = cv2.VideoWriter('with_binary.mp4', fourcc, fps, (width, height))
frame_number = 0

while True:
    ret, frame = video_in.read()

    if ret:
        frame_number += 1
        image = cv2.resize(frame, (1704, 482))
        size = image.shape[:2]
        if debug_mode: print("Size", size)

        spacing = 57  # spacing between LED pixels, used to determine column
        zero_x = 40

        lower_bound = (150, 150, 150)
        upper_bound = (256, 256, 256)

        # mask = cv2.inRange(image, lower_bound, upper_bound)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = cv2.threshold(image_gray, 215, 255, cv2.THRESH_BINARY)[1]
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        mask = cv2.rectangle(mask, (0, 0), (size[1], 250), (0, 0, 0), -1)
        mask = cv2.rectangle(mask, (0, 325), (size[1], size[0]), (0, 0, 0), -1)
        mask = cv2.rectangle(mask, (1620, 0), (size[1], size[0]), (0, 0, 0), -1)
        mask = cv2.rectangle(mask, (1360, 250), (1420, 350), (0, 0, 0), -1)
        mask = cv2.rectangle(mask, (1240, 250), (1263, 350), (0, 0, 0), -1)
        mask = cv2.rectangle(mask, (1240, 250), (1310, 275), (0, 0, 0), -1)
        mask = cv2.rectangle(mask, (888, 250), (915, 350), (0, 0, 0), -1)
        mask = cv2.rectangle(mask, (638, 250), (655, 350), (0, 0, 0), -1)
        im2, contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if debug_mode: cv2.imshow('image', image)
        rects = []
        area_low = 200
        area_high = 700
        max_height = 38
        ratio_thresh = 1.8
        for cont in contours:
            rect = cv2.boundingRect(cont)
            if (area_low < cv2.contourArea(cont) < area_high) and \
                    (1 / ratio_thresh < (rect[2] / rect[3]) < ratio_thresh) and rect[3] < max_height:
                if debug_mode: print(cv2.contourArea(cont))
                rects.append(rect)

        if debug_mode: print(rects)
        out = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        leds_on = []
        leds_pos = []
        for rect in rects:
            out = cv2.rectangle(out, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0))
            x_pos = rect[0] + rect[2] // 2
            pixel_num = (x_pos - zero_x) // spacing
            leds_on.append(pixel_num)
            leds_pos.append(x_pos)

        # if 26 in leds_on and 25 not in leds_on:
        for led_x in leds_pos:
            cv2.putText(image, str(1), (led_x - 20, 400), cv2.FONT_HERSHEY_PLAIN, 4, (0, 255, 0), 6)

        for num in range(1, 27):
            if num not in leds_on:
                x_position = (num - 1) * 59 + 75
                cv2.putText(image, str(0), (x_position, 400), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 0), 6)

        if debug_mode:
            print(leds_on)
            cv2.imshow("final", out)
            cv2.imshow("image out", image)
            cv2.waitKey(0)
        if not debug_mode:
            image = cv2.resize(image, (width, height))
            out_writer.write(image)
        print("Frame #:", frame_number)

    else:
        break

video_in.release()
out_writer.release()

cv2.destroyAllWindows()
