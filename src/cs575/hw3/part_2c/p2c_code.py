import cv2
import numpy as np

my_file = open("p2c_output.txt",'w')

video_in = cv2.VideoCapture('input2c_small.mp4')
width = int(video_in.get(3))
height = int(video_in.get(4))
debug_mode = False
last_state = 1

if not video_in.isOpened():
    print("Video not found!")

fps = video_in.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'H264')
out = cv2.VideoWriter('with_both.mp4', fourcc, fps, (width, height))


def contour_area(binary_in, in_rect):
    section = binary_in[in_rect[1]:in_rect[1] + in_rect[3], in_rect[0]:in_rect[0] + in_rect[2]]
    _, section_conts, _ = cv2.findContours(section, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_area = -1
    for sec_cont in section_conts:
        if cv2.contourArea(sec_cont) > max_area:
            max_area = cv2.contourArea(sec_cont)
    return max_area


def search_number(image_binary, filepath):
    number_image = cv2.imread(str(filepath))
    number_gray = cv2.cvtColor(number_image, cv2.COLOR_BGR2GRAY)
    number_only = cv2.threshold(number_gray, 220, 255, cv2.THRESH_BINARY)[1]
    number_only = cv2.erode(number_only, (3, 3))
    _, number_contours, _ = cv2.findContours(number_only, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    number_area = 0
    for num_cont in number_contours:
        if cv2.contourArea(num_cont) > number_area:
            number_area = cv2.contourArea(num_cont)
    if debug_mode: print("Real "+str(file_num)+" Area: " + str(number_area))
    se_num = np.array(number_only, np.uint8)
    if debug_mode: cv2.imshow("num", number_only)

    out = cv2.erode(image_binary, se_num)
    se_num = cv2.flip(number_only, -1)
    out = cv2.dilate(out, se_num)
    # cv2.imshow("out", out)

    _, contours, _ = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    # out = cv2.drawContours(out, contours, -1, (0, 255, 0))
    rects = []
    for cont in contours:
        rect = cv2.boundingRect(cont)
        rects.append(rect)

    if debug_mode: print(rects)
    for rect in rects:
        if debug_mode: print("Rect Num Area: ", contour_area(image_binary, rect))
        if not ((number_area - area_threshold_down) < contour_area(image_binary, rect) < (number_area + area_threshold_up)):
            rects.remove(rect)

    for rect in rects:
        out = cv2.rectangle(out, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0))

    if debug_mode: print(rects)
    if debug_mode:
        cv2.imshow("out", out)
        cv2.waitKey(0)
    return rects


def add_binary(input_frame, state_last):

    image = cv2.resize(input_frame, (1704, 482))
    size = image.shape[:2]
    if debug_mode: print("Size", size)

    spacing = 57  # spacing between LED pixels, used to determine column
    zero_x = 40

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

    for led_x in leds_pos:
        cv2.putText(image, str(1), (led_x - 20, 400), cv2.FONT_HERSHEY_PLAIN, 4, (0, 255, 0), 6)

    for num in range(1, 27):
        if num not in leds_on:
            x_position = (num - 1) * 59 + 75
            cv2.putText(image, str(0), (x_position, 400), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 0), 6)

    # if 26 in leds_on and 25 not in leds_on:
    test = (1 if 26 in leds_on else 0) is not state_last
    if test:
        data1 = ""
        for i in range(1, 7):
            num = 1 if i in leds_on else 0
            data1 += str(num)
        data2 = ""
        for i in range(11, 19):
            num = 1 if i in leds_on else 0
            data2 += str(num)
        data3 = ""
        for i in range(21, 23):
            num = 1 if i in leds_on else 0
            data3 += str(num)
        data4 = ""
        for i in range(25, 27):
            num = 1 if i in leds_on else 0
            data4 += str(num)
        data5 = ""
        for i in display_list[0:4]:
            data5 += str(i[1])
        data6 = ""
        for i in display_list[4:8]:
            data6 += str(i[1])

        my_file.write(data1+"\t"+data2+"\t"+data3+"\t"+data4+"\t"+data5+"\t"+data6+"\n")

    state_last = 1 if 26 in leds_on else 0

    if debug_mode:
        print(leds_on)
        cv2.imshow("final", out)
        cv2.imshow("image out", image)
        cv2.waitKey(0)

    return image, state_last


image_file = str("test_image.png")
area_threshold_up = 2000
area_threshold_down = 400
frame_number = 0

while True:
    ret, frame = video_in.read()

    if ret:
        frame_number += 1
        image = cv2.resize(frame, (1704, 482))
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_binary = cv2.threshold(image_gray, 210, 255, cv2.THRESH_BINARY)[1]
        if debug_mode: cv2.imshow("Image", image_binary)

        number_files = [8, 0, 2, 3, 4, 6, 5, 7, 1]
        display_count = 8
        display_list = []  # (x coord, number)
        coord_thresh = 50
        coord_y = 175
        coord_thresh_y = 15
        coord_max = 800
        for file_num in number_files:
            if file_num is 5:
                image_binary = cv2.threshold(image_gray, 190, 255, cv2.THRESH_BINARY)[1]
            if file_num is 0:
                image_binary = cv2.threshold(image_gray, 190, 255, cv2.THRESH_BINARY)[1]
            image_binary = cv2.GaussianBlur(image_binary, (3, 3), 0)
            image_binary = cv2.threshold(image_binary, 25, 255, cv2.THRESH_BINARY)[1]
            filename = "numbers/" + str(file_num) + ".png"
            rects = search_number(image_binary, filename)
            for rect in rects:
                x_coord = rect[0] + (rect[2] // 2)
                y_coord = rect[1] + (rect[3] // 2)
                out_range = True
                for coord in display_list:
                    if coord[0] - coord_thresh < x_coord < coord[0] + coord_thresh or x_coord > coord_max \
                            or not (coord_y-coord_thresh_y < y_coord < coord_y+coord_thresh_y):
                        out_range = False
                if out_range:
                    display_list.append((x_coord, file_num))

        display_list.sort()
        if debug_mode: print(display_list)

        for num in display_list:
            cv2.putText(image, str(num[1]), (num[0] - 20, 100), cv2.FONT_HERSHEY_PLAIN, 6, (255, 255, 0), 10)

        image, last_state = add_binary(image, last_state)

        # cv2.imshow("Final", image)
        if debug_mode: cv2.imshow('frame', image)
        if not debug_mode:
            image = cv2.resize(image, (width, height))
            out.write(image)
        print("Frame #:", frame_number)

    else:
        break

video_in.release()
out.release()
my_file.close()

cv2.destroyAllWindows()
