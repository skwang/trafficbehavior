import cv2, sys
from trafficbehavior.common import image_util

def draw_lanes(img, lanes, color=(0,255,0), thickness=3):
    for lane in lanes:
        [pt1, pt2] = lane
        pt1 = (int(pt1[0]), int(pt1[1]))
        pt2 = (int(pt2[0]), int(pt2[1]))
        cv2.line(img, pt1, pt2, color, thickness)

def draw_lines(img, lines, color=(255, 0, 0), thickness=2):
    for line in lines:
        pt1 = (int(line.x1), int(line.y1))
        pt2 = (int(line.x2), int(line.y2))
        cv2.line(img, pt1, pt2, color, thickness)

def draw_boxes(img, boxes, color=(255,0,0), fontsize=0.7, 
                font=cv2.FONT_HERSHEY_PLAIN, thickness=1):
    for box in boxes:
        [x1, y1, x2, y2] = box
        #box_id = "n/a"
        cv2.rectangle(img, (x1, y1), (x2, y2), color=color, thickness=thickness)
        # cv2.putText(img, "Id:{}".format(box_id), (x1, y1-5), font, 
        #             fontsize, color=color)

def draw_boxdicts(img, boxes, color=(255,0,0), fontsize=0.7, 
                    font=cv2.FONT_HERSHEY_PLAIN, thickness=1):
    for box_dict in boxes:
        [x1, y1, x2, y2] = box_dict['box']
        box_id = box_dict['id']
        cv2.rectangle(img, (x1, y1), (x2, y2), color=color, thickness=thickness)
        cv2.putText(img, "Id:{}".format(box_id), (x1, y1-5), font, 
                    fontsize, color=color)

def show_progressbar(curr_index, filenames):
    toolbar_width = 40
    fname = filenames[curr_index]
    progress = float(curr_index)/len(filenames)
    percent_str = "{}%".format(int(progress*100))
    # setup toolbar
    sys.stdout.write("\r{}|{}|{}".format(fname, " "*toolbar_width, percent_str))
    sys.stdout.flush()
    # goes back to right before the first |
    sys.stdout.write("\b" * (toolbar_width+1+len(percent_str)))
    num_to_draw = int(progress*toolbar_width)
    for i in range(num_to_draw):
        sys.stdout.write("#")
    sys.stdout.flush()
