import numpy as np, cv2, argparse, os, sys, json

code_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
parent_dir = os.path.dirname(code_dir)
sys.path.append(parent_dir)

import cv2
from trafficbehavior.common import io_util
from trafficbehavior.common import display

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('-v', '--videopath', required=True, 
                    help="Relative Path to Video")
    ap.add_argument('-b', '--boxes', default=True)
    ap.add_argument('-l', '--lanes', default=True)
    args = ap.parse_args()
    relativepath = args.videopath
    view_boxes = args.boxes
    view_lanes = args.lanes

    # Read the filenames from the list.txt file
    filenames = io_util.load_fnames(relativepath)

    # Dictionary where key = filename,
    # value = list of boxes of objects in that image
    fname_boxes_dict = {}
    if view_boxes:
        fname_boxes_dict = io_util.load_json(relativepath, 
                                'boxes_output_detector_session2_raw_8.5.json')

    fname_lanes_dict = {}
    if view_lanes:
        fname_lanes_dict = io_util.load_json(relativepath, 'nearby_lanes.json')

    for fname in filenames:
        img1 = cv2.imread(os.path.join(relativepath, fname))
        if fname in fname_boxes_dict:
            display.draw_boxdicts(img1, fname_boxes_dict[fname])
        if fname in fname_lanes_dict:
            display.draw_lanes(img1, fname_lanes_dict[fname])
        cv2.imshow('Green Lanes and blue boxes', img1)
        cv2.waitKey(30)
    cv2.destroyAllWindows()