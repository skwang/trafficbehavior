import os, sys, argparse

code_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
parent_dir = os.path.dirname(code_dir)
sys.path.append(parent_dir)

import cv2
from trafficbehavior.vehicledetection import ObjectTracker
from trafficbehavior.vehicledetection import VehicleDetector
from trafficbehavior.common import io_util
from trafficbehavior.common import display
import numpy as np

def ObjectTracker_Test(pic_dir):
    # Read the filenames from the list.txt file
    filenames = io_util.load_fnames(pic_dir)

    init_frame = 15
    confidence_thresh = 7
    filenames = io_util.load_fnames(pic_dir)[init_frame:init_frame+50]

    ftmp1 = cv2.imread(os.path.join(pic_dir, filenames[init_frame]), 
                        cv2.IMREAD_GRAYSCALE).astype('uint8')
    # Top car
    left = 295
    right = 335
    top = 225
    bot = 265

    # Alternatively, track the side car
    # left = 440
    # right = 550
    # top = 205
    # bot = 305
    tracker = ObjectTracker.dlib_ObjectTracker(ftmp1, [left, top, right, bot])
    [x1, y1, x2, y2] = tracker.get_position().astype(np.int64)
    cv2.rectangle(ftmp1, (x1, y1), (x2, y2), color=(255,0,0))
    cv2.imshow('img', ftmp1)
    cv2.waitKey(1)

    for i in xrange(init_frame, len(filenames)):
        display.show_progressbar(i, filenames)
        ftmp = cv2.imread(os.path.join(pic_dir, filenames[i]), 
                            cv2.IMREAD_GRAYSCALE).astype('uint8')
        value = tracker.update(ftmp)
        [x1, y1, x2, y2] = tracker.get_position().astype(np.int64)
        cv2.rectangle(ftmp, (x1, y1), (x2, y2), color=(255,0,0))
        cv2.imshow('img', ftmp)
        if value < confidence_thresh:
            print '\n\tlost object'
            break
        else:
            cv2.waitKey(1)
    print ""
    cv2.destroyAllWindows()

def VehicleDetector_Test(svm, pic_dir):
    vd = VehicleDetector.VehicleDetector(svm)
    font = cv2.FONT_HERSHEY_PLAIN

    # Read the filenames from the list.txt file
    init_frame = 1000
    filenames = io_util.load_fnames(pic_dir)[init_frame:init_frame+50]

    for i in range(len(filenames)):
        display.show_progressbar(i, filenames)
        ftmp = cv2.resize(cv2.imread(os.path.join(pic_dir, filenames[i]), 
                                        cv2.IMREAD_GRAYSCALE).astype('uint8'), 
                                        (320, 240))
        boxes = vd.update(ftmp)
        for vehicle_box in boxes:
            [x1, y1, x2, y2] = vehicle_box['box'].astype(np.int64)
            box_id = vehicle_box['id']
            cv2.rectangle(ftmp, (x1, y1), (x2, y2), color=(255,0,0))
            cv2.putText(ftmp, "Id:{}".format(box_id), (x1, y1 - 5), 
                        font, 1, color=(255, 0, 0))
        cv2.imshow('img', ftmp)
        cv2.waitKey(1)
    print ""
    cv2.destroyAllWindows()

# Sample main that uses the object tracker
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-v', '--videopath', help='Relative Path to Video', 
                    default='../Videos/Dashcam/PCH')
    ap.add_argument('-s', '--svm', help='Relative Path to SVM', 
                    default='./vehicledetection/svm/detector_session2_raw.svm')
    args = ap.parse_args()
    relativepath = args.videopath
    svm = args.svm

    run_ObjectTracker_Test = True
    run_VehicleDetector_Test = True

    if run_ObjectTracker_Test:
        print 'Starting object tracker test'
        ObjectTracker_Test(relativepath)
        print '\tdone'
    if run_VehicleDetector_Test:
        print 'Starting vehicle detector test'
        VehicleDetector_Test(svm, relativepath)
        print '\tdone'
    print 'Done with all vehicledetection unit tests'








    
    
    