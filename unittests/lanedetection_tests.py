import os, sys, argparse

code_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
parent_dir = os.path.dirname(code_dir)
sys.path.append(parent_dir)

import cv2
from trafficbehavior.lanedetection import LaneDetector
from trafficbehavior.common import LaneMarker
from trafficbehavior.common import IPM
from trafficbehavior.common import image_util
from trafficbehavior.common import io_util
from trafficbehavior.common import display
import numpy as np

def convertIPMLaneToLane(lane, ipm):
    fitX = lane.xs
    fitY = lane.ys
    xns = []
    yns = []
    for i in [0, fitY.shape[0] - 1]:
        y = fitY[i][0]
        x = fitX[i][0]
        xn, yn = ipm.pointFromIPM((x, y))
        xns.append(xn)
        yns.append(yn)
    return xns, yns

def LineTracker_Test(pic_dir, ipm, laneMarker, laneDetector, init_line=None):
    # Read the filenames from the list.txt file
    filenames = io_util.load_fnames(pic_dir)
    ftmp1 = cv2.imread(os.path.join(pic_dir, filenames[0]), 
                                        cv2.IMREAD_GRAYSCALE).astype('uint8')
    if init_line is None:
        init_line = image_util.Line(274, 281, 125, 399)
    x1, y1 = ipm.pointToIPM((init_line.x1, init_line.y1))
    x2, y2 = ipm.pointToIPM((init_line.x2, init_line.y2))
    ipm_line = image_util.Line(x1, y1, x2, y2)

    tracker = LaneDetector.LineTracker(ipm_line, ftmp1.shape)
    display.draw_lines(ftmp1, [init_line])
    cv2.imshow('img', ftmp1)

    for i in range(len(filenames)):
        print '\r{}'.format(filenames[i]),
        ftmp = cv2.imread(os.path.join(pic_dir, filenames[i]), 
                            cv2.IMREAD_GRAYSCALE).astype('float')
        ipm_img = ipm.imageToIPM(ftmp)
        lanemarker_img = laneMarker.detect(ipm_img, 5)
        threshold_img = image_util.thresholdImage(lanemarker_img, 95)
        # cv2.imshow('ipm', ipm_img.astype('uint8'))
        # cv2.imshow('lanemarker', lanemarker_img.astype('uint8'))
        # cv2.imshow('threshold_img', threshold_img.astype('uint8'))
        lane = tracker.update(threshold_img, laneDetector)
        if lane is not None:
            xns, yns = convertIPMLaneToLane(lane, ipm)
            pt1 = (int(xns[0]), int(yns[0]))
            pt2 = (int(xns[-1]), int(yns[-1]))
            cv2.line(ftmp, pt1, pt2, (255,0,0),5)
            cv2.imshow('img', ftmp.astype('uint8'))
            cv2.waitKey(15)
        else:
            break
    print ""
    cv2.destroyAllWindows()

def LaneDetector_Test(pic_dir, ipm, LaneMarker, laneDetectorModel):
    filenames = io_util.load_fnames(pic_dir)
    ftmp1 = cv2.imread(os.path.join(pic_dir, filenames[0]), 
                                        cv2.IMREAD_GRAYSCALE).astype('uint8')

    for i in range(len(filenames)):
        display.show_progressbar(i, filenames)
        ftmp = cv2.imread(os.path.join(pic_dir, filenames[i]), 
                            cv2.IMREAD_GRAYSCALE).astype('float')
        ipm_img = ipm.imageToIPM(ftmp)
        lanemarker_img = laneMarker.detect(ipm_img, 5)
        threshold_img = image_util.thresholdImage(lanemarker_img, 95)
        # cv2.imshow('ipm', ipm_img.astype('uint8'))
        # cv2.imshow('lanemarker', lanemarker_img.astype('uint8'))
        # cv2.imshow('threshold_img', threshold_img.astype('uint8'))
        degree = 1
        lanes = laneDetectorModel.updateLaneModel(threshold_img, degree)
        # Save lanes
        for lane in lanes:
            xns, yns = convertIPMLaneToLane(lane, ipm)
            pt1 = (int(xns[0]), int(yns[0]))
            pt2 = (int(xns[-1]), int(yns[-1]))
            cv2.line(ftmp, pt1, pt2, (255,0,0),5)
        cv2.imshow('img', ftmp.astype('uint8'))
        cv2.waitKey(15)
    print ""
    cv2.destroyAllWindows()


# Sample main that uses the lane tracker
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-v', '--videopath', help='Relative Path to Video', 
                    default='../Videos/Dashcam/PCH')
    ap.add_argument('-vp', '--vanishingpoint', help="VP as X,Y", 
                    default="309.74,242.14")
    args = ap.parse_args()
    relativepath = args.videopath
    tmp = args.vanishingpoint.split(',')
    vpx = float(tmp[0])
    vpy = float(tmp[1])
    vp = [vpx, vpy]

    run_LineTracker_Test = False
    run_LaneDetector_Test = True

    filenames = io_util.load_fnames(relativepath)
    initimg = cv2.imread(os.path.join(relativepath, filenames[0]), 
                                        cv2.IMREAD_GRAYSCALE).astype('uint8')

    laneMarker = LaneMarker.NietoLaneMarker()
    laneDetector = LaneDetector.PolyRANSAC_LaneDetector(verbose=False)
    laneDetectorModel = LaneDetector.LaneDetectorModel([], initimg.shape, laneDetector)

    origPts, destPts = image_util.locateROI(initimg.shape, x_frac=1/3., 
                                      y_frac_top=1/12., y_frac_bot=1/6., vp=vp)

    show_roi = False
    if show_roi:
        for ii in range(len(origPts)):
            if ii == (len(origPts) - 1):
                cv2.line(initimg, tuple(origPts[ii]), tuple(origPts[0]), 
                            (255, 255, 255), 2)
            else:   
                cv2.line(initimg, tuple(origPts[ii]), tuple(origPts[ii+1]), 
                            (255, 255, 255), 2)
        cv2.imshow('initimg', initimg)
        cv2.waitKey(0)

    print 'Starting IPM'
    ipm = IPM.IPM(initimg.shape, initimg.shape, origPts, destPts)
    print 'Done with IPM'

    if run_LineTracker_Test:
        print 'Starting line tracker test'
        LineTracker_Test(relativepath, ipm, laneMarker, laneDetector)
        print '\tdone'
    if run_LaneDetector_Test:
        print 'Starting lane detector test'
        LaneDetector_Test(relativepath, ipm, laneMarker, laneDetectorModel)
        print '\tdone'
    print 'Done with all lanedetection unit tests'