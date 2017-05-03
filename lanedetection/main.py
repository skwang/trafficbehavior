import numpy as np, cv2, argparse, os, sys, json
code_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
parent_dir = os.path.dirname(code_dir)
sys.path.append(parent_dir)

from trafficbehavior.common import VPEstimator
from trafficbehavior.common import image_util
from trafficbehavior.common import display
from trafficbehavior.common import io_util
from trafficbehavior.common import LaneMarker
from trafficbehavior.common import IPM
from trafficbehavior.lanedetection import LaneDetector

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

def lane_detector_main(relativepath, boxes_fn, vp_fn, display_, verbose):
    # Read the filenames from the list.txt file
    filenames = io_util.load_fnames(relativepath)

    #filenames = filenames[700:]
    initimg = cv2.imread(os.path.join(relativepath, filenames[0]), 
                            cv2.IMREAD_GRAYSCALE)
    imgshape = initimg.shape
    imgheight, imgwidth = imgshape

    # Initialize the objects 
    lanemarker = LaneMarker.NietoLaneMarker()
    laneDetector = LaneDetector.PolyRANSAC_LaneDetector(verbose=False)
    laneDetectorModel = LaneDetector.LaneDetectorModel([], imgshape, laneDetector)

    vp_dict = io_util.load_json(relativepath, vp_fn)
    vp = vp_dict['vp']
    origPts, destPts = image_util.locateROI(imgshape, x_frac=1/3., 
                                      y_frac_top=1/25., y_frac_bot=1/6., vp=vp)


    # for ii in xrange(len(origPts)):
    #     if ii == (len(origPts) - 1):
    #         cv2.line(initimg, tuple(origPts[ii]), tuple(origPts[0]), (255, 255, 255), 2)
    #     else:   
    #         cv2.line(initimg, tuple(origPts[ii]), tuple(origPts[ii+1]), (255, 255, 255), 2)
    #cv2.imshow('initimg', initimg)
    #cv2.waitKey(0)

    if verbose:
        print 'Starting IPM'
    ipm = IPM.IPM(imgshape, imgshape, origPts, destPts)
    if verbose:
        print 'Done with IPM'

    # Dictionary where key = filename,
    # value = list of boxes of objects in that image
    fname_boxes_dict = io_util.load_json(relativepath, boxes_fn)

    fname_lanes_dict = {}
    # For each image
    for fi in range(len(filenames)):
        display.show_progressbar(fi, filenames)
        fname = filenames[fi]
        img = cv2.imread(os.path.join(relativepath,fname))
        image_util.block_EBRON(img)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype('float')

        # Check for found vehicles in the image. For each one, zero out the
        # region where the vehicle has been found
        if fname in fname_boxes_dict:
            box_dicts = fname_boxes_dict[fname] 
            for box_dict in box_dicts:
                x1, y1, x2, y2 = np.array(box_dict['box']).astype(np.int64)
                gray_img[y1:y2,x1:x2] = 0
                if display_:
                    cv2.rectangle(img, (x1, y1), (x2, y2), color=(255,0,0))

        # Perform IPM, lane marking detection, and thresholding 
        ipm_img = ipm.imageToIPM(gray_img)

        lanemarker_img = lanemarker.detect(ipm_img, 5)
        threshold_img = image_util.thresholdImage(lanemarker_img, 95)
        if display_ and verbose:
            cv2.imshow('IPMImage', ipm_img.astype('uint8'))
            cv2.imshow('LaneMarker Raw', lanemarker_img.astype('uint8'))
            cv2.imshow('laneMarker Thresholded', threshold_img.astype('uint8'))
            cv2.waitKey(1)

        # Update the lane detector model with the image to get the lanes
        degree = 1
        lanes = laneDetectorModel.updateLaneModel(threshold_img, degree)
        # Save lanes
        for lane in lanes:
            xns, yns = convertIPMLaneToLane(lane, ipm)
            pt1 = (int(xns[0]), int(yns[0]))
            pt2 = (int(xns[-1]), int(yns[-1]))
            if display_:
                cv2.line(img, pt1, pt2, (255,0,0),5)
            if fname in fname_lanes_dict:
                fname_lanes_dict[fname].append([pt1, pt2])
            else:
                fname_lanes_dict[fname] = [[pt1, pt2]]

        # For new lanes, perform backward tracking with a LineTracker object
        # (same steps as above)
        new_lanes = laneDetectorModel.get_new_lanes()
        for new_lane in new_lanes:
            new_line = new_lane.asLine()
            # if verbose:
            #     print '\t Starting backwards track'

            prev_lane_tracker = LaneDetector.LineTracker(new_line, imgshape)
            for fj in range(fi - 1, -1, -1):
                prev_fname = filenames[fj]
                prev_img = cv2.imread(os.path.join(relativepath, prev_fname))
                image_util.block_EBRON(prev_img)
                prev_gray_img = cv2.cvtColor(prev_img, 
                                            cv2.COLOR_BGR2GRAY).astype('float')
                if prev_fname in fname_boxes_dict:
                    box_dicts = fname_boxes_dict[prev_fname] 
                    for box_dict in box_dicts:
                        left, top, right, bot = box_dict['box']
                        prev_gray_img[top:bot,left:right] = 0
                prev_ipm_img = ipm.imageToIPM(prev_gray_img)
                prev_lanemarker_img = lanemarker.detect(prev_ipm_img, 5)
                prev_threshold_img = image_util.thresholdImage(
                                                        prev_lanemarker_img, 95)

                prev_lane = prev_lane_tracker.update(prev_threshold_img, 
                                                        laneDetector, degree)
                if prev_lane is None:
                    #if verbose:
                    #    print '\t Breaking, no suitable backwards lane found'
                    break
                else:
                    xns, yns = convertIPMLaneToLane(prev_lane, ipm)
                    pt1 = (int(xns[0]), int(yns[0]))
                    pt2 = (int(xns[-1]), int(yns[-1]))
                    if prev_fname not in fname_lanes_dict:
                        fname_lanes_dict[prev_fname] = [[pt1, pt2]]
                    else:
                        # Backwards lane pruning. But note this compares lines
                        # in the non-IPM space
                        line = image_util.Lane(xns, yns).asLine()
                        is_new_line = True
                        for [ppt1, ppt2] in fname_lanes_dict[prev_fname]:
                            prev_line = image_util.Line(ppt1[0], ppt1[1], 
                                                        ppt2[0], ppt2[1])
                            if line.checkLineIntersection(prev_line):
                                is_new_line = False
                                break
                        if is_new_line:
                            fname_lanes_dict[prev_fname].append([pt1, pt2])
                        else:
                            # if verbose:
                            #     print '\t Breaking, intersected with previous lane'
                            break           
            #if verbose:
            #    print '\t Ended backwards track at ' + filenames[fj]
        if display_:
            cv2.imshow('frame', img)
            cv2.waitKey(1)
    print ""

    output_fname = os.path.join(relativepath, 'lanes_output_new.json')
    with open(output_fname, 'w') as outfile:
       json.dump(fname_lanes_dict, outfile, indent=4, sort_keys=True)
       print 'Wrote json output file'
    cv2.destroyAllWindows()

def get_validation_dirs(package_dir):
    validation_dir = os.path.join(package_dir, 'validation')
    validation_dirs = []
    for item in os.listdir(validation_dir):
        if item[0] == 'v':
            validation_dirs.append(item)
    return validation_dirs

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('-v', '--videopath', required=True, 
                    help="Relative Path to Video")
    ap.add_argument('-b', '--boxes', help='Filename for boxes', 
                    default='boxes_output_detector_session2_raw_8.5.json')
    ap.add_argument('-d', '--display', default=True)
    args = ap.parse_args()

    relativepath = args.videopath
    display_ = args.display
    verbose = True
    boxes_fn = args.boxes
    vp_fn = 'vp.json'

    lane_detector_main(relativepath, boxes_fn, vp_fn, display_, verbose)
