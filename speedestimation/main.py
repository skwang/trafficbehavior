import numpy as np, cv2, argparse, os, sys, json
code_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
parent_dir = os.path.dirname(code_dir)
sys.path.append(parent_dir)

from trafficbehavior.common import image_util
from trafficbehavior.common import display
from trafficbehavior.common import io_util
from trafficbehavior.common import LaneMarker
from trafficbehavior.common import IPM
from trafficbehavior.speedestimation import SpeedEstimator

def computeXLineDifferences(lineL, lineR):
        slope1 = lineL.getSlopeXY()
        slope2 = lineR.getSlopeXY()
        b1 = lineL.x1 - slope1 * lineL.y1
        b2 = lineR.x1 - slope1 * lineR.y1
        if b2 > b1:
            return slope2 - slope1, b2 - b1
        else:
            return slope1 - slope2, b1 - b2

def speed_estimation_main(relativepath, uniform_conversion=False, 
                            lanes_fn='nearby_lanes.json', 
                            true_spd_fn='true_speed.json', vp_fn='vp.json', 
                            boxes_fn='boxes_output_detector_session2_raw_8.5.json'):
    print 'Starting speed estimation for {}'.format(relativepath)
    
    filenames = io_util.load_fnames(relativepath)

    fname_lanes_dict = io_util.load_json(relativepath, lanes_fn)
    vp_dict = io_util.load_json(relativepath, vp_fn)
    vp = vp_dict['vp']
    fname_boxes_dict = io_util.load_json(relativepath, boxes_fn)
        
    initimg = cv2.imread(os.path.join(relativepath, filenames[0]), cv2.IMREAD_GRAYSCALE)
    imgshape = initimg.shape
    imgheight, imgwidth = imgshape
    origPts, destPts = image_util.locateROI(imgshape, x_frac=1/3., 
                                            y_frac_top=1/25., y_frac_bot=1/6., 
                                            vp=vp)

    ipm = IPM.IPM(imgshape, imgshape, origPts, destPts)
    lanemarker = LaneMarker.NietoLaneMarker()

    if uniform_conversion:
        slopes = []
        bs = []
        for fname in filenames:
            if fname in fname_lanes_dict:
                lanes = fname_lanes_dict[fname]
                lines = []
                for lane in lanes:
                    [x1, y1] = ipm.pointToIPM(lane[0])
                    [x2, y2] = ipm.pointToIPM(lane[1])
                    lines.append(image_util.Line(x1, y1, x2, y2))
                slope, b = computeXLineDifferences(lines[0], lines[1])
                slopes.append(slope)
                bs.append(b)
        mean_slope = np.mean(np.array(slopes))
        mean_b = np.mean(np.array(bs))
        speed_detector = SpeedEstimator.Lane_SpeedEstimator(lanemarker, mean_slope, mean_b)
    else:
        speed_detector = SpeedEstimator.Lane_SpeedEstimator(lanemarker)

    output_left = {}
    output_right = {}
    for i in range(len(filenames)):
        fname = filenames[i]
        display.show_progressbar(i, filenames)
        if fname in fname_lanes_dict:
            img = cv2.imread(os.path.join(relativepath, fname))
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype('float')
            image_util.block_EBRON(gray_img)

            if fname in fname_boxes_dict:
                box_dicts = fname_boxes_dict[fname] 
                for box_dict in box_dicts:
                    x1, y1, x2, y2 = np.array(box_dict['box']).astype(np.int64)
                    gray_img[y1:y2,x1:x2] = 0

            ipm_img = ipm.imageToIPM(gray_img)
            tmp = cv2.cvtColor(ipm_img.astype('uint8').copy(), cv2.COLOR_GRAY2BGR)


            lanes = fname_lanes_dict[fname]
            lines = []
            for lane in lanes:
                [x1, y1] = ipm.pointToIPM(lane[0])
                [x2, y2] = ipm.pointToIPM(lane[1])
                lines.append(image_util.Line(x1, y1, x2, y2))
            display.draw_lines(tmp, lines, color=(0, 255, 0))
            cv2.imshow('Tmp', tmp)
            cv2.waitKey(0)


            speed_est_L, speed_est_R = speed_detector.update(ipm_img, lines[0], lines[1])
            
            output_left[fname] = speed_est_L
            output_right[fname] = speed_est_R
    print ""

    if uniform_conversion:
        io_util.save_json(relativepath, 'speed_raw_left_uni.json', output_left)
        io_util.save_json(relativepath, 'speed_raw_right_uni.json', output_right)
    else:
        io_util.save_json(relativepath, 'speed_raw_left.json', output_left)
        io_util.save_json(relativepath, 'speed_raw_right.json', output_right)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('-v', '--videopath', required=True, 
                    help="Relative Path to Video")
    args = ap.parse_args()
    relativepath = args.videopath
    speed_estimation_main(relativepath)



    