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

from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt

def computeXLineDifferences(lineL, lineR):
        slope1 = lineL.getSlopeXY()
        slope2 = lineR.getSlopeXY()
        b1 = lineL.x1 - slope1 * lineL.y1
        b2 = lineR.x1 - slope1 * lineR.y1
        if b2 > b1:
            return slope2 - slope1, b2 - b1
        else:
            return slope1 - slope2, b1 - b2

def following_analysis(relativepath, distances_fn='follow_distance_speed.json'):
    filenames = io_util.load_fnames(relativepath)
    follow_dict = io_util.load_json(relativepath, distances_fn)
    speed_dict = io_util.load_json(relativepath, 'true_speed.json')
    distances = []
    speeds = []
    true_speeds = []
    for fname in filenames:
        if fname in follow_dict:
            distances.append(follow_dict[fname]['distance'])
            speeds.append(follow_dict[fname]['speed'])
            true_speeds.append(speed_dict[fname])

    print 'Pearson coefficient and pvalue:',
    print pearsonr(distances, speeds)
    print 'Pearson coefficient and pvalue with true speed:',
    print pearsonr(distances, true_speeds)


    plt.figure(1)
    plt.scatter(distances, speeds)
    plt.xlabel('Following distance to car in front (ft)')
    plt.xlim([20, 60])
    plt.ylim([60, 80])
    plt.ylabel('Speed (MPH)')
    plt.show()

    plt.figure(2)
    plt.scatter(distances, true_speeds)
    plt.xlabel('Following distance to car in front (ft)')
    plt.xlim([20, 60])
    plt.ylim([60, 72])
    plt.ylabel('Speed (MPH)')
    plt.show()

def following_main(relativepath,
                            lanes_fn='nearby_lanes.json', 
                            spd_fn='speed_acclimit.json', vp_fn='vp.json', 
                            boxes_fn='boxes_output_detector_session2_raw_8.5.json',
                            cv_display=True):
    
    filenames = io_util.load_fnames(relativepath)

    fname_lanes_dict = io_util.load_json(relativepath, lanes_fn)
    vp_dict = io_util.load_json(relativepath, vp_fn)
    vp = vp_dict['vp']
    fname_boxes_dict = io_util.load_json(relativepath, boxes_fn)
    speed_dict = io_util.load_json(relativepath, spd_fn)
        
    initimg = cv2.imread(os.path.join(relativepath, filenames[0]), cv2.IMREAD_GRAYSCALE)
    imgshape = initimg.shape
    imgheight, imgwidth = imgshape
    origPts, destPts = image_util.locateROI(imgshape, x_frac=1/10., 
                                            y_frac_top=1/60., y_frac_bot=1/6., 
                                            vp=vp)

    for ii in xrange(len(origPts)):
        if ii == (len(origPts) - 1):
            cv2.line(initimg, tuple(origPts[ii]), tuple(origPts[0]), (255, 255, 255), 2)
        else:   
            cv2.line(initimg, tuple(origPts[ii]), tuple(origPts[ii+1]), (255, 255, 255), 2)
    cv2.imshow('initimg', initimg)
    cv2.waitKey(0)

    ipm = IPM.IPM(imgshape, imgshape, origPts, destPts)
    lanemarker = LaneMarker.NietoLaneMarker()
    follow_dict = {}
    for i in range(len(filenames)):
        fname = filenames[i]
        display.show_progressbar(i, filenames)
        if fname in fname_lanes_dict:
            img = cv2.imread(os.path.join(relativepath, fname))
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype('float')
            image_util.block_EBRON(gray_img)

            new_pts = []
            if fname in fname_boxes_dict:
                box_dicts = fname_boxes_dict[fname] 
                for box_dict in box_dicts:
                    x1, y1, x2, y2 = np.array(box_dict['box']).astype(np.int64)
                    gray_img[y1:y2,x1:x2] = 0

                    bottom_center = [(x1 + x2)/2., y2]
                    new_pt = ipm.pointToIPM(bottom_center)
                    new_pts.append(new_pt)

            ipm_img = ipm.imageToIPM(gray_img)
            tmp = cv2.cvtColor(ipm_img.astype('uint8').copy(), cv2.COLOR_GRAY2BGR)

            lanes = fname_lanes_dict[fname]
            lines = []
            for lane in lanes:
                [x1, y1] = ipm.pointToIPM(lane[0])
                [x2, y2] = ipm.pointToIPM(lane[1])
                lines.append(image_util.Line(x1, y1, x2, y2))
            slope, b = computeXLineDifferences(lines[0], lines[1])
            for new_pt in new_pts:
                [x, y] = new_pt
                if (x > lines[0].x1 and x > lines[0].x2 and 
                    x < lines[1].x1 and x < lines[1].x2 and 
                    y < imgshape[0] and y > 0):
                    cv2.circle(tmp, (int(x), int(y)), 5, (255, 0, 0), thickness=3)
                    conversion = 12./(slope * y + b)
                    distance = conversion*(imgshape[0] - y)
                    #print distance
                    follow_dict[fname] = {'distance': distance, 'speed':speed_dict[fname]}
                    display.draw_lines(tmp, lines, color=(0, 255, 0))
                    if cv_display:
                        cv2.imshow('Tmp', tmp)
                        cv2.waitKey(1)        
    print ""
    cv2.destroyAllWindows()
    io_util.save_json(relativepath, 'follow_distance_speed.json', follow_dict)
    print 'Done'


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('-v', '--videopath', required=True, 
                    help="Relative Path to Video")
    args = ap.parse_args()
    relativepath = args.videopath
    #following_main(relativepath)
    following_analysis(relativepath)



    