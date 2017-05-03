import numpy as np, cv2, argparse, json, os, sys
code_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
parent_dir = os.path.dirname(code_dir)
sys.path.append(parent_dir)

from trafficbehavior.common import IPM
from trafficbehavior.common import image_util
from trafficbehavior.common import io_util

def distToCenter(lane, ipm, width=640, height=480):
    [[x1, y1], [x2, y2]] = lane
    x1, y1 = ipm.pointToIPM((x1, y1))
    x2, y2 = ipm.pointToIPM((x2, y2))
    x = (x1 + x2)/2.
    return (x - width/2)/width

def get_nearby_lanes(relativepath, threshold=0.125, 
                        lane_fname='lanes_output_new.json', vp_fname='vp.json'):
    print 'Finding nearby lanes for {}'.format(relativepath)
    filenames = io_util.load_fnames(relativepath)
    fname_lanes_dict = io_util.load_json(relativepath, lane_fname)
    vp_dict = io_util.load_json(relativepath, vp_fname)
    vp = vp_dict['vp']
        
    initimg = cv2.imread(os.path.join(relativepath, filenames[0]), cv2.IMREAD_GRAYSCALE)
    imgshape = initimg.shape
    imgheight, imgwidth = imgshape
    origPts, destPts = image_util.locateROI(imgshape, x_frac=1/3., 
                                        y_frac_top=1/25., y_frac_bot=1/6., 
                                        vp=vp)

    ipm = IPM.IPM(imgshape, imgshape, origPts, destPts)

    new_fname_json = {}
    for fname in filenames:
        if fname in fname_lanes_dict:
            lanes = fname_lanes_dict[fname]
            distances = []
            for lane in lanes:
                dist = distToCenter(lane, ipm)
                distances.append((dist, lane))
            if len(distances) >= 2:
                distances.sort(key=lambda x:abs(x[0]))
                if abs(abs(distances[0][0]) - abs(distances[1][0])) < threshold:
                    if (distances[0][0] <= 0 and distances[1][0] >= 0):
                        new_fname_json[fname] = [distances[0][1], distances[1][1]]
                    elif (distances[0][0] >= 0 and distances[1][0] <= 0):
                        new_fname_json[fname] = [distances[1][1], distances[0][1]]
                else:
                    print abs(abs(distances[0][0]) - abs(distances[1][0]))
    return new_fname_json

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-v', '--videopath', required=True, help="Relative Path to Video")
    args = ap.parse_args()
    relativepath = args.videopath

    nearby_lanes = get_nearby_lanes(relativepath)

    output_fname = 'nearby_lanes.json'
    io_util.save_json(relativepath, output_fname, nearby_lanes)
    print 'Wrote json output file'
