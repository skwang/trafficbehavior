import os, sys, argparse, cv2, numpy as np
np.set_printoptions(precision=2)
code_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
parent_dir = os.path.dirname(code_dir)
sys.path.append(parent_dir)
    
from trafficbehavior.common import image_util
from trafficbehavior.common import io_util
from trafficbehavior.common import IPM

import matplotlib.pyplot as plt
from scipy.signal import medfilt

def visualize_lane_changes(relativepath):
    lanes_fn = 'nearby_lanes.json'
    vp_fn = 'vp.json'
    filenames = io_util.load_fnames(relativepath)
    fname_lanes_dict = io_util.load_json(relativepath, lanes_fn)
    vp_dict = io_util.load_json(relativepath, vp_fn)
    vp = vp_dict['vp']

    initimg = cv2.imread(os.path.join(relativepath, filenames[0]), cv2.IMREAD_GRAYSCALE)
    imgshape = initimg.shape
    imgheight, imgwidth = imgshape
    origPts, destPts = image_util.locateROI(imgshape, x_frac=1/3., 
                                            y_frac_top=1/25., y_frac_bot=1/6., 
                                            vp=vp)

    ipm = IPM.IPM(imgshape, imgshape, origPts, destPts)

    lpts = []
    rpts = []
    for i in range(0, 900):
        fname = filenames[i]
        if fname in fname_lanes_dict:
            lanes = fname_lanes_dict[fname]
            j = 0
            for lane in lanes:
                [x1, y1] = ipm.pointToIPM(lane[0])
                [x2, y2] = ipm.pointToIPM(lane[1])
                if j == 0:
                    lpts.append((x1 + x2) / 2.)
                    j = 1
                else:
                    rpts.append((x1 + x2) / 2.)
        else:
            lpts.append(None)
            rpts.append(None)

    l = []
    r = []
    window_size = 1
    for i in range(len(lpts)):
        if i < window_size:
            l.append(0)
        elif lpts[i] is None or lpts[i - window_size] is None:
            l.append(0)
        else:
            l.append(lpts[i] - lpts[i - window_size])
    l = np.array(l)

    for i in range(len(rpts)):
        if i < window_size:
            r.append(0)
        elif rpts[i] is None or rpts[i - window_size] is None:
            r.append(0)
        else:
            r.append(rpts[i] - rpts[i - window_size])
    r = np.array(r)

    def movingaverage (values, window=15):
        weights = np.repeat(1.0, window)/window
        sma = np.convolve(values, weights, 'same')
        return sma

    plt.figure(1)
    #plt.scatter(np.arange(l.shape[0]), l, color='b', label='Raw',s=5)
    #plt.scatter(np.arange(l.shape[0]), movingaverage(l), color='r', label='Average',s=5)
    plt.plot(np.arange(l.shape[0]), (l + r)/2, color='g', label='Raw')
    plt.plot(np.arange(l.shape[0]), movingaverage((l + r)/2), color='r', label='Moving Average')
    plt.plot(np.arange(l.shape[0]), medfilt(movingaverage((l + r)/2), 15), color='b', label='Median and Moving Average')
    plt.ylim([-5,5])
    plt.xlabel('Frame')
    plt.ylabel('$\Delta$x of Lane Boundaries (in pixels)')
    plt.legend(loc='lower left')
    plt.show()

    # plt.figure(2)
    # # plt.scatter(np.arange(r.shape[0]), r, color='b', label='Raw',s=5)
    # # plt.scatter(np.arange(r.shape[0]), movingaverage(r), color='r', label='Average',s=5)
    # plt.plot(np.arange(r.shape[0]), r, color='g', label='Raw')
    # plt.plot(np.arange(r.shape[0]), movingaverage(r), color='r', label='Moving Average')
    # #plt.plot(np.arange(r.shape[0]), medfilt(r, 15), color='b', label='Median')
    # plt.xlabel('Frame')
    # plt.ylabel('$\Delta$x of Right Boundary (in pixels)')
    # plt.legend(loc='lower left')
    # plt.ylim([-5,5])
    # plt.show()

if __name__ == '__main__':
    #relativepath = 'validation/v1_shadows_low_gray'
    relativepath = 'validation/v11_shadows_med_black'
    #relativepath = 'validation/v3_shadows_med_gray'
    #relativepath = 'validation/v6_bright_med_gray'
    #relativepath = 'validation/v7_bright_med_black'

    visualize_lane_changes(relativepath)
