import os, sys, argparse, cv2, numpy as np
np.set_printoptions(precision=2)
code_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
parent_dir = os.path.dirname(code_dir)
sys.path.append(parent_dir)

import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# module imports
from trafficbehavior.common import io_util
from trafficbehavior.common import image_util
from trafficbehavior.common import display
from trafficbehavior.speedestimation import find_nearby_lanes
from trafficbehavior.speedestimation import SpeedRecognizer
from trafficbehavior.speedestimation import main
from trafficbehavior.speedestimation import Filters

def get_validation_dirs(package_dir):
    validation_dir = os.path.join(package_dir, 'validation')
    validation_dirs = []
    for item in os.listdir(validation_dir):
        if item[0] == 'v':
            validation_dirs.append(item)
    #return validation_dirs[-1:]
    return validation_dirs[3:] + validation_dirs[0:3]

def read_true_speed(relativepath, speed_recognizer):
    print 'Extracting true speed from {}'.format(relativepath)
    filenames = io_util.load_fnames(relativepath)
    output_dict = {}
    for fname in filenames:
        img = cv2.imread(os.path.join(relativepath, fname))
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        curr_speed = speedrec.predict(gray_img)
        output_dict[fname] = curr_speed
    return output_dict

def filter_speed(relativepath, speed_fn, plot=True, dt=1/15.):
    print 'Starting filter analysis on {} for {}'.format(relativepath, speed_fn)
    filenames = io_util.load_fnames(relativepath)
    true_speed_dict = io_util.load_json(relativepath, 'true_speed.json')
    #true_speed_dict = io_util.load_json(relativepath, 'speed_readings.json')
    raw = io_util.load_json(relativepath, speed_fn)
    init_speed_est = 75 # in mph
    max_acc = 3.5 # in meters per sec
    names = ['AccLimit', 'KF']
    filters = [
                Filters.AccLimitFilter(init_speed_est, max_acc),
                Filters.KalmanFilter(np.array([[init_speed_est],[0]]),
                            np.array([[1,0],[0,1]]),
                            np.array([[1,0],[0,1]]))
                ]

    est_speeds = [[],[]]
    true_speed = []
    obv_speed = []
    cp = []
    mask = []

    acclimit_dict = {}

    for fname in filenames:
        i = 0
        obv = None
        if fname in raw:
            obv = raw[fname]
        for flter in filters:
            flter.predict(dt)
            if obv and not np.isinf(obv):
                flter.update(obv, dt)
            est = flter.get_estimate()
            if isinstance(est, np.ndarray):
                est = est[0]
            est_speeds[i].append(est)
            if names[i] == 'AccLimit':
                acclimit_dict[fname] = est
            i += 1

        if obv:
            cp.append(obv)
            mask.append(1)
        else:
            cp.append(init_speed_est)
            mask.append(0)
        obv_speed.append(obv)
        true_speed.append(true_speed_dict[fname])

    mask = np.array(mask)
    names.append('Butter LPF')
    names.append('Median Filter')
    names.append('Baseline')

    perc_obv = 1 - np.count_nonzero(np.equal(np.array(obv_speed), None))/float(len(obv_speed))
    print '\t% video w/ est {}'.format(perc_obv)

    if perc_obv < 0.5:
        return None

    est_speeds.append(Filters.butter_lpf(cp, cutoff=1, fs=15, order=6))
    est_speeds.append(Filters.median_filter(cp))
    est_speeds.append(np.array(cp))

    

    ret = []
    der = []
    for i in range(len(names) - 1, -1, -1):
        name = names[i]
        est_arr = np.array(est_speeds[i])
        true_arr = np.array(true_speed)
        l1_diff = np.abs(est_arr[np.where(mask == 1)] - true_arr[np.where(mask == 1)])
        derivative_est = savgol_filter(est_arr, window_length=31, polyorder=2, deriv=1)
        derivative_true = savgol_filter(true_arr, window_length=31, polyorder=2, deriv=1) 
        deriv_diff = np.abs(derivative_est[np.where(mask == 1)] - derivative_true[np.where(mask == 1)])

        # plt.figure(1)
        # plt.title('Using Automatic Markings')
        # plt.plot(np.arange(len(obv_speed)), obv_speed, label='Raw Est Speed', color='g')
        # plt.plot(np.arange(len(true_speed)), true_speed, label='True Speed', color='b')
        # plt.plot(np.arange(len(est_speeds[i])), est_speeds[i], label=names[i], color='r')
        # plt.legend(loc='lower right')
        # plt.ylim([20, 120])
        # plt.ylabel('Speed (MPH)')
        # plt.xlabel('Frame')
        # plt.show()

        # plt.figure(1)
        # plt.title('Using Automatic Markings')
        # plt.plot(np.arange(derivative_est.shape[0]), derivative_est, color='r', label=names[i])
        # plt.plot(np.arange(derivative_true.shape[0]), derivative_true, color='b', label='True speed')
        # plt.legend(loc='lower right')
        # plt.ylim([-0.5, 0.5])
        # plt.ylabel('Acceleration (MPH per second)')
        # plt.xlabel('Frame')
        # plt.show()

        print '\t' + name,
        print 'L1 Diff mean {}, std {}'.format(np.mean(l1_diff), np.std(l1_diff))
        print 'Derivative Diff mean {}, std {}'.format(np.mean(deriv_diff), np.std(deriv_diff))
        ret.append(l1_diff)
        der.append(deriv_diff)
    



    # if plot:
    #     plt.figure(1)
    #     plt.plot(np.arange(len(true_speed)), true_speed, label='True Speed')
    #     plt.plot(np.arange(len(obv_speed)), obv_speed, label='Raw Est Speed')
    #     for i in range(len(names)):
    #         plt.plot(np.arange(len(est_speeds[i])), est_speeds[i], label=names[i])

    #     plt.legend(loc='lower right')
    #     plt.ylim([0, 120])
    #     plt.show()

    return (ret, perc_obv, der, acclimit_dict)

if __name__ == '__main__':
    package_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    run_find_nearby_lines = False
    run_read_true_speed = False
    run_main = False
    run_analysis = True

    if run_find_nearby_lines:
        dirs = get_validation_dirs(package_dir)
        for valid_dir in dirs:
            relativepath = os.path.join(package_dir, 'validation', valid_dir)
            nearby_lanes = find_nearby_lanes.get_nearby_lanes(relativepath)
            io_util.save_json(relativepath, 'nearby_lanes.json', nearby_lanes)

    if run_read_true_speed:
        speedrec = SpeedRecognizer.SpeedRecognizer()
        dirs = get_validation_dirs(package_dir)
        for valid_dir in dirs:
            relativepath = os.path.join(package_dir, 'validation',  valid_dir)
            save_dict = read_true_speed(relativepath, speedrec)
            io_util.save_json(relativepath, 'true_speed.json', save_dict)

    if run_main:
        dirs = get_validation_dirs(package_dir)
        for valid_dir in dirs:
            relativepath = os.path.join(package_dir, 'validation',  valid_dir)
            main.speed_estimation_main(relativepath, True)

    if run_analysis:
        dirs = get_validation_dirs(package_dir)
        ltot = [[],[],[],[],[]]
        rtot = [[],[],[],[],[]]
        for valid_dir in dirs:
            relativepath = os.path.join(package_dir, 'validation',  valid_dir)
            #lret = filter_speed('../Videos/SD/i5_2', 'estimated_speed_closing.json')
            #rret = None
            lret = filter_speed(relativepath, 'speed_raw_left.json')
            rret = filter_speed(relativepath, 'speed_raw_right.json')
            # filter_speed(relativepath, 'speed_raw_left_uni.json')
            # filter_speed(relativepath, 'speed_raw_right_uni.json')
            perc_left = 0
            perc_right = 0
            if lret is not None:
                perc_left = lret[1]
            if rret is not None:
                perc_right = rret[1]

            if (perc_left > perc_right):
                io_util.save_json(relativepath, 'speed_acclimit.json', lret[3])
                for i in range(len(lret[0])):
                    ltot[i] = ltot[i] + lret[0][i].tolist()
                    rtot[i] = rtot[i] + lret[2][i].tolist()
            elif (perc_left < perc_right):
                io_util.save_json(relativepath, 'speed_acclimit.json', rret[3])
                for i in range(len(rret[0])):
                    ltot[i] = ltot[i] + rret[0][i].tolist()
                    rtot[i] = rtot[i] + rret[2][i].tolist()
        names = ['AccLimit', 'KF', 'Butter', 'Median', 'Baseline']
        names = names[::-1]
        print "--"
        for i in range(len(ltot)):
            print '\t' + names[i],
            print 'Total L1 Diff mean {}, std {}'.format(np.mean(ltot[i]), np.std(ltot[i]))
            print 'Total Derivative Diff mean {}, std {}'.format(np.mean(rtot[i]), np.std(rtot[i]))

