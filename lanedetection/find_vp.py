import os, sys, argparse

code_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
parent_dir = os.path.dirname(code_dir)
sys.path.append(parent_dir)

import cv2
from trafficbehavior.common import LaneMarker
from trafficbehavior.common import VPEstimator
from trafficbehavior.common import image_util
from trafficbehavior.common import io_util
from trafficbehavior.common import display
import numpy as np

def get_vps(relativepath, boxes_fn):
    lanewidth = 5
    filenames = io_util.load_fnames(relativepath)
    init_img = cv2.imread(os.path.join(relativepath, filenames[0]), 
                                        cv2.IMREAD_GRAYSCALE).astype('uint8')
    lanemarker = LaneMarker.NietoLaneMarker()
    vpEstimator = VPEstimator.MSAC_LS(init_img.shape)

    boxes_dict = io_util.load_json(relativepath, boxes_fn)
    data = []
    for i in range(len(filenames)):
        fname = filenames[i]
        display.show_progressbar(i, filenames)
        color_img = cv2.imread(os.path.join(relativepath, filenames[i]))
        gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY).astype('uint8')
        # block region of the image - top, text, and vehicles
        blockedimg = image_util.blockTopImage(gray_img, 0.6)
        #cv2.imshow('block',blockedimg.astype('uint8'))
        image_util.block_EBRON(blockedimg)
        if fname in boxes_dict:
            for box_dict in boxes_dict[fname]:
                [x1, y1, x2, y2] = box_dict['box']
                blockedimg[y1:y2, x1:x2] = 0
                color_img[y1:y2, x1:x2] = 0

        threshimg = image_util.thresholdImage(lanemarker.detect(blockedimg, lanewidth))
        lines = cv2.HoughLines(threshimg, 1, cv2.cv.CV_PI/180, 30)
        if lines is None:
            continue
        linesegments = VPEstimator.preprocess_lines(lines[0])
        if len(linesegments) < 5:
            continue
        vps, lineSegmentsClusters, numInliers = vpEstimator.multipleVPEstimation(linesegments, 1)

        for line in linesegments:
            [x1, y1, x2, y2] = line
            cv2.line(threshimg, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 1)
            cv2.line(color_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
        if vps is not None:
            x, y = vps[0][0][0], vps[0][1][0]
            cv2.circle(color_img, (int(x), int(y)), 5, (255, 0, 0), thickness=3)
            cv2.circle(threshimg, (int(x), int(y)), 5, (255, 255, 255),thickness=3)
            data.append([x, y])
        cv2.imshow('color image', color_img)
        #cv2.imshow('thresh image', threshimg)
        cv2.waitKey(1pospost)
        #break
    print ""
    cv2.destroyAllWindows()

    return np.array(data)

def DB_Cluster(X, plot=True):
    from sklearn.cluster import DBSCAN
    from sklearn import metrics
    from sklearn.preprocessing import StandardScaler
    

    XC = StandardScaler().fit_transform(X)
    db = DBSCAN(eps=0.3, min_samples=10).fit(XC)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    # this one looks at how close it is to the center
    mean_pt = None
    center_pt = np.array([320, 240])
    best_norm = None
    for i in range(n_clusters_):
        new_mean_pt = np.mean(X[np.where(labels == i)], axis=0)
        new_norm = np.linalg.norm(new_mean_pt - center_pt)
        if (mean_pt is None or new_norm < best_norm):
            mean_pt = new_mean_pt
            best_norm = new_norm

    # This one looks at frequency:
    # counts = np.bincount(labels[np.where(labels != -1)])
    # center_label = np.argmax(counts)
    # mean_pt = np.mean(X[np.where(labels == center_label)], axis=0)

    if plot:
        import matplotlib.pyplot as plt
        print('Estimated number of clusters: %d' % n_clusters_)
        plt.figure(1)
        # Black removed and is used for noise instead.
        unique_labels = set(labels)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = 'k'

            class_member_mask = (labels == k)

            xy = X[class_member_mask & core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                     markeredgecolor='k', markersize=14)

            xy = X[class_member_mask & ~core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                     markeredgecolor='k', markersize=6)

        plt.title('Estimated number of clusters: %d' % n_clusters_)
        plt.ylim([480, 0])
        plt.xlim([0, 640])
        plt.show()

    return mean_pt

def MS_Cluster(X, plot=False):
    from sklearn.cluster import MeanShift, estimate_bandwidth
    from sklearn.preprocessing import StandardScaler

    XC = StandardScaler().fit_transform(X)
    bandwidth = estimate_bandwidth(XC, quantile=0.2)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(XC)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    counts = np.bincount(labels[np.where(labels != -1)])
    center_label = np.argmax(counts)
    mean_pt = np.mean(X[np.where(labels == center_label)], axis=0)

    if plot:
        import matplotlib.pyplot as plt
        from itertools import cycle
        print("number of estimated clusters : %d" % n_clusters_)

        plt.figure(1)
        plt.clf()

        colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
        for k, col in zip(range(n_clusters_), colors):
            my_members = labels == k
            cluster_center = cluster_centers[k]
            plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
            plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                     markeredgecolor='k', markersize=14)
        plt.title('Estimated number of clusters: %d' % n_clusters_)
        plt.ylim([480, 0])
        plt.xlim([0, 640])
        plt.show()

    return mean_pt

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-v', '--videopath', help='Relative Path to Video', 
                    default='./validation/v1_shadows_low_gray')
    ap.add_argument('-b', '--boxes', help='Filename for boxes', 
                    default='boxes_output_detector_session2_raw_8.5.json')
    args = ap.parse_args()
    relativepath = args.videopath
    boxes_fn = args.boxes

    run_vps = True
    run_clustering = True
    save_vp = True

    if run_vps:
        data = get_vps(relativepath, boxes_fn)
        np.save(os.path.join(relativepath, 'vp.npy'), data)
    else:
        data = np.load(os.path.join(relativepath, 'vp.npy'))

    if run_clustering:
        mean_pt = DB_Cluster(data)
        vp_dict = {'vp':mean_pt}
        # MS_Cluster(data)
        print mean_pt
        if save_vp:
            io_util.save_json(relativepath, 'vp.json', vp_dict)
            print 'Saved as json'


    


