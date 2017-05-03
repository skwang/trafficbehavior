import numpy as np
import cv2, collections
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import RANSACRegressor
from sklearn import linear_model
from sklearn.pipeline import make_pipeline

from trafficbehavior.common import image_util

# For tracking a list of possible lanes over a fixed number of frames, and if
# a lane appears in a consistent area in enough of them, returns it for
# tracking. Simple method to avoid False Positives
class PossibleLaneTracker:
    def __init__(self, img_shape, number_to_track=5):
        # List of deque of boxes for each last time a possible object 
        # was detected. Lists have None if no object was detected there frame
        self.possible_lanes_list = []
        # How many frames to track potential objects
        self.max_size = number_to_track
        # How many times an object has to be tracked before we return it as
        # an object to track. If an object is found decided_count times out of
        # the last max_size frames, it is returned.
        self.decided_count = 2
        self.img_shape = img_shape
        # amount of area overlapping to consider it to be the same object as
        # found in the previous frame
        self.line_diff_threshold = img_shape[0]*img_shape[1]/5.
        self.angle_diff_threshold = 5

    # potential_new_boxes should be an iterable
    def update(self, potential_new_lanes):
        new_lanes = [] # list of boxes that do not have a match
        # List of booleans of whether a possible tracked object was updated
        updated_lane_list = [False]*len(self.possible_lanes_list)

        for lane in potential_new_lanes:
            line = lane.asLine()
            found_match = False
            # For each possible tracked object, compute average overlap
            # with the previous items in the list. If it is above a certain
            # threshold, add new box to the list, and mark that possible
            # object as updated
            for ii in xrange(len(self.possible_lanes_list)):
                prev_lanes = self.possible_lanes_list[ii]
                average_diff = 0.
                average_angle = 0.
                number_not_none = 0.
                for prev_lane in prev_lanes:
                    if prev_lane is not None:
                        prev_line = prev_lane.asLine()
                        number_not_none += 1
                        average_diff += line.calculateLineDifferenceTo(
                                                                    prev_line)
                        average_angle += line.calculateAngleDifferenceTo(
                                                                    prev_line)
                if number_not_none != 0:
                    average_diff /= number_not_none
                    average_angle /= number_not_none
                    if (average_diff < self.line_diff_threshold 
                      and average_angle < self.angle_diff_threshold):
                        prev_lanes.append(lane)
                        if len(prev_lanes) > self.max_size:
                            prev_lanes.popleft()
                        updated_lane_list[ii] = True
                        found_match = True
                        break

            # If did not find a match, mark it as a new box, and make
            # a deque to track it later
            if not found_match:
                new_lanes.append(lane)

        # Append None for possible objects that weren't updated
        for ii in xrange(len(updated_lane_list)):
            if not updated_lane_list[ii]:
                prev_lanes = self.possible_lanes_list[ii]
                prev_lanes.append(None)
                if len(prev_lanes) > self.max_size:
                        prev_lanes.popleft()

        # Add new boxes to the list
        for lane in new_lanes:
            new_deque = collections.deque()
            new_deque.append(lane)
            self.possible_lanes_list.append(new_deque)

        # Check if possible object. If it has at least self.decided_count
        # non-None objects, use the most recent box as an object. Otherwise
        # if it is the max_size and all None's, stop tracking it.
        return_lanes = []
        i = 0
        while i < len(self.possible_lanes_list):
            lanes = self.possible_lanes_list[i]
            number_not_none = 0
            for lane in lanes:
                if lane is not None:
                    number_not_none += 1
            if number_not_none >= self.decided_count:
                for lane in reversed(lanes):
                    if lane is not None:
                        return_lanes.append(lane)
                        break
                del self.possible_lanes_list[i]
                i -= 1
            elif number_not_none == 0 and len(lanes) == self.max_size:
                del self.possible_lanes_list[i]
                i -= 1
            i += 1
        return return_lanes

# Class for tracking a single line. Initialize with location of a line. 
# Pass updates with a new image, a lane_detector object, and the line tracker
# will return the lane_detector of searching a box around the previous line
class LineTracker:
    def __init__(self, line, img_shape):
        self.line = line
        self.angle_diff_threshold = 5
        self.line_diff_threshold = img_shape[0]*img_shape[1]/100.

    def get_line(self):
        return self.line

    def update(self, new_image, laneDetector, degree=1):
        old_line = self.get_line()
        box = laneDetector.getLineBoundingBox(old_line, new_image.shape)
        filtered_img = laneDetector.boxMaskImage(new_image, box)
        self.imgarea = new_image.shape[0]*new_image.shape[1]
        if filtered_img is not None:
            try:
                new_lane = laneDetector.fitRANSAC(filtered_img, degree)
                new_line = new_lane.asLine()
                lineDiff = old_line.calculateLineDifferenceTo(new_line)
                angleDiff = old_line.calculateAngleDifferenceTo(new_line)
                #print '\tangleDiff {}'.format(angleDiff)
                if lineDiff > self.line_diff_threshold:
                    return None
                elif angleDiff > self.angle_diff_threshold:
                    # cv2.imshow('fi', filtered_img)
                    # tmp_ = cv2.cvtColor(filtered_img.astype('uint8'), cv2.COLOR_GRAY2BGR)
                    # cv2.line(tmp_, (old_line.x1,old_line.y1), (old_line.x2,old_line.y2), (0,255,0),1)
                    # cv2.line(tmp_, (new_line.x1,new_line.y1), (new_line.x2,new_line.y2), (0,0,255),1)
                    # cv2.imshow('AngleDifference, green old, red bad',tmp_)
                    # cv2.waitKey(0)
                    #print '\tangleDiff over threshold',
                    #print angleDiff
                    return None
                else:
                    self.line = new_line
                    return new_lane
            except ValueError:
                print 'Lost lane'
                return None

class LaneDetectorModel:
    def __init__(self, initial_lanes, image_shape, laneDetector):
        self.previous_lines = []
        self.image_shape = image_shape
        self.laneDetector = laneDetector
        for lane in initial_lanes:
            line = lane.asLine()
            self.previous_lines.append(LineTracker(line, image_shape))
        self.possibleLaneTracker = PossibleLaneTracker(image_shape)
        self.new_lanes = []

    def get_new_lanes(self):
        return self.new_lanes

    def updateLaneModel(self, laneImg, degree=1):
        new_lines = [] # the new lines to keep track of after this update
        lanes = []     # lanes to be returned
        self.new_lanes = []
        # a copy of the laneImg. We will block out the bounding boxes of
        # all the previous lines before looking for new lanes here
        copyLaneImg = np.copy(laneImg)
        imgarea = laneImg.shape[0]*laneImg.shape[1]

        # For each previous line, get the bounding box, 0 out the area in the
        # copy of the lane img, then look for a new lane in that bounding box.
        # If there is a new line that does not intersect with other new lines
        # and is not within a certain distance of the other new lines, add it
        # to the list of new lines to keep track of
        for line_tracker in self.previous_lines[:]:
            line = line_tracker.get_line()
            box = self.laneDetector.getLineBoundingBox(line, laneImg.shape)
            xmin = box[0]
            xmax = box[1]
            ymin = box[2]
            ymax = box[3]
            copyLaneImg[xmin:xmax, ymin:ymax] = 0

            updated_lane = line_tracker.update(laneImg, self.laneDetector, degree)
            if updated_lane is not None:
                found_overlap = False
                new_line = line_tracker.get_line()
                for other_line_tracker in new_lines:
                    other_line = other_line_tracker.get_line()
                    if new_line.checkLineIntersection(other_line):
                        found_overlap = True
                        break
                    elif new_line.calculateLineDifferenceTo(other_line) < imgarea/100.:
                        found_overlap = True
                        break
                if not found_overlap:
                    lanes.append(updated_lane)
                    new_lines.append(line_tracker)

        # Detect lanes in the copy of laneImg. 
        # If there is a new line that does not intersect with other new lines
        # and is not within a certain distance of the other new lines, add it
        # to the list of new lines to keep track of
        potential_new_lanes = []
        new_lanes = self.laneDetector.detectLanes(copyLaneImg, degree)
        if new_lanes is not None:
            for new_lane in new_lanes:
                new_line = new_lane.asLine()
                found_overlap = False
                for other_line_tracker in new_lines:
                    other_line = other_line_tracker.get_line()
                    if new_line.checkLineIntersection(other_line):
                        found_overlap = True
                        break
                    elif new_line.calculateLineDifferenceTo(other_line) < imgarea/100.:
                        found_overlap = True
                        break
                if not found_overlap:
                    potential_new_lanes.append(new_lane)

        lanes_to_track = self.possibleLaneTracker.update(potential_new_lanes)
        for lane in lanes_to_track:
            lanes.append(lane)
            new_line = lane.asLine()
            new_lines.append(LineTracker(new_line, self.image_shape))
            self.new_lanes.append(lane)

        self.previous_lines = new_lines
        for new_line_tracker in new_lines:
            new_line = new_line_tracker.get_line()
            pt1 = (new_line.x1, new_line.y1)
            pt2 = (new_line.x2, new_line.y2)
            cv2.line(laneImg, pt1, pt2, (255,0,0),5)
        # cv2.imshow('Lanes', laneImg.astype('uint8'))
        # cv2.waitKey(1)

        return lanes


class PolyRANSAC_LaneDetector:
    def __init__(self, houghTransformLength=25, verbose=False):
        self.houghTransformLength = houghTransformLength
        self.verbose = verbose
        self.mad = 0 # median absolute deviation

    # imgsize = (width, height)
    # line = [x1, x2, y1, y2]
    # Given a line and imgsize, return the bounding box for that line
    # as [xmin, xmax, ymin, ymax], with the box width extended by w_frac*width
    # on each side
    def getLineBoundingBox(self, line, imgshape, w_frac=0.05):
        height, width = imgshape
        #x1, x2, y1, y2 = line
        xmin = max(0, min(line.x1, line.x2) - np.round(width*w_frac))
        xmax = min(width, max(line.x1, line.x2) + np.round(width*w_frac))
        ymin = max(0, min(line.y1, line.y2) - np.round(height*w_frac))
        ymax = min(height, max(line.y1, line.y2) + np.round(height*w_frac))
        return np.array([xmin, xmax, ymin, ymax]).astype(int)

    # Given an img and a bounding box as [xmin, xmax, ymin, ymax], 
    # return the image with all black except the box
    # If the img has mostly zero values, return None
    def boxMaskImage(self, img, box):
        newimg = np.zeros(img.shape)
        xmin = box[0]
        xmax = box[1]
        ymin = box[2]
        ymax = box[3]
        newimg[ymin:ymax,xmin:xmax] = img[ymin:ymax,xmin:xmax]
        if np.count_nonzero(newimg) <= 5:
            return None
        return newimg

    # Convert the lines from cv2.HoughLines format (rho, theta)
    # to [x1, x2, y1, y2] where y1 = 0 and y2 = height of image
    def convertLines(self, lines, imgshape):
        height, width = imgshape
        vs = np.array([]).reshape(0,4)
        for line in lines[0]:
            rho = line[0]
            theta = line[1]
            a = np.cos(theta)
            x = a*rho
            b = np.sin(theta)
            y = b*rho
            y1 = 0
            y2 = height
            if a == 0:
                a = 1e-6
            s1 = (y1 - y)/a
            x1 = np.round(x + s1 * (-b))
            x2 = np.round(x - s1 * (-b))

            v = np.array([x1, y1, x2, y2])
            vs = np.vstack([vs, v])
        return vs

    # Cluster the lines (given in [x1, x2, y1, y2]) using MeanShift
    # with the quantile (median of pairwise distances used) 
    def clusterLines(self, lines, quantile):
        # Meanshift clustering runs into errors when we have isolated lanes
        # To handle, add lines on either side
        newlines = np.array([]).reshape(0,4)
        for line in lines:
            newlines = np.vstack([newlines, line])
            [x1, y1, x2, y2] = line
            line1 = np.array([x1+0.01, y1, x2+0.01, y2])
            line2 = np.array([x1-0.01, y1, x2-0.01, y2])
            newlines = np.vstack([newlines, line1])
            newlines = np.vstack([newlines, line2])  

        bandwidth = estimate_bandwidth(newlines, quantile=quantile)
        while bandwidth == 0:
            print newlines
            quantile = quantile*2.
            bandwidth = estimate_bandwidth(newlines, quantile=quantile)
            print bandwidth
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(newlines)
        cluster_centers = ms.cluster_centers_
        if len(cluster_centers.shape) > 2:
            cluster_centers = cluster_centers[0]
        if self.verbose:
            labels = ms.labels_
            labels_unique = np.unique(labels)
            n_clusters_ = len(labels_unique)
            print "{} clusters: ".format(n_clusters_)
            print cluster_centers

        cluster_centers_lines = []
        for center in cluster_centers:
            x1, y1, x2, y2 = center
            new_line = image_util.Line(x1, y1, x2, y2)
            cluster_centers_lines.append(new_line)
        return cluster_centers_lines

    # Data validation for RANSAC, returns false when there are no inliers
    # which forces the algorithm to pick another sample. Needed to avoid
    # an exception when there are no inliers (special case)
    def is_data_valid(self, X, y):
        lr = linear_model.LinearRegression()
        lr.fit(X, y)
        inliers = (np.power(np.absolute(y - lr.predict(X)), 2) - self.mad)
        if np.count_nonzero(inliers <= 0) == 0:
            return False
        return True
    
    # Fit a RANSAC polynomial of degree, vertically on the 
    # nonzero pixels of maskedImg. Return either the partial 
    # or full lines in IPM coordinates as [[x pts], [y pts]] 
    def fitRANSAC(self, maskedImg, degree, partial=False):
        nonzeroind = np.where(maskedImg > 0)
        rY = nonzeroind[1].reshape(-1, 1)
        rX = nonzeroind[0].reshape(-1, 1)

        # calculate median absolute deviance
        self.mad = np.median(np.absolute(rY - np.median(rY)))
        # if it is 0, then all Y values are the same
        if self.mad == 0:
            fitY = np.arange(0, maskedImg.shape[0]).reshape(-1, 1)
            fitX = np.zeros(fitY.shape) + np.median(rY)
            return image_util.Lane(fitX, fitY)

        estimator = RANSACRegressor(random_state=42, min_samples=3, 
                                    is_data_valid=self.is_data_valid)
        model = make_pipeline(PolynomialFeatures(degree), estimator)
        model.fit(rX, rY)
        if partial:
            fitY = np.arange(np.min(rX), np.max(rX)).reshape(-1, 1)
        else:
            fitY = np.arange(0, maskedImg.shape[0]).reshape(-1, 1)
        fitX = model.predict(fitY)
        return image_util.Lane(fitX, fitY)


    # given a thresholded IPM image, detect the lanes in the image
    # using the following routine:
    # 1. Hough Transform to find lines
    # 2. MeanShift clustering for the lines
    # 3. Get bounding box for each cluster center
    # 4. Fit RANSAC polynomial of degree to the bounding box region
    # Returns the lanes as a list, in IPM coordinates, of form x, y
    def detectLanes(self, laneImg, degree):
        height, width = laneImg.shape
        
        # 1. Hough Transform
        # 40/2 for the commaai
        lines = cv2.HoughLines(laneImg, 1, cv2.cv.CV_PI/180,
                                self.houghTransformLength) # TODO: magic numbers
        # If we cannot detect any lines, return None
        if lines is None:
            return None
        vs = self.convertLines(lines, laneImg.shape)
        if self.verbose:
            tmp_img = cv2.cvtColor(laneImg, cv2.COLOR_GRAY2BGR)
            for line_points in vs:
                [x1, y1, x2, y2] = line_points.astype(np.int32)
                pt1 = (x1, y1)
                pt2 = (x2, y2)
                try:
                    cv2.line(tmp_img, pt1, pt2, (0,0,255),1)
                except OverflowError:
                    # Sometimes the x-value of the points will overflow and not
                    # be able to be drawn. In this case, we just skip drawing it
                    continue
            cv2.imshow('hough transform lines', tmp_img)
            cv2.waitKey(1)

        # If not enough lines, just return None, we cannot detect lanes.
        if vs.shape[0] < 2:
            return None

        # 2. Meanshift clustering
        cluster_lines = self.clusterLines(vs, 0.25) # TODO: magic numbers
        # (Prune lines that are too diagonal)
        prune_width = np.round(width/10)
        lines_pruned = []
        for line in cluster_lines:
            if abs(line.x1 - line.x2) < prune_width:
                lines_pruned.append(line)

        if self.verbose:
            tmp_img = cv2.cvtColor(laneImg, cv2.COLOR_GRAY2BGR)
            for line_points in vs:
                [x1, y1, x2, y2] = line_points.astype(np.int64)
                pt1 = (x1, y1)
                pt2 = (x2, y2)
                if abs(x1 - x2) < prune_width:
                    # Red for lines we keep
                    cv2.line(tmp_img, pt1, pt2, (0,0,255),1)
                else:
                    # Blue for lines we get rid of
                    try: 
                        cv2.line(tmp_img, pt1, pt2, (255,0,0),1)
                    except OverflowError:
                        # Sometimes the x-value of the points will overflow and
                        # cannot be drawn. In this case, we just skip drawing it
                        continue
            cv2.imshow('pruned lines (red keep, blue not)', tmp_img)
            cv2.waitKey(1)

        # 3. Get bounding boxes for clusters
        boxes = np.array([], dtype='int').reshape(0,4)
        for line in lines_pruned:
            box = self.getLineBoundingBox(line, laneImg.shape)
            boxes = np.vstack([boxes, box])        
        if self.verbose:
            tmp_img = cv2.cvtColor(laneImg, cv2.COLOR_GRAY2BGR)
            for box in boxes:
                [xmin, xmax, ymin, ymax] = box.astype(np.int64)
                pt1 = (xmin, ymin + 2)
                pt2 = (xmax, ymax - 2)
                cv2.rectangle(tmp_img, pt1, pt2, (255, 0, 0), 2)
            cv2.imshow('bounding boxes', tmp_img)
            cv2.waitKey(1)
        
        # 4. Fit RANSAC Polynomial
        lanes = []
        for box in boxes:
            filtered_img = self.boxMaskImage(laneImg, box)
            if filtered_img is not None:
                predicted_lane = self.fitRANSAC(filtered_img, degree)
                lanes.append(predicted_lane)

        if self.verbose:
            tmp_img = cv2.cvtColor(laneImg, cv2.COLOR_GRAY2BGR)
            for lane in lanes:
                for i in xrange(0, len(lane.xs) - 1):
                    pt1 = (lane.xs[i], lane.ys[i])
                    pt2 = (lane.xs[i+1], lane.ys[i+1])
                    cv2.line(tmp_img, pt1, pt2, (0,255,0),1)
            cv2.imshow('RANSAC fit lanes', tmp_img)
            cv2.waitKey(1)

        return lanes
