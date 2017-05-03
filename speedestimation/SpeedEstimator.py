import numpy as np, cv2, json

from trafficbehavior.common import IPM
from trafficbehavior.common import LaneMarker
from trafficbehavior.common import image_util
from trafficbehavior.common import io_util

import SpeedRecognizer

class Lane_SpeedEstimator:
    def __init__(self, lanemarker, slope=None, b=None, dt=1/15.):
        self.dt = dt
        self.prev_pts_L = []
        self.prev_speed_L = None
        self.prev_pts_R = []
        self.prev_speed_R = None
        self.lanemarker = lanemarker
        self.feet_to_miles = 0.681818
        self.slope = slope
        self.b = b

    def find_contour_ends(self, cnt):
        pt_top = None
        pt_bot = None
        for pt in cnt:
            pt_x = pt[0][0]
            pt_y = pt[0][1]
            if pt_top is None or pt_y < pt_top[1]:
                pt_top = pt[0]
            if pt_bot is None or pt_y > pt_bot[1]:
                pt_bot = pt[0]
        return pt_top, pt_bot

    def getLineBoundingImage(self, line, image, w_frac=0.05):
        height, width = image.shape
        xmin = int(np.rint(max(0, min(line.x1, line.x2) - width*w_frac)))
        xmax = int(np.rint(min(width, max(line.x1, line.x2) + width*w_frac)))
        ymin = 0
        ymax = height
        return image[ymin:ymax, xmin:xmax]

    def computeXLineDifferences(self, lineL, lineR):
        slope1 = lineL.getSlopeXY()
        slope2 = lineR.getSlopeXY()
        b1 = lineL.x1 - slope1 * lineL.y1
        b2 = lineR.x1 - slope1 * lineR.y1
        if b2 > b1:
            return slope2 - slope1, b2 - b1
        else:
            return slope1 - slope2, b1 - b2

    def update_single(self, image, line, is_left, slope, b):
        if is_left:
            prev_pts = self.prev_pts_L
            prev_speed = self.prev_speed_L
        else:
            prev_pts = self.prev_pts_R
            prev_speed = self.prev_speed_R

        imgheight = image.shape[0]

        thresh = image_util.thresholdImage(self.lanemarker.detect(image, 5))
        lane_img = self.getLineBoundingImage(line, thresh)

        gray = lane_img.astype('uint8')
        kernel = np.ones((12,3),np.uint8)
        closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        #closing = gray

        # cv2.imshow('gray', gray)
        # cv2.imshow('closing', closing)
        contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, 
                                                cv2.CHAIN_APPROX_SIMPLE)
        i = 0
        blank = np.zeros(gray.shape)
        curr_pts = []
        for cnt in contours:
            # For each contour find the top and bot most points of the contour
            pt_top, pt_bot = self.find_contour_ends(cnt)
            if pt_top is not None and pt_bot is not None:
                dy = abs(pt_top[1] - pt_bot[1])
                if (dy >= imgheight/10. and dy <= imgheight/2. 
                    and pt_top[1] > 5 and pt_bot[1] < image.shape[0] - 5):
                    cv2.drawContours(blank, contours, i, (255, 255, 255))
                    width = 8
                    pixels_from_bot = 480 - np.linspace(pt_top[1], pt_bot[1], width)
                    conversions = 12/(slope*np.linspace(pt_top[1], pt_bot[1], width) + b)
                    curr_pts.append((pixels_from_bot)*conversions)
            i += 1

        mean_speed = None
        mean_speeds = []
        for curr_pt in curr_pts:
            best_mean_speed = None
            best_prev_pt = None
            for prev_pt in prev_pts:
                if np.mean(prev_pt - curr_pt) < 0:
                    continue
                speed_estimate = (prev_pt - curr_pt)*self.feet_to_miles/self.dt
                mean_speed = np.mean(speed_estimate)
                if prev_speed and (best_mean_speed is None 
                    or abs(mean_speed - prev_speed) < best_mean_speed):
                        best_mean_speed = abs(mean_speed - prev_speed)
                        best_prev_pt = prev_pt
                elif best_mean_speed is None or mean_speed < best_mean_speed:
                        best_mean_speed = mean_speed
                        best_prev_pt = prev_pt
            if best_mean_speed:
                mean_speeds.append((best_prev_pt - curr_pt)*self.feet_to_miles/self.dt)

        mean_speed = np.mean(np.array(mean_speeds))
        if np.isnan(mean_speed):
            mean_speed = None

        if (is_left):
            cv2.imshow('contours', blank)
        #cv2.waitKey(1)

        if is_left:
            self.prev_pts_L = curr_pts
            self.prev_speed_L = mean_speed
        else:
            self.prev_pts_R = curr_pts
            self.prev_speed_R = mean_speed

        return mean_speed


    def update(self, image, lineL, lineR):
        if self.slope is None or self.b is None:
            slope, b = self.computeXLineDifferences(lineL, lineR)
        else:
            slope, b = self.slope, self.b
        L_est = self.update_single(image, lineL, True, slope, b)
        R_est = self.update_single(image, lineR, False, slope, b)
        return L_est, R_est

    def update_man(self, image, lineL, lineR, line1, line2):
        slope, b = self.computeXLineDifferences(line1, line2)
        L_est = self.update_single(image, lineL, True, slope, b)
        R_est = self.update_single(image, lineR, False, slope, b)
        return L_est, R_est





