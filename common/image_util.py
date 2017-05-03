import numpy as np
import cv2

# Black out the region in the EBRON region with text, since it is just noise
def block_EBRON(img):
    img[390:460, 460:600] = 0

# Representation of a line (two points)
class Line:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    # Return the y-intercept
    def getIntercept(self):
        slope = self.getSlope()
        b = self.y1 - slope*self.x1
        return b

    # Return the dy/dx slope of the line
    def getSlope(self):
        return (self.y1 - self.y2)/(self.x1 - self.x2 + 1e-10)

    # Return the angle relative to the positive x axis
    def getAngle(self):
        return np.degrees(np.arctan(self.getSlope()))

    # Dot Angle Difference
    def getDotAngle(self, other_line):
        vec1 = np.array([self.x1 - self.x2, self.y1 - self.y2])
        vec2 = np.array([other_line.x1 - other_line.x2, 
                         other_line.y1 - other_line.y2])
        dot = np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
        if dot - 1 < 1e-10:
            return 0
        ang = np.degrees(np.arccos(dot))
        while ang > 90:
            ang -= 180
        return ang

    # Return the dx/dy slope of the line
    def getSlopeXY(self):
        return (self.x1 - self.x2)/(self.y1 - self.y2)

    # Calculate the normalized difference between this line and other_line
    def calculateSlopeDifferenceTo(self, other_line):
        slope1 = self.getSlope()
        slope2 = other_line.getSlope()
        return abs((slope1 - slope2)/(min(slope1, slope2) + 1e-10))

    def calculateAngleDifferenceTo(self, other_line):
        slope1 = self.getSlopeXY()
        slope2 = other_line.getSlopeXY()
        return np.degrees(abs(np.arctan(slope1) - np.arctan(slope2)))

    # Calculate the minimum distance between the two points of this line and
    # other_line
    def calculateLineDifferenceTo(self, other_line):
        diff1 = (self.x1 - other_line.x1)**2 + (self.y1 - other_line.y1)**2
        diff2 = (self.x2 - other_line.x2)**2 + (self.y2 - other_line.y2)**2
        return min(diff1, diff2)

    # Source: http://www.geeksforgeeks.org/
    #                       check-if-two-given-line-segments-intersect/
    # Checks if this line intersects with other_line
    def checkLineIntersection(self, other_line):
        
        def onSegment(p, q, r):
            px, py = p
            qx, qy = q
            rx, ry = r
            if (qx <= max(px, rx) and px >= min(px, rx) and
                qy <= max(py, ry) and qy >= min(py, ry)):
                return True
            return False

        # Returns 0 for colinear, 1 for clockwise, 2 for ccw
        def orientation(p, q, r):
            # See http://www.geeksforgeeks.org/orientation-3-ordered-points/
            # for details of below formula.
            px, py = p
            qx, qy = q
            rx, ry = r
            val = (qy - py) * (rx - qx) - (qx - px) * (ry - qy)
            if val == 0:
                return 0
            elif val > 0:
                return 1
            else:
                return 2

        p1 = (self.x1, self.y1)
        q1 = (self.x2, self.y2)
        p2 = (other_line.x1, other_line.y1)
        q2 = (other_line.x2, other_line.y2)
        
        o1 = orientation(p1, q1, p2)
        o2 = orientation(p1, q1, q2)
        o3 = orientation(p2, q2, p1)
        o4 = orientation(p2, q2, q1)
        # General case
        if (o1 != o2 and o3 != o4):
            return True
        # Special Cases
        if (o1 == 0 and onSegment(p1, p2, q2)):
            return True
        if (o2 == 0 and onSegment(p1, q2, q1)):
            return True
        if (o3 == 0 and onSegment(p2, p1, q2)):
            return True
        if (o4 == 0 and onSegment(p2, q1, q2)):
            return True
        # None of the above
        return False

# Representation of a lane, which has separate lists of x and y coordinates
class Lane:
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys

    def asLine(self):
        return Line(self.xs[0], self.ys[0], self.xs[-1], self.ys[-1])

    def asList(self):
        return [self.xs[0], self.ys[0], self.xs[-1], self.ys[-1]]


# Blocks out (sets to 0) the top fraction (float) width
# of the img (single-channel np array) and returns that.
def blockTopImage(img, fraction):
    output = np.zeros(img.shape)
    top_start = int(fraction * img.shape[0])
    for i in xrange(top_start, img.shape[0]):
        output[i, :] = img[i, :]
    return output

# Thresholds the top percentile pixels and set the rest to 0.
# Find the top percentile pixel values of img, and return
# an image with those pixels set to 255 and rest set to 0.
def thresholdImage(img, percentile=95):
    output = np.zeros(img.shape).astype('uint8')
    threshold = np.percentile(img[np.nonzero(img)], percentile)
    output[img >= threshold] = 255
    return output

# Given an img, its vanishing point (vp as (x, y))
# and the x_frac (2*x_frac of the width, centered on VP, used for top region)
# and the y_frac (y_frac of height above bottom, 1/2*y_frac of height from VP)
# compute the coordinates of a bounding box with 4 point correspondance in the
# original image (origPts), and return that and the corresponding destination 
# points (destPts).
# Use with IPM 
def locateROI(imgshape, x_frac, y_frac_top, y_frac_bot, vp):
    height, width = imgshape
    vp_x, vp_y = vp[0], vp[1]
    x_unit = x_frac * width
    y_unit_top = y_frac_top * height
    y_unit_bot = y_frac_bot * height

    y_bot = int(height - y_unit_bot)
    y_top = int(vp_y + y_unit_top)

    x_top_right = int(vp_x + x_unit)
    x_top_left = int(vp_x - x_unit)

    left_slope = (y_top - vp_y)/float(x_top_left - vp_x)
    right_slope = (y_top - vp_y)/float(x_top_right - vp_x)

    left_intercept = vp_y - left_slope*vp_x
    right_intercept = vp_y - right_slope*vp_x

    x_bot_right = int((y_bot - right_intercept)/right_slope)
    x_bot_left = int((y_bot - left_intercept)/left_slope)

    origPts = [[x_bot_left, y_bot], [x_bot_right, y_bot],
                [x_top_right, y_top], [x_top_left, y_top]]
    destPts = [[0, height], [width, height], [width, 0], [0, 0]]

    return origPts, destPts

# Source: http://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python
# Clips the polygon (list of vertices) with a rectangle (left, top, right, bot)
def clipPolygon(subjectPolygon, clipRectangle):
    oleft, otop, oright, obot = clipRectangle
    clipPolygon = [[oleft, otop], [oright, otop], [oright, obot], [oleft, obot]]
    def inside(p):
        return(cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])
 
    def computeIntersection():
        dc = [ cp1[0] - cp2[0], cp1[1] - cp2[1] ]
        dp = [ s[0] - e[0], s[1] - e[1] ]
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0] 
        n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
        return [(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3]
 
    outputList = subjectPolygon
    cp1 = clipPolygon[-1]
 
    for clipVertex in clipPolygon:
        cp2 = clipVertex
        inputList = outputList
        outputList = []
        s = inputList[-1]
 
        for subjectVertex in inputList:
            e = subjectVertex
            if inside(e):
                if not inside(s):
                    outputList.append(computeIntersection())
                outputList.append(e)
            elif inside(s):
                outputList.append(computeIntersection())
            s = e
        cp1 = cp2
    return(outputList)

# Canny edge detection with preset parameters
def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image.astype('uint8'), lower, upper)
    return edged

# Boxes should be represented as [left, top, right, bot] lists
# Finds the overlapping box between two boxes, or returns None if there
# is no such box
def find_overlap(box1, box2):
    [left1, top1, right1, bot1] = box1
    [left2, top2, right2, bot2] = box2
    if left1 > right2 or left2 > right1:
        return None
    if top1 > bot2 or top2 > bot1:
        return None
    left = max(left1, left2)
    right = min(right1, right2)
    # Because images y-axis are flipped
    top = max(top1, top2)
    bot = min(bot1, bot2)
    overlap_box = np.array([left, top, right, bot])
    return overlap_box

# Compute the area of a given box
def compute_area(box):
    [left, top, right, bot] = box
    return abs(left - right) * abs(top - bot)
