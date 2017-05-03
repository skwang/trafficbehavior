import sys, numpy as np, cv2

def preprocess_lines(lines):
    linesegments = []

    for line in lines:
        rho = line[0]
        theta = line[1]
        angle = np.degrees(theta)
        while angle >= 180:
            angle -= 180
        while angle < 0:
            angle += 180
        a = np.cos(theta)
        x = a*rho
        b = np.sin(theta)
        y = b*rho
        x1 = np.round(x + 1000*(-b))
        y1 = np.round(y + 1000*a)
        x2 = np.round(x - 1000*(-b))
        y2 = np.round(y - 1000*a)
        if angle < 150 and angle > 30:
            linesegments.append([x1, y1, x2, y2])
    return linesegments

class MSAC_LS:
    def __init__(self, imSize, verbose=False):
        # Arguments
        self.__width = imSize[1]  # Image width
        self.__height = imSize[0] # Image height
        self.__verbose = verbose  # To print debugging messages

        # MSAC parameters
        self.__epsilon = 1e-6
        self.__T_noise_squared = 0.01623**2 # something to play with
        self.__min_iters = 10 # another thing to play with
        self.__max_iters = sys.maxint

        # minimal number of lines needed to estimate VP
        self.__minimal_sample_set_dimension = 2

        # Minimal sample set
        self.__MSS = []
        for i in xrange(0, self.__minimal_sample_set_dimension):
            self.__MSS.append(0)

        # Set default camera calibration matrix
        self.__K = np.zeros((3, 3))
        self.__K[0, 0] = self.__width
        self.__K[0, 2] = self.__width/2
        self.__K[1, 1] = self.__height
        self.__K[1, 2] = self.__height/2
        self.__K[2, 2] = 1

        self.__Li = None # Matrix of appended line segments for N line segments (N x 3)
        self.__Mi = None # Matrix of middle points (N X 3)
        self.__Lengths = None # Matrix of lengths (N x N)

        self.__CS_idx = None
        self.__CS_best = None

    # Helper function that formats lineSegments into normalized form
    # Inputs: lineSegments as a list of line segments given as [p1.x, p1.y, p2.x, p2.y]
    # Returns: None (fill class variables Li, Mi, Lengths)
    def fillDataContainers(self, lineSegments):
        numLines = len(lineSegments)
        if (self.__verbose):
            print "Number of line segments: {}".format(numLines)

        self.__Li = np.zeros((numLines, 3))
        self.__Mi = np.zeros((numLines, 3))
        self.__Lengths = np.zeros((numLines, numLines))

        sum_lengths = 0.
        for i in xrange(numLines):
            line = lineSegments[i]
            # treat line endpoints as homogeneous coordinates
            a = np.zeros((3, 1))
            a[0, 0] = line[0]
            a[1, 0] = line[1]
            a[2, 0] = 1

            b = np.zeros((3, 1))
            b[0, 0] = line[2]
            b[1, 0] = line[3]
            b[2, 0] = 1

            # calculate line length
            line_len = np.sqrt((line[2] - line[0])**2 + (line[3] - line[1])**2)
            sum_lengths += line_len

            self.__Lengths[i, i] = line_len

            # Invert with the camera projection matrix to get global coordinates
            an = np.dot(np.linalg.inv(self.__K), a)
            bn = np.dot(np.linalg.inv(self.__K), b)

            # Compute cross product and normalize
            li = np.cross(an[:,0], bn[:,0])
            li = li / np.linalg.norm(li)

            self.__Li[i, 0] = li[0]
            self.__Li[i, 1] = li[1]
            self.__Li[i, 2] = li[2]

        self.__Lengths = self.__Lengths*(1/sum_lengths)

    # Least Squares Estimation Function
    # Inputs: Li (N x 3 Matrix of normalized lines), Lengths (N x N Matrix of line lengths),
    #         ind_set (list of indices to consider from Li), set_length (number of indices to consider)
    # Returns: vanishing point (3 x 1 matrix)
    def estimateLS(self, Li, Lengths, ind_set, set_length):
        vp = np.zeros((3, 1))

        if (set_length < self.__minimal_sample_set_dimension):
            print "Error: at least 2 line-segments required"
            return None
        # If only two indices, intersection is just the cross product
        elif (set_length == self.__minimal_sample_set_dimension):
            ls0 = Li[ind_set[0],:]
            ls1 = Li[ind_set[1],:]
            vp = np.cross(ls0, ls1)
            vp = vp / np.linalg.norm(vp)
            return vp
        # For more than two lines, use SVD
        else:
            li_set = np.zeros((3, set_length))
            Lengths_set = np.zeros((set_length, set_length))

            for i in xrange(set_length):
                li_set[:, i] = Li[ind_set[i], :]
                Lengths_set[i, i] = Lengths[ind_set[i], ind_set[i]]

            # Form symmetric matrix out of lengths and lines
            L = li_set.T
            Tau = Lengths_set
            ATA = np.dot(np.dot(L.T, Tau.T), np.dot(Tau, L))

            # Compute SVD, use the smallest left singular vector as VP
            U, S, V_T = np.linalg.svd(ATA) 
            vp[:, 0] = U[:, 2]
            vp = vp / np.linalg.norm(vp)

            return vp

    # Compute and return a vanishing point using a random sample of 
    # the minimum number of lines (2)
    # Input: Li (N x 3 Matrix of normalized lines), Lengths (N x N Matrix of line lengths)
    # Returns: vanishing point (3 x 1 matrix)
    def GetMinimalSampleSet(self, Li, Lengths):
        N = Li.shape[0]
        r = np.arange(N)
        np.random.shuffle(r)
        random2 = r[0:self.__minimal_sample_set_dimension]
        return self.estimateLS(Li, Lengths, random2, 2)

    # Compute error of each calibrated lines for a given vanishing point,
    # track if less than threshold adding to consensus set
    # Input: vpNum (int, which vanishing point we're looking for), Li (N x 3 matrix of norm lines)
    #        vp (3 x 1 vanishing point), Error (N x 1 list for tracking error for each line)
    # Returns: Total error, number of lines in consensus set
    def GetConsensusSet(self, vpNum, Li, vp, Error):
        # Reset consensus set indices
        for i in xrange(len(self.__CS_idx)):
            self.__CS_idx[i] = -1

        vn = vp
        vn_norm = np.linalg.norm(vn)

        J = 0.         # Error
        CS_counter = 0 # Size of consensus set for this VP

        for i in xrange(Li.shape[0]):
            # Compute dot product of VP and line
            li = Li[i,:]
            li_norm = np.linalg.norm(li)
            di = np.dot(vn, li)
            di /= (vn_norm*li_norm)
            Error[i] = di*di

            if (Error[i] <= self.__T_noise_squared):
                self.__CS_idx[i] = vpNum # mark in consensus set
                CS_counter += 1
                J += Error[i]
            else:
                J += self.__T_noise_squared

        J /= CS_counter
        return J, CS_counter

    # Given raw linesegments, and number of vanishing points
    # compute most likely vanishing points using LS estimation
    # Input: lineSegmentsRaw (N list of lineSegments as [p1.x, p1.y, p2.x, p2.y])
    # Returns: list of vps, list of lineSegments, number of inliers (used for VP estimation)
    def multipleVPEstimation(self, lineSegmentsRaw, numVps=1):
        vps = []
        lineSegmentsClusters = []
        numInliers = []
        lineSegments = lineSegmentsRaw[:]
        # Loop over max number of vanishing points
        for vpNum in xrange(numVps):
            self.fillDataContainers(lineSegments)
            numLines = len(lineSegments)

            if self.__verbose:
                print "VP {}".format(vpNum)

            if numLines < self.__minimal_sample_set_dimension:
                print "Error: {} is not enough line segments to compute vanishing point"
                break

            # list of line indices in consensus set for current vp
            ind_CS = []
            # number of indices in best vp so far
            N_I_best = self.__minimal_sample_set_dimension
            # Error of best vp so far
            J_best = float('Inf')

            iteration = 0
            T_iter = sys.maxint # adjustable total number of iterations

            # number of consecutive iterations with no updates
            no_updates = 0 
            max_no_updates = 100

            curr_vp = np.zeros((3, 1))

            self.__CS_best = [0]*numLines
            self.__CS_idx = [0]*numLines
            Error = [0.]*numLines

            if self.__verbose:
                print "Method: Calibrated Least Squares"
                print "Starting MSAC"

            while ((iteration <= self.__min_iters) or ((iteration <= T_iter) and (iteration <= self.__max_iters) and (no_updates <= max_no_updates))):
                iteration += 1
                no_updates = 0

                if iteration >= self.__max_iters:
                    break

                # Hypothesis
                # Select Minimal sample set to get a VP
                if (self.__Li.shape[0] < self.__minimal_sample_set_dimension):
                    break
                vpAux = self.GetMinimalSampleSet(self.__Li, self.__Lengths)

                # Test cost of this VP, store cost (J) and number of inlier (size of consensus set)
                J, N_I = self.GetConsensusSet(vpNum, self.__Li, vpAux, Error)

                # Update 
                # If new cost is better than the best one, update 
                notify = False
                if (N_I >= self.__minimal_sample_set_dimension and (J < J_best)) or (J == J_best and N_I > N_I_best):
                    notify = True


                    J_best = J
                    self.__CS_best[:] = self.__CS_idx[:]
                    curr_vp = vpAux # store into __vp (current best hypothesis), vp is calibrated

                    # If we found more inliers, update total number of iterations
                    update_T_iter = False
                    if (N_I > N_I_best):
                        update_T_iter = True

                    N_I_best = N_I

                    if (update_T_iter):
                        # Update number of iterations
                        q = 0.
                        if self.__minimal_sample_set_dimension > N_I_best:
                            print "Error, number of inliers must be higher than minimal sample set"

                        if numLines == N_I_best:
                            q = 1.
                        else:
                            q = 1.
                            for j in xrange(self.__minimal_sample_set_dimension):
                                q *= (N_I_best - j)/float(numLines - j)

                        if ((1 - q) > 1e-12):
                            T_iter = np.ceil(np.log(self.__epsilon)/ np.log(1 - q))
                        else:
                            T_iter = 0
                else:
                    no_updates += 1


                if self.__verbose and notify:
                    aux = max(T_iter, self.__min_iters)
                    print "Iteration = {}/{}".format(iteration, aux)
                    print "Inliers = {}/{} (cost is J = {})".format(N_I_best, numLines, J_best)

                    print "MSS Cal.VP = {}, {}, {}".format(curr_vp[0], curr_vp[1], curr_vp[2])

                # Check CS length (for case all line segments are in the same CS)
                if N_I_best == numLines:
                    if self.__verbose:
                        print "All line segments are inliers. End MSAC at iteration {}".format(iteration)
                        break

            # Reestimate if necessary
            if (self.__verbose): 
                print "Number of interations: {}".format(iteration)
                print "Final number of inliers = {}/{}".format(N_I_best, numLines)

            lineSegmentsCurrent = []
            for i in xrange(numLines):
                if (self.__CS_best[i] == vpNum):
                    ind_CS.append(i)
                    lineSegmentsCurrent.append(lineSegments[i])

            if J_best > 0 and len(ind_CS) > self.__minimal_sample_set_dimension:

                if self.__verbose:
                    print "Reestimating the solution..."

                curr_vp = self.estimateLS(self.__Li, self.__Lengths, ind_CS, int(N_I_best))

                if self.__verbose:
                    print "Done!"
                    print "Cal.VP = {},{},{}".format(curr_vp[0], curr_vp[1], curr_vp[2])

                # Project VP back into homogeneous coordinates
                curr_vp = np.dot(self.__K, curr_vp)
                if curr_vp[2] != 0:
                    curr_vp /= curr_vp[2]
                else:
                    curr_vp = np.dot(self.__K, curr_vp)

                if self.__verbose:
                    print "VP = {},{},{}".format(curr_vp[0], curr_vp[1], curr_vp[2])

                vps.append(curr_vp)
            else:
                print 'Error: VP is at infinity, J_best == 0, no intersection of lines (!)'
                return None, None, None

            if N_I_best > 2:
                while len(ind_CS) > 0:
                    if ind_CS[-1] < len(lineSegments):
                        del lineSegments[ind_CS[-1]]
                    ind_CS.pop()

            lineSegmentsClusters.append(lineSegmentsCurrent)
            numInliers.append(N_I_best)
            
            lineSegments = np.copy(lineSegmentsRaw)

        return vps, lineSegmentsClusters, numInliers
