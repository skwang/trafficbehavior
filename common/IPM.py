import cv2, numpy as np, time

class IPM:
    # origSize = (height, width) of input image
    # dstSize  = (height, width) of output image
    def __init__(self, origSize, destSize, origPts, destPts):
        curr_time = time.time()
        self.__origSize = origSize
        self.__destSize = destSize

        self.__origPts = np.zeros((4, 2), dtype = 'float32')
        self.__destPts = np.zeros((4, 2), dtype = 'float32')

        for i in xrange(0, 4):
            self.__origPts[i,:] = origPts[i]
            self.__destPts[i,:] = destPts[i]

        curr_time = time.time()
        # forward homography map 
        self.__map_H = cv2.getPerspectiveTransform(self.__origPts, 
            self.__destPts)
        print "getPersectiveTransform time: {}".format(time.time() - curr_time)
        curr_time = time.time()
        # inverse homography map
        self.__map_H_inv = np.linalg.inv(self.__map_H)
        print "find inv transform time: {}".format(time.time() - curr_time)
        curr_time = time.time()

        # build the mapX, mapY, mapX_inv, mapY_inv maps for using remap
        # for going from orig to IPM
        def getXYmap(H):
            X1 = H[0, 0]*np.ones((destSize[0], 1)) * np.arange(0, destSize[1]).reshape((1, destSize[1]))
            X2 = np.arange(0, destSize[0]).reshape((destSize[0], 1)) * H[0, 1]*np.ones((1, destSize[1]))
            X3 = H[0, 2]*np.ones(destSize)
            X = (X1 + X2 + X3).astype('float32')

            Y1 = H[1, 0]*np.ones((destSize[0], 1)) * np.arange(0, destSize[1]).reshape((1, destSize[1]))
            Y2 = np.arange(0, destSize[0]).reshape((destSize[0], 1)) * H[1, 1]*np.ones((1, destSize[1]))
            Y3 = H[1, 2]*np.ones(destSize)

            Y = (Y1 + Y2 + Y3).astype('float32')

            N1 = H[2, 0]*np.ones((destSize[0], 1)) * np.arange(0, destSize[1]).reshape((1, destSize[1]))
            N2 = np.arange(0, destSize[0]).reshape((destSize[0], 1)) * H[2, 1]*np.ones((1, destSize[1]))
            N3 = H[2, 2]*np.ones(destSize)

            N = (N1 + N2 + N3).astype('float32')

            ret_X = np.divide(X, N)
            ret_Y = np.divide(Y, N)
            return ret_X, ret_Y

        self.__mapX, self.__mapY = getXYmap(self.__map_H_inv)
        self.__mapX_inv, self.__mapY_inv  = getXYmap(self.__map_H)
        print "matrix maps time: {}".format(time.time() - curr_time)

        # self.__mapX = np.zeros((self.__destSize), dtype = 'float32')
        # self.__mapY = np.zeros((self.__destSize), dtype = 'float32')
        # for j in xrange(destSize[0]):
        #     for i in xrange(destSize[1]):
        #         pt = np.dot(self.__map_H_inv, np.array([i, j, 1]).reshape((3, 1)))
        #         if pt[2] != 0:
        #             self.__mapX[j, i] = pt[0]/pt[2]
        #             self.__mapY[j, i] = pt[1]/pt[2]
        #         else:
        #             self.__mapX[j, i] = -1
        #             self.__mapY[j, i] = -1
        # print "build mapX and mapY time: {}".format(time.time() - curr_time)
        # curr_time = time.time()


        
        # curr_time = time.time()
        # diff = np.abs(self.__mapX - X)
        # print np.max(diff)
        # # print np.average(diff[np.where(np.isclose(self.__mapX, BD) != True)])
        # print len(np.where(np.isclose(self.__mapX, X) != True)[0])
        # print len(np.where(np.isclose(self.__mapY, Y) != True)[0])


        # for going from IPM back to orig
        # self.__mapX_inv = np.zeros((self.__destSize), dtype = 'float32')
        # self.__mapY_inv = np.zeros((self.__destSize), dtype = 'float32')
        # for j in xrange(origSize[0]):
        #     for i in xrange(origSize[1]):
        #         pt = np.array([i, j, 1]).reshape((3, 1))
        #         pt = np.dot(self.__map_H, pt)
        #         if pt[2] != 0:
        #             self.__mapX_inv[j, i] = pt[0]/pt[2]
        #             self.__mapY_inv[j, i] = pt[1]/pt[2]
        #         else:
        #             self.__mapX_inv[j, i] = -1
        #             self.__mapY_inv[j, i] = -1
        # print "build inv mapX and mapY time: {}".format(time.time() - curr_time)
        # curr_time = time.time()

    def getOrigSize(self):
        return self.__origSize

    def getDestSize(self):
        return self.__destSize

    # Creates and returns the IPM image of the srcImg
    def imageToIPM(self, srcImg):
        return cv2.remap(srcImg, self.__mapX, self.__mapY, cv2.INTER_LINEAR, 
            borderMode = cv2.BORDER_CONSTANT)

    # Creates and returns the reverse IPM image of the srcImg
    def imageFromIPM(self, srcImg):
        return cv2.remap(srcImg, self.__mapX_inv, self.__mapY_inv, 
            cv2.INTER_LINEAR, borderMode = cv2.BORDER_CONSTANT)

    # Given the (x, y) location of a point, finds its location in the IPM image
    def pointToIPM(self, pt):
        tmp = np.array([pt[0], pt[1], 1]).reshape((3, 1))
        tmp = np.dot(self.__map_H, tmp).flatten()
        if tmp[2] != 0:
            tmp[0] = tmp[0]/tmp[2]
            tmp[1] = tmp[1]/tmp[2]
        else:
            tmp[0] = -1
            tmp[1] = -1
        return tmp[0:2]

    # Given the (x, y) location of a point, finds its location in the reverse 
    # (original) IPM image
    def pointFromIPM(self, pt):
        tmp = np.array([pt[0], pt[1], 1]).reshape((3, 1))
        tmp = np.dot(self.__map_H_inv, tmp).flatten()
        if tmp[2] != 0:
            tmp[0] = tmp[0]/tmp[2]
            tmp[1] = tmp[1]/tmp[2]
        else:
            tmp[0] = -1
            tmp[1] = -1
        return tmp[0:2]
        