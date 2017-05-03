import numpy as np

class NietoLaneMarker:

    def __init__(self):
        pass

    # a lane marker detector
    # img is input image
    # width is the expected width of the lane
    def detect(self, img, width):
        rows, cols = img.shape
        dest = np.zeros((rows, cols)).astype('int')
        aux = 0
        
        for j in xrange(width, cols - width, 1):
            aux = 2*img[:,j]
            aux -= img[:,j-width]
            aux -= img[:,j+width]
            aux -= np.absolute(img[:,j-width] - img[:,j+width])
            aux[aux < 0] = 0
            aux[aux > 255] = 255
            dest[:,j] = aux
        return dest

    def detectBottom(self, img, width, height=1):
        rows, cols = img.shape
        dest = np.zeros((rows, cols)).astype('int')

        for j in xrange(width, cols - width, 1):
            aux = 2*img[:,j]
            aux -= img[:,j-width]
            aux -= img[:,j+width]
            aux -= np.absolute(img[:,j-width] - img[:,j+width])
            aux[aux < 0] = 0
            aux[aux > 255] = 255
            dest[:,j] = (0.3*aux).astype('int')
        for i in xrange(height, rows-height, 1):
            aux = np.zeros(img[i,:].shape)
            aux += img[i-1,:]
            aux -= img[i+1,:]
            aux += np.absolute(img[i-1,:] - img[i+1,:])
            aux[aux < 0] = 0
            aux[aux > 255] = 255
            dest[i,:] += (0.7*aux).astype('int')
        return dest
