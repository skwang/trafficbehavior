import numpy as np
from scipy.signal import butter, lfilter, freqz, medfilt

class KalmanFilter:
    # init_var is initial var
    def __init__(self, init_state, init_var, process_noise):
        # initial state and covariance
        self.state = init_state.copy() #2x1 np array
        self.covar = init_var.copy()   # 2x2 np array
        # process noise
        self.Q = process_noise.copy() # 2x2 np array
        # state[0] is speed, state[1] is acceleration

    # predict our observation of the speed after dt seconds have passed
    def predict(self, dt):
        H = np.array([1, 0]) # observation matrix
        F = np.array([[1, dt], [0, 1]]) # state transition matrix
        self.state = F.dot(self.state)
        self.covar = F.dot(self.covar).dot(F.T) + self.Q

    # update our filter with the given measurement of speed, after dt seconds,
    # and accounting for sensor_noise to the measurement
    def update(self, measurement, dt, sensor_noise=15):
        measurement = np.array([measurement]).reshape((1,1))
        H = np.array([1, 0]).reshape((1, 2))
        R = np.array([sensor_noise]).reshape((1, 1))
        gain = self.covar.dot(H.T).dot(1./(H.dot(self.covar).dot(H.T) + R))
        self.state = self.state + gain.dot(measurement - H.dot(self.state))
        self.covar = self.covar - gain.dot(H).dot(self.covar)
        # Make sure covariance doesn't explode...
        self.covar = np.clip(self.covar, -20, 20)
        return self.state[0]

    def get_estimate(self):
        return self.state[0]

class AccLimitFilter:

    def __init__(self, init_speed, max_accel, avg_width=5):
        mph_to_mps = 0.44704
        self.max_accel = max_accel / mph_to_mps
        self.prev_speeds = []
        self.curr_est = init_speed
        self.avg_width = avg_width

    def predict(self, dt):
        if len(self.prev_speeds) > 0:
            self.curr_est = np.mean(self.prev_speeds[-min(self.avg_width, len(self.prev_speeds)):])

    def update(self, measurement, dt):
        old_speed = self.curr_est
        if len(self.prev_speeds) > 0:
            old_speed = self.prev_speeds[-1]
        if measurement is not None:
            accel = (measurement - old_speed)/dt
            accel = min(self.max_accel, max(-self.max_accel, accel))
            self.curr_est = (accel * dt) + old_speed
        self.prev_speeds.append(self.curr_est)
        if len(self.prev_speeds) > self.avg_width:
            self.prev_speeds = self.prev_speeds[-self.avg_width:]
        return self.curr_est

    def get_estimate(self):
        return self.curr_est

def butter_lpf(data, cutoff, fs, order=5):
    def butter_lowpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a
    
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

def median_filter(data, width=15):
    return medfilt(data, width)