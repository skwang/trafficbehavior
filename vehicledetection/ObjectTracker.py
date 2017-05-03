import collections, dlib
import numpy as np

NEXT_DLIB_OBJECT_TRACKER_ID = 0

# Tracks a single object using the dlib implementation of the scale estimation
# correlation tracker.
class dlib_ObjectTracker:
    # first_image is a numpy array
    # bounding_box is [left, top, right, bot] giving initial object coordinates
    # store_size is the number of confidences to average over when returning
    # the dlib confidence value
    # tracker_id is the id for the object_tracker. If it is None (default),
    # then it is automatically initialized to the next available id
    def __init__(self, first_image, bounding_box, tracker_id=None, 
                    store_size=25):
        global NEXT_DLIB_OBJECT_TRACKER_ID
        self.tracker = dlib.correlation_tracker()
        # Using: dlib.rectangle(*[left, top, right, bottom])
        dlib_rect = dlib.rectangle(*bounding_box[:])
        self.num_confidences = store_size
        self.previous_confidences = collections.deque()
        self.tracker.start_track(first_image, dlib_rect)
        if tracker_id is None:
            self.tracker_id = NEXT_DLIB_OBJECT_TRACKER_ID
            NEXT_DLIB_OBJECT_TRACKER_ID += 1
        else:
            self.tracker_id = tracker_id

    # Update the dlib correlation tracker with the new_image, and return the
    # averaged confidence over the last store_size frames
    def update(self, new_image):
        confidence = self.tracker.update(new_image)
        self.previous_confidences.append(confidence)
        if len(self.previous_confidences) > self.num_confidences:
            self.previous_confidences.popleft()
        average = sum(self.previous_confidences)/len(self.previous_confidences)
        return average

    # Return the current estimation for a bounding box of the object
    def get_position(self):
        dlib_rect = self.tracker.get_position()
        bounding_box = np.array([dlib_rect.left(), dlib_rect.top(), 
                                dlib_rect.right(), dlib_rect.bottom()])
        return bounding_box

    # Return the id of the object tracker
    def get_id(self):
        return self.tracker_id

    # Return the dictionary representation of the object tracker
    def get_dict(self):
        return {'box': self.get_position().astype(np.int64), 
                'id': self.get_id()}
