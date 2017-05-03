import dlib
import numpy as np
import collections

from trafficbehavior.vehicledetection import ObjectTracker
from trafficbehavior.common import image_util

# PossibleObjectTracker tracks potential objects over a fixed number of frames.
# If an object that appeared in certain area for a certain number of frames,
# we can begin tracking it using the ObjectTracker (which is more expensive)
# This is a simple method of pruning so we don't begin tracking every false
# positive.
class PossibleObjectTracker:
    def __init__(self, number_to_track=3):
        # List of deque of boxes for each last time a possible object 
        # was detected. Lists have None if no object was detected there frame
        self.possible_object_list = []
        # How many frames to track potential objects
        self.max_size = number_to_track
        # How many times an object has to be tracked before we return it as
        # an object to track. If an object is found decided_count times out of
        # the last max_size frames, it is returned.
        self.decided_count = 2
        # amount of area overlapping to consider it to be the same object as
        # found in the previous frame
        self.new_tracker_threshold = 0.8

    # potential_new_boxes should be an iterable
    def update(self, potential_new_boxes):
        new_boxes = [] # list of boxes that do not have a match
        # List of booleans of whether a possible tracked object was updated
        updated_object_list = [False]*len(self.possible_object_list)

        for box in potential_new_boxes:
            found_match = False
            # For each possible tracked object, compute average overlap
            # with the previous items in the list. If it is above a certain
            # threshold, add new box to the list, and mark that possible
            # object as updated
            for ii in range(len(self.possible_object_list)):
                prev_boxes = self.possible_object_list[ii]
                average_overlap = 0
                number_not_none = 0.
                for prev_box in prev_boxes:
                    if prev_box is None:
                        continue
                    overlap_box = image_util.find_overlap(box, prev_box)
                    if overlap_box is None:
                        continue
                    overlap_area = image_util.compute_area(overlap_box)
                    area = image_util.compute_area(box)
                    prev_area = image_util.compute_area(prev_box)
                    percent_overlap = float(overlap_area)/min(prev_area, area)
                    average_overlap += percent_overlap
                    number_not_none += 1
                if number_not_none != 0:
                    average_overlap /= number_not_none
                    if average_overlap > self.new_tracker_threshold:
                        self.possible_object_list[ii].append(box)
                        if len(self.possible_object_list[ii]) > self.max_size:
                            self.possible_object_list[ii].popleft()
                        updated_object_list[ii] = True
                        found_match = True
                        break
            # If did not find a match, mark it as a new box
            if not found_match:
                new_boxes.append(box)

        # Append None for possible objects that weren't updated
        for ii in range(len(updated_object_list)):
            if not updated_object_list[ii]:
                self.possible_object_list[ii].append(None)
                if len(self.possible_object_list[ii]) > self.max_size:
                        self.possible_object_list[ii].popleft()

        # Add new boxes to the list by creating a deque for each one
        for box in new_boxes:
            new_deque = collections.deque()
            new_deque.append(box)
            self.possible_object_list.append(new_deque)

        # Check each possible object. If it has at least self.decided_count
        # not None objects, use the most recent box as an object and remove it 
        # from the possible_object_list. 
        # Else if it is the max_size and all None's, stop tracking it and remove
        # it from the possible_object_list.
        return_boxes = []
        i = 0
        while i < len(self.possible_object_list):
            boxes = self.possible_object_list[i]
            number_not_none = 0
            for box in boxes:
                if box is not None:
                    number_not_none += 1
            if number_not_none > self.decided_count:
                for box in reversed(boxes):
                    if box is not None:
                        return_boxes.append(box)
                        break
                del self.possible_object_list[i]
                i -= 1
            elif number_not_none == 0 and len(boxes) == self.max_size:
                del self.possible_object_list[i]
                i -= 1
            i += 1
        return return_boxes

# VehicleDetector detects and tracks the rears of vehicles when given a sequence
# of images that correspond to video.
class VehicleDetector:
    # Constructor a Vehicle Detector using the object detector as
    # detector_filepath
    def __init__(self, detector_filepath, confidence_threshold=7.5, 
                    backwards_confidence_threshold=7.5):
        # Our object detector
        self.dlib_detector = dlib.simple_object_detector(detector_filepath)
        # List of tracked objects, which are dlib_ObjectTracker objects
        self.tracked_objects = []
        # Threshold for confidence. If an object tracker falls below this
        # confidence, do not consider it a new object 
        self.confidence_threshold = confidence_threshold
        # If a new object overlaps with at least new_area_threshold,
        # we DO NOT consider it a new object
        self.new_area_threshold = 0.5 
        # If two existing objects overlap with at least 
        # exist_area_threshold, we do NOT consider them different and
        # will merge them into a single object
        self.existing_area_threshold = 0.7
        # For adding new objects
        # Helper object for tracking possible new objects (need to see it in 
        # multiple consecutive frames before starting to fully track it)
        self.possible_object_tracker = PossibleObjectTracker()
        # List of bounding boxes on new objects that we have begun to track
        # This is emptied at the start of each call to update, but can be called
        # after update to get the locations of new objects, if any
        self.new_bounding_boxes = []

        self.backwards_tracked_objects = []
        self.backwards_confidence_threshold = backwards_confidence_threshold

    # Pass the next frame (img) and return a list of dictionaries of the 
    # box and id of each detected vehicle
    def update(self, img):
        # Empty the list of bounding boxes
        self.new_bounding_boxes = []
        bounding_boxes = []

        # Get existing tracked objects. If any existing tracked objects 
        # are below the confidence threshold, stop tracking them 
        # This works due to the shallow copy in [:]
        for tracked_object in self.tracked_objects[:]:
            confidence = tracked_object.update(img)
            if confidence < self.confidence_threshold:
                self.tracked_objects.remove(tracked_object)

        # For the existing tracked objects, check if they overlap with
        # another object within a certain threshold. If so, remove the 
        # smaller object of the two (makes up smaller percent of area). 
        for tracked_object in self.tracked_objects[:]:
            box1 = tracked_object.get_position()
            remove_object = False
            for other_tracked_object in self.tracked_objects[:]:
                if tracked_object == other_tracked_object:
                    continue
                box2 = other_tracked_object.get_position()
                overlap_box = image_util.find_overlap(box1, box2)
                if overlap_box is None:
                    continue
                overlap_area = image_util.compute_area(overlap_box)
                orig_area = image_util.compute_area(box1)
                percent_overlap = float(overlap_area)/orig_area
                if percent_overlap > self.existing_area_threshold:
                    remove_object = True
            if remove_object:
                self.tracked_objects.remove(tracked_object)
            else:
                bounding_boxes.append(tracked_object.get_dict())

        # For all new detected objects, check all existing tracked objects
        # If the overlap area above a certain threshold, count it as the
        # same object. 
        potential_new_boxes = []
        found_list = self.dlib_detector(img)
        for dlib_rect in found_list:
            box = self.convert_dlib_rect(dlib_rect)
            max_overlap = 0
            for box_dict in bounding_boxes[:]:
                other_box = box_dict['box']
                overlap_box = image_util.find_overlap(box, other_box)
                if overlap_box is None:
                    continue
                overlap_area = image_util.compute_area(overlap_box)
                orig_area = image_util.compute_area(box)
                other_area = image_util.compute_area(other_box)
                percent_overlap = float(overlap_area)/min(orig_area, other_area)
                if max_overlap < percent_overlap:
                    max_overlap = percent_overlap
            if max_overlap > self.new_area_threshold:
                continue
            else:
                potential_new_boxes.append(box)

        # Use the possible object tracker on the potential_new_boxes. It will 
        # return a list of new objects to initialize
        new_objects = self.possible_object_tracker.update(potential_new_boxes)
        for box in new_objects:
            new_object_tracker = ObjectTracker.dlib_ObjectTracker(img, box)
            self.tracked_objects.append(new_object_tracker)
            self.new_bounding_boxes.append(new_object_tracker.get_dict())
            bounding_boxes.append(new_object_tracker.get_dict())

        return bounding_boxes

    # Return a list of bounding boxes of new objects that were detected in the
    # last call to update
    def get_new_boxes(self):
        return self.new_bounding_boxes

    # Delete old backwards trackers, and create an object tracker for each 
    # new bounding box with the given img
    def initialize_backwards(self, img):
        self.backwards_tracked_objects = []
        for box_dict in self.new_bounding_boxes:
            new_box = box_dict['box']
            new_box_id = box_dict['id']
            new_object_tracker = ObjectTracker.dlib_ObjectTracker(img, new_box,
                                                    new_box_id, store_size=1)
            self.backwards_tracked_objects.append(new_object_tracker)

    # Update all backwards bounding boxes with the given img
    # Remove trackers that are below the confidence threshold.
    def update_backwards(self, img):
        if len(self.backwards_tracked_objects) == 0:
            return None
        bounding_boxes = []
        for tracked_object in self.backwards_tracked_objects[:]:
            confidence = tracked_object.update(img)
            if confidence < self.confidence_threshold:
                self.backwards_tracked_objects.remove(tracked_object)
            else:
                bounding_boxes.append(tracked_object.get_dict())
        return bounding_boxes

    # Convert a dlib rectangle object to a numpy array of [x1, y1, x2, y2]
    def convert_dlib_rect(self, dlib_rect):
        box = np.array([dlib_rect.left(), dlib_rect.top(), 
                        dlib_rect.right(), dlib_rect.bottom()])
        return box
