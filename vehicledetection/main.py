import numpy as np, cv2, argparse, os, sys, json

code_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
parent_dir = os.path.dirname(code_dir)
sys.path.append(parent_dir)

from trafficbehavior.vehicledetection import ObjectTracker
from trafficbehavior.vehicledetection import VehicleDetector
from trafficbehavior.common import io_util
from trafficbehavior.common import display

def vehicle_detector_main(relativepath, display_frames, dlib_detector_fname, 
            verbose, forward_confidence_thresh, backward_confidence_thresh, 
            run_backwards_tracking=True):
    if verbose:
        print 'Starting vehicle detector main for {}'.format(relativepath)

    # # Read the filenames from the list.txt file
    filenames = io_util.load_fnames(relativepath)

    # Get the shape from the first image
    tmp_img = cv2.imread(os.path.join(relativepath, filenames[0]), 
                            cv2.IMREAD_GRAYSCALE)
    imgshape = tmp_img.shape
    imgheight, imgwidth = imgshape

    # Confidence value where if we fall below this, we stop backwards tracking
    confidence_threshold = forward_confidence_thresh
    previous_track_confidence = backward_confidence_thresh

    # Set up the vehicle detector object
    vehicleDetector = VehicleDetector.VehicleDetector(dlib_detector_fname, 
                                                        confidence_threshold,
                                                        previous_track_confidence)

    # Output dictionary where key = filename,
    # value = list of boxes + ids of objects in that image
    fname_boxes_dict = {}

    for fi in range(len(filenames)):
        if verbose:
            display.show_progressbar(fi, filenames)
        fname = filenames[fi]
        img = cv2.imread(os.path.join(relativepath, fname),cv2.IMREAD_GRAYSCALE)
        # Add any detected boxes to the dictionary
        vehicle_boxes = vehicleDetector.update(img)
        if fname not in fname_boxes_dict:
            fname_boxes_dict[fname] = []
        for box_dict in vehicle_boxes:
            fname_boxes_dict[fname].append(box_dict)

        # For new detected boxes, initialize an object tracker to go backwards
        # and detect boxes in previous frames, until the confidence falls below
        # a certain threshold
        new_box_dicts = vehicleDetector.get_new_boxes()
        if run_backwards_tracking:
            vehicleDetector.initialize_backwards(img)
            for fj in range(fi - 1, -1, -1):
                prev_fname = filenames[fj]
                prev_img = cv2.imread(os.path.join(relativepath,prev_fname), 
                                        cv2.IMREAD_GRAYSCALE)
                boxes = vehicleDetector.update_backwards(prev_img)
                if boxes is None:
                    break
                else:
                    if prev_fname in fname_boxes_dict:
                        fname_boxes_dict[prev_fname] += boxes
                    else:
                        fname_boxes_dict[prev_fname] = boxes
            # for box_dict in new_box_dicts:
            #     #if verbose:
            #     #    print '\n\t Starting backwards track'
            #     new_box = box_dict['box']
            #     new_box_id = box_dict['id']
            #     new_object_tracker = ObjectTracker.dlib_ObjectTracker(img, new_box,
            #                             new_box_id, store_size=1)
            #     for fj in range(fi - 1, -1, -1):
            #         prev_fname = filenames[fj]
            #         prev_img = cv2.imread(os.path.join(relativepath,prev_fname), 
            #                                 cv2.IMREAD_GRAYSCALE)
            #         confidence = new_object_tracker.update(prev_img)
            #         if confidence < previous_track_confidence:
            #             break
            #         else:
            #             if prev_fname in fname_boxes_dict:
            #                 fname_boxes_dict[prev_fname].append(
            #                     new_object_tracker.get_dict())
            #             else:
            #                 fname_boxes_dict[prev_fname] = [
            #                     new_object_tracker.get_dict()]
            #     #if verbose:
            #     #    print '\t Ended backwards track at ' + filenames[fj]
    if verbose:
        print ""

    # Create the output json file of the boxes
    detector_fn = dlib_detector_fname.split('/')[-1].split('.svm')[0]
    output_fname = os.path.join(relativepath, 
                                    'boxes_output_{}_{}_run2.json'.format(
                                    detector_fn, confidence_threshold))
    with open(output_fname, 'w') as outfile:
       json.dump(fname_boxes_dict, outfile, indent=4,
                 sort_keys=True, cls=io_util.NumpyEncoder)
    
    # Display the images with the boxes and ids on them
    if display_frames:
        for fname in filenames:
            boxes = fname_boxes_dict[fname]
            img = cv2.imread(os.path.join(relativepath, fname))
            display.draw_boxdicts(img, boxes, (255,0,0))
            cv2.imshow('With box and id labels', img)
            cv2.waitKey(1)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('-v', '--videopath', required=True, 
                    help="Relative Path to Video")
    ap.add_argument('-s', '--svm', help='Relative Path to SVM', 
                    default='./vehicledetection/svm/detector_session2_raw.svm')
    ap.add_argument('-d', '--display', default = True)
    ap.add_argument('-f', '--fc', default = 10)
    ap.add_argument('-b', '--bc', default = 10)
    args = ap.parse_args()
    relativepath = args.videopath
    display_frames = args.display
    dlib_detector_fname = args.svm
    verbose = True
    run_backwards_tracking = True
    forward_confidence_thresh = float(args.fc)
    backwards_confidence_thresh = float(args.bc)
    vehicle_detector_main(relativepath, display_frames, dlib_detector_fname, 
            verbose, forward_confidence_thresh, backwards_confidence_thresh, 
            run_backwards_tracking)

    

