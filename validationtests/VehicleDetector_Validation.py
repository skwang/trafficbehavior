import os, sys, argparse, cv2, numpy as np, dlib
code_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
parent_dir = os.path.dirname(code_dir)
sys.path.append(parent_dir)

# module imports
from trafficbehavior.common import io_util
from trafficbehavior.common import image_util
from trafficbehavior.common import display
from trafficbehavior.vehicledetection import main

# same directory imports
import convert_boxes_csv

def get_validation_dirs(package_dir):
    validation_dir = os.path.join(package_dir, 'validation')
    validation_dirs = []
    for item in os.listdir(validation_dir):
        if item[0] == 'v':
            validation_dirs.append(item)
    return validation_dirs

def boxes_close_enough(box1, box2):
    overlap_box = image_util.find_overlap(box1, box2)
    if overlap_box is None:
        return False
    overlap_perc = float(image_util.compute_area(overlap_box))/min(
                            image_util.compute_area(box1), 
                            image_util.compute_area(box2))
    if (overlap_perc > 0.5):
        [x1, y1, x2, y2] = box1
        [xx1, yy1, xx2, yy2] = box2
        com_diff = ((x2 - x1) - (xx2 - xx1))**2 + ((y2 - y1) - (yy2 - yy1))**2
        if com_diff**0.5 <= 1*min([x2-x1, y2-y1, xx2 - xx1, yy2 - yy1]):
            return True
    return False

def same_boxes(box1, box2):
    overlap_box = image_util.find_overlap(box1, box2)
    if overlap_box is None:
        return False
    overlap_perc = float(image_util.compute_area(overlap_box))/min(
                            image_util.compute_area(box1), 
                            image_util.compute_area(box2))
    if (overlap_perc > 0.7):
        [x1, y1, x2, y2] = box1
        [xx1, yy1, xx2, yy2] = box2
        com_diff = ((x2 - x1) - (xx2 - xx1))**2 + ((y2 - y1) - (yy2 - yy1))**2
        if com_diff**0.5 <= 1*min([x2-x1, y2-y1, xx2 - xx1, yy2 - yy1]):
            return True
    return False

def convert_dlib_rect(dlib_rect):
    box = [dlib_rect.left(), dlib_rect.top(), 
            dlib_rect.right(), dlib_rect.bottom()]
    return box

def svm_detection_validation(full_path, detector_filepath, min_area=None, 
                                load_boxes=False, stepsize=15):
    print 'Starting svm validation test for {}'.format(full_path)
    filenames = io_util.load_fnames(full_path)
    hand_dict = io_util.load_vehicle_boxes(full_path, use_hand_labeled=True)
    true_positives = 0.
    false_positives = 0.
    false_negatives = 0.
    detector_fn = detector_filepath.split('/')[-1].split('.svm')[0]
    if min_area:
        test_dict_fn = 'boxes_{}.json'.format(detector_fn)
    else:
        test_dict_fn = 'boxes_nominarea_{}.json'.format(detector_fn)
    test_dict = {}
    if load_boxes:
        test_dict = io_util.load_json(full_path, test_dict_fn)
    else:
        dlib_detector = dlib.simple_object_detector(detector_filepath)
    
    for fname in filenames[::stepsize]:
        if fname not in hand_dict:
            continue
        hand_boxes = hand_dict[fname]
        test_boxes = []
        if load_boxes:
            test_boxes = test_dict[fname]
        else:
            img = cv2.imread(os.path.join(full_path, fname))
            found_list = dlib_detector(img)
            for dlib_rect in found_list:
                test_boxes.append(convert_dlib_rect(dlib_rect))
            test_dict[fname] = np.array(test_boxes)

        if min_area:
            for test_box in test_boxes[:]:
                if image_util.compute_area(test_box) < min_area:
                    test_boxes.remove(test_box)
            for hand_box in hand_boxes[:]:
                if image_util.compute_area(hand_box) < min_area:
                    hand_boxes.remove(hand_box)

        for test_box in test_boxes[:]:
            for hand_box in hand_boxes[:]:
                if boxes_close_enough(test_box, hand_box):
                    hand_boxes.remove(hand_box)
                    test_boxes.remove(test_box)
                    true_positives += 1
                    break

        false_positives += len(test_boxes)
        false_negatives += len(hand_boxes)

    if not load_boxes:
        io_util.save_json(full_path, test_dict_fn, test_dict)

    precision = 0
    recall = 0
    if true_positives != 0:
        precision = true_positives/(true_positives + false_positives)
        recall = true_positives/(true_positives + false_negatives)
    print '\tPrecision: {} ({}/{})'.format(precision, true_positives, 
                                            (true_positives + false_positives))
    print '\tRecall: {} ({}/{})'.format(recall, true_positives,
                                            (true_positives + false_negatives))
    return true_positives, false_positives, false_negatives


def remove_overlapping(test_boxes, full_path, fname):
    boxdicts = test_boxes[:]
    for test_boxdict in boxdicts[:]:
        for other_boxdict in boxdicts[:]:
            if test_boxdict == other_boxdict:
                continue
            if (same_boxes(test_boxdict['box'], other_boxdict['box'])):
                if (test_boxdict['id'] <= other_boxdict['id'] and 
                        other_boxdict in boxdicts):
                    boxdicts.remove(other_boxdict)
                elif (test_boxdict['id'] >= other_boxdict['id'] and 
                        test_boxdict in boxdicts):
                    boxdicts.remove(test_boxdict)
    return boxdicts


def vehicle_detection_validation(full_path, detector_filepath, min_area=None, stepsize=15):
    print 'Starting vehicle detection validation test for {} using {}'.format(full_path, detector_filepath)
    filenames = io_util.load_fnames(full_path)
    detector_fn = detector_filepath.split('/')[-1].split('.svm')[0]
    test_dict_fn = 'boxes_output_{}_8.5.json'.format(detector_fn)
    #test_dict_fn = 'boxes_output_{}_8.5_run2.json'.format(detector_fn)
    test_dict = io_util.load_json(full_path, test_dict_fn)
    #test_dict = io_util.load_vehicle_boxes(full_path, use_hand_labeled=False)
    hand_dict = io_util.load_vehicle_boxes(full_path, use_hand_labeled=True)

    true_positives = 0.
    false_positives = 0.
    false_negatives = 0.
    for fname in filenames[::stepsize]:
        hand_boxes = []
        if fname in hand_dict:
            hand_boxes = hand_dict[fname][:]
        test_boxes = []
        if fname in test_dict:
            test_boxes = test_dict[fname]

        if min_area:
            for test_boxdict in test_boxes[:]:
                test_box = test_boxdict['box']
                if image_util.compute_area(test_box) < min_area:
                    test_boxes.remove(test_boxdict)
            for hand_box in hand_boxes[:]:
                if image_util.compute_area(hand_box) < min_area:
                    hand_boxes.remove(hand_box)

        img1 = cv2.imread(os.path.join(full_path, fname))
        display.draw_boxes(img1, hand_boxes)
        display.draw_boxdicts(img1, test_boxes, (0,255,0))
        cv2.imshow('original', img1)
        

        test_boxes = remove_overlapping(test_boxes, full_path, fname)

        for test_boxdict in test_boxes[:]:
            test_box = test_boxdict['box']
            for hand_box in hand_boxes[:]:
                if boxes_close_enough(test_box, hand_box):
                    hand_boxes.remove(hand_box)
                    test_boxes.remove(test_boxdict)
                    true_positives += 1
                    break

        false_positives += len(test_boxes)
        false_negatives += len(hand_boxes)

        if len(hand_boxes) > 0:
            # hand_boxes = []
            # if fname in hand_dict:
            #     hand_boxes = hand_dict[fname]
            # test_boxes = []
            # if fname in test_dict:
            #     test_boxes = test_dict[fname]
            img1 = cv2.imread(os.path.join(full_path, fname))
            # display.draw_boxes(img1, hand_boxes)
            # display.draw_boxdicts(img1, test_boxes, (0,0,255))
            # cv2.imshow('FN - truth is blue, red is code', img1)
            # while True:
            #     if cv2.waitKey(1) & 0xFF == ord('s'):
            #         cv2.imwrite(os.path.join(full_path, 'fn_{}'.format(fname)), img1)
            #         break
            #     elif cv2.waitKey(1) & 0xFF == ord(' '):
            #         break

        if len(test_boxes) > 0:
            # hand_boxes = []
            # if fname in hand_dict:
            #     hand_boxes = hand_dict[fname]
            # test_boxes = []
            # if fname in test_dict:
            #     test_boxes = test_dict[fname]
            img1 = cv2.imread(os.path.join(full_path, fname))
            display.draw_boxes(img1, hand_boxes)
            display.draw_boxdicts(img1, test_boxes, (0,0,255))
            # for box in test_boxes:
            #     [x1, y1, x2, y2] = box['box']
            #     width = x2 - x1
            #     height = y2 - y1
            #     print '{},{},{},{},{},{},{}'.format(x1 + width/2., y1 + height/2.,  x1, y1, width, height,fname)
            cv2.imshow('FP - truth is blue, red is code', img1)
            cv2.waitKey(0)
            # while True:
            #     if cv2.waitKey(1) & 0xFF == ord('s'):
            #         cv2.imwrite(os.path.join(full_path, 'fp_{}'.format(fname)), img1)
            #         break
            #     elif cv2.waitKey(1) & 0xFF == ord(' '):
            #         break
    
    precision = 0
    recall = 0
    f1 = 0
    if true_positives != 0:
        precision = true_positives/(true_positives + false_positives)
        recall = true_positives/(true_positives + false_negatives)
        f1 = 2*(precision * recall)/(precision + recall)
    print '\tPrecision: {} ({}/{})'.format(precision, true_positives, 
                                            (true_positives + false_positives))
    print '\tRecall: {} ({}/{})'.format(recall, true_positives,
                                            (true_positives + false_negatives))
    print '\tF1 Score: {}'.format(f1)
    return true_positives, false_positives, false_negatives

if __name__ == '__main__':
    package_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    svm_dir = './vehicledetection/svm/'
    default_svm = os.path.join(svm_dir, 'detector_session2_raw.svm')

    # ap = argparse.ArgumentParser()
    # ap.add_argument('-v', '--videopath', help='Relative Path to Video', 
    #                 required=True)
    # args = ap.parse_args()
    # relativepath = args.videopath
    # min_area = None # min number of pixels in a bounding box for us to count it
    # vehicle_detection_validation(relativepath, default_svm, min_area)

    run_main = False
    run_csv_conversion = True
    run_validation = True
    run_svm_validation = False



    if run_main:
        svm_fps = []
        svm_fps.append(os.path.join(svm_dir, 'detector_session2_raw.svm'))
        # for c in [10]:
        #     for eps in [1e-4]:
        #         detector_name = "detector_session3_c{}_e{}.svm".format(c, eps)
        #         svm_fp = os.path.join(svm_dir, detector_name)
        #         svm_fps.append(svm_fp)
        for svm_fp in svm_fps:
            for valid_dir in get_validation_dirs(package_dir):
                relativepath = os.path.join(package_dir, 'validation', valid_dir)
                main.vehicle_detector_main(relativepath, 
                        display_frames=False, 
                        dlib_detector_fname=svm_fp, 
                        verbose=True, 
                        forward_confidence_thresh=8.5, 
                        backward_confidence_thresh=8.5, 
                        run_backwards_tracking=True)

    if run_csv_conversion:
        correct_aspect_ratio = False
        for valid_dir in get_validation_dirs(package_dir):
            relativepath = os.path.join(package_dir, 'validation', valid_dir)
            csvfilename = 'manual_boxes.csv'
            csvpath = os.path.join(relativepath, csvfilename)
            convert_boxes_csv.convert_manual_boxes(csvpath, relativepath, None, 
                                                    correct_aspect_ratio)

    if run_validation:
        min_area = None
        svm_fps = []
        svm_fps.append(os.path.join(svm_dir, 'detector_session2_raw.svm'))
        # for c in [10]:
        #     for eps in [1e-2]:
        #         detector_name = "detector_session3_c{}_e{}.svm".format(c, eps)
        #         svm_fp = os.path.join(svm_dir, detector_name)
        #         svm_fps.append(svm_fp)
        for svm_fp in svm_fps:
            true_positives = 0.
            false_positives = 0.
            false_negatives = 0.
            for valid_dir in get_validation_dirs(package_dir):
                relativepath = os.path.join(package_dir, 'validation', valid_dir)
                tp, fp, fn = vehicle_detection_validation(relativepath, svm_fp, min_area)
                true_positives += tp
                false_positives += fp
                false_negatives += fn

            precision = true_positives/(true_positives + false_positives)
            recall = true_positives/(true_positives + false_negatives)
            print 'Total validation results using {}:'.format(svm_fp)
            print 'Average precision: {} ({}/{})'.format(precision, true_positives, 
                                                (true_positives + false_positives))
            print 'Average recall: {} ({}/{})'.format(recall, true_positives,
                                                (true_positives + false_negatives))
            print 'F1 Score: {}'.format(2*(precision * recall)/(precision + recall))

    if run_svm_validation:
        min_area = None
        # if true, looks for the json file. if false, runs the detection and saves it 
        load_boxes = True

        #svm_fps = []
        svm_fps = [os.path.join(svm_dir, 'detector_session2_raw.svm')]
        # for c in [1, 5, 10]:
        #     for eps in [1e-2, 1e-3, 1e-4]:
        #         detector_name = "detector_session3_c{}_e{}.svm".format(c, eps)
        #         svm_fp = os.path.join(svm_dir, detector_name)
        #         svm_fps.append(svm_fp)

        for svm_fp in svm_fps:
            print 'SVM Detection with {}'.format(svm_fp)
            true_positives = 0.
            false_positives = 0.
            false_negatives = 0.
            for valid_dir in get_validation_dirs(package_dir):
                relativepath = os.path.join(package_dir, 'validation', valid_dir)
                tp, fp, fn = svm_detection_validation(relativepath, svm_fp, min_area, load_boxes)
                true_positives += tp
                false_positives += fp
                false_negatives += fn
                

            precision = true_positives/(true_positives + false_positives)
            recall = true_positives/(true_positives + false_negatives)
            print 'Total results using {}'.format(svm_fp)
            print 'Average precision: {} ({}/{})'.format(precision, true_positives, 
                                                (true_positives + false_positives))
            print 'Average recall: {} ({}/{})'.format(recall, true_positives,
                                                (true_positives + false_negatives))  
            print 'F1 Score: {}'.format(2*(precision * recall)/(precision + recall))

        

        



