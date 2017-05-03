import numpy as np, cv2, os, sys, json, csv
code_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
parent_dir = os.path.dirname(code_dir)
sys.path.append(parent_dir)
from trafficbehavior.lanedetection import main
from trafficbehavior.common import io_util
from trafficbehavior.common import image_util
from trafficbehavior.common import display

def rint(num):
    return int(round(float(num)))

def get_validation_dirs(package_dir):
    valid_dir = os.path.join(package_dir, 'validation')
    validation_dirs = []
    for item in os.listdir(valid_dir):
        if item[0] == 'v':
            validation_dirs.append(item)
    return validation_dirs

def convert_manual_lanes(csvpath, relativepath):
    output_dict = {}
    with open(csvpath, 'rb') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            fname = row['Filename'] + '.png'

            x1 = rint(row['x1'])
            x2 = rint(row['x2'])
            y1 = rint(row['y1'])
            y2 = rint(row['y2'])

            entry = [[x1, y1], [x2, y2]]
            
            if fname in output_dict:
                output_dict[fname].append(entry)
            else:
                output_dict[fname] = [entry]

    output_fname = os.path.join(relativepath, 'manual_lanes.json')
    with open(output_fname, 'w') as outfile:
       json.dump(output_dict, outfile, indent=4, sort_keys=True)
    print 'Wrote manual lanes json file for {}'.format(csvpath)

def lines_close_enough(line1, line2, height=480.):
    angle_diff = abs(line1.getAngle() - line2.getAngle())/90.
    intercept_diff = abs(line1.getIntercept() - line2.getIntercept())/height
    if line1.checkLineIntersection(line2) and line1.getDotAngle(line2) < 8:
        return True
    if angle_diff < 5/90. and intercept_diff < 0.12:
        return True
    else:
        return False

def get_nearby_lanes(lines, vp, width=640):
    min_bound = vp[0] - width/3.
    max_bound = vp[0] + width/3.
    for line in lines[:]:
        if (line.x1 < min_bound or line.x2 < min_bound or 
            line.x1 > max_bound or line.x2 > max_bound):
            lines.remove(line)
    return lines

def remove_duplicates(lines):
    for line in lines[:]:
        for other_line in lines[:]:
            if line == other_line:
                continue
            if line.checkLineIntersection(other_line):
                lines.remove(other_line)
    return lines 


def lane_detection_validation(relativepath, test_lanes_fn, 
                            true_lanes_fn='manual_lanes.json', stepsize = 15,
                            vp_fn = 'vp.json'):
    print 'Lane Detection Validation for {}'.format(relativepath)
    filenames = io_util.load_fnames(relativepath)
    test_dict = io_util.load_json(relativepath, test_lanes_fn)
    hand_dict = io_util.load_json(relativepath, true_lanes_fn)

    vp_dict = io_util.load_json(relativepath, vp_fn)
    vp = vp_dict['vp']

    true_positives = 0.
    false_positives = 0.
    false_negatives = 0.
    for fname in filenames[::stepsize]:
        hand_lanes = []
        if fname in hand_dict:
            for line in hand_dict[fname]:
                [x1, y1] = line[0]
                [x2, y2] = line[1]
                hand_lanes.append(image_util.Line(x1, y1, x2, y2))
        test_lanes = []
        if fname in test_dict:
            for line in test_dict[fname]:
                [x1, y1] = line[0]
                [x2, y2] = line[1]
                test_lanes.append(image_util.Line(x1, y1, x2, y2))

        test_lanes = remove_duplicates(test_lanes)

        #hand_lanes = get_nearby_lanes(hand_lanes, vp)
        #test_lanes = get_nearby_lanes(test_lanes, vp)

        img = cv2.imread(os.path.join(relativepath, fname))
        display.draw_lines(img, hand_lanes, (255, 0, 0))
        display.draw_lines(img, test_lanes, (0,0,255))
        cv2.imshow('orig', img) 

        for test_line in test_lanes[:]:
            for hand_line in hand_lanes[:]:
                if lines_close_enough(test_line, hand_line):
                    hand_lanes.remove(hand_line)
                    test_lanes.remove(test_line)
                    true_positives += 1
                    break

        false_positives += len(test_lanes)
        false_negatives += len(hand_lanes)

        if len(test_lanes) > 0:
            img = cv2.imread(os.path.join(relativepath, fname))
            # hand_lanes = []
            # if fname in hand_dict:
            #     for line in hand_dict[fname]:
            #         [x1, y1] = line[0]
            #         [x2, y2] = line[1]
            #         hand_lanes.append(image_util.Line(x1, y1, x2, y2))
            # test_lanes = []
            # if fname in test_dict:
            #     for line in test_dict[fname]:
            #         [x1, y1] = line[0]
            #         [x2, y2] = line[1]
            #         test_lanes.append(image_util.Line(x1, y1, x2, y2))
            display.draw_lines(img, hand_lanes, (255, 0, 0))
            display.draw_lines(img, test_lanes, (0,0,255))
            cv2.imshow('FP - truth is blue, red is code', img)
            # angle_diff = abs(hand_lanes[0].getAngle() - test_lanes[0].getAngle())/90.
            # print angle_diff*90.
            # intercept_diff = abs(hand_lanes[0].getIntercept() - test_lanes[0].getIntercept())
            # print intercept_diff
            #print 'FP ',
            for i in range(len(test_lanes)):
               print fname, test_lanes[i].x1, test_lanes[i].y1, test_lanes[i].x2, test_lanes[i].y2
            cv2.waitKey(0)

        if len(hand_lanes) > 0:
            #print 'FN ',
            #for i in range(len(hand_lanes)):
            #    print fname, hand_lanes[i].x1, hand_lanes[i].y1, hand_lanes[i].x2, hand_lanes[i].y2
            img = cv2.imread(os.path.join(relativepath, fname))
            display.draw_lines(img, hand_lanes, (255, 0, 0))
            display.draw_lines(img, test_lanes, (0,0,255))
            cv2.imshow('FN - truth is blue, red is code', img)
            cv2.waitKey(0)

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






if __name__ == "__main__":
    package_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    #relativepath = args.videopath
    display_ = False
    verbose = False
    boxes_fn = 'boxes_output_detector_session2_raw_8.5.json'
    vp_fn = 'vp.json'

    run_main = False
    run_csv_conv = False
    run_validation = True

    validation_dirs = get_validation_dirs(package_dir)

    if run_main:
        for valid_dir in validation_dirs:
            relativepath = os.path.join(package_dir, 'validation', valid_dir)
            main.lane_detector_main(relativepath, boxes_fn, vp_fn, display_, verbose)

    if run_csv_conv:
        for valid_dir in validation_dirs:
            relativepath = os.path.join(package_dir, 'validation', valid_dir)
            csvpath = os.path.join(package_dir, 'validation', valid_dir, 'manual_lanes.csv')
            convert_manual_lanes(csvpath, relativepath)

    if run_validation:
        test_lanes_fn = 'lanes_output_new.json'
        true_positives = 0.
        false_positives = 0.
        false_negatives = 0.
        for valid_dir in validation_dirs[0:]:
            relativepath = os.path.join(package_dir, 'validation', valid_dir)
            tp, fp, fn = lane_detection_validation(relativepath, test_lanes_fn)
            true_positives += tp
            false_positives += fp
            false_negatives += fn

        precision = true_positives/(true_positives + false_positives)
        recall = true_positives/(true_positives + false_negatives)
        print 'Total validation results for {}'.format(test_lanes_fn)
        print 'Average precision: {} ({}/{})'.format(precision, true_positives, 
                                            (true_positives + false_positives))
        print 'Average recall: {} ({}/{})'.format(recall, true_positives,
                                            (true_positives + false_negatives))
        print 'F1 Score: {}'.format(2*(precision * recall)/(precision + recall))
