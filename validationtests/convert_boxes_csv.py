import csv, json, argparse, os

def rint(num):
    return int(round(float(num)))

def convert_manual_boxes(csvpath, relativepath, min_area=None, correct_ar=False):
    output_dict = {}
    with open(csvpath, 'rb') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if correct_ar:
                fname = row['Filename'] + '_.png'
            else:
                fname = row['Filename'] + '.png'

            width = float(row['Width'])
            height = float(row['Height'])

            if min_area and width*height < min_area:
                continue

            x1 = float(row['BX'])
            y1 = float(row['BY'])
            x2 = x1 + width
            y2 = y1 + height

            if correct_ar:
                entry = [rint(x1), rint(y1*.75), rint(x2), rint(y2*.75)]
            else:
                entry = [rint(x1), rint(y1), rint(x2), rint(y2)]
            
            if fname in output_dict:
                output_dict[fname].append(entry)
            else:
                output_dict[fname] = [entry]

    output_fname = relativepath + '/manual_boxes.json'
    with open(output_fname, 'w') as outfile:
       json.dump(output_dict, outfile, indent=4, sort_keys=True)
       print 'Wrote json output file'

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-v', '--videopath', help='Relative Path to Video', 
                    required=True)
    args = ap.parse_args()
    relativepath = args.videopath
    csvfilename = 'manual_boxes.csv'
    csvpath = os.path.join(relativepath, csvfilename)
    convert_manual_boxes(csvpath, relativepath)




