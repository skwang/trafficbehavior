import json, os, numpy as np

# Open the file (filename) located at relativepath, split it by new lines, and 
# return the contents of all lines as a list
def load_list(relativepath, filename):
    output = []
    with open(os.path.join(relativepath, filename), 'r') as f:
        for line in f:
            output.append(line.split('\n')[0])
    return output

# Open the json (filename) located at relativepath, and return it as a 
# dictionary
def load_json(relativepath, filename):
    output_dict = {}
    with open(os.path.join(relativepath, filename)) as data_file:
        output_dict = json.load(data_file)
    return output_dict

# Load the video fnames at relativepath as a list
def load_fnames(relativepath):
    return load_list(relativepath, 'list.txt')

# Load the video fnames at relativepath as a list
def load_vehicle_boxes(relativepath, use_hand_labeled=False):
    boxes_fname = 'boxes_output_7.5_7.5.json' #'vehicle_boxes.json' 
    if use_hand_labeled:
        boxes_fname = 'manual_boxes.json' #'vehicle_boxes_hand.json'
    return load_json(relativepath, boxes_fname)

# Needed to properly save the numpy arrays into json
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)

# Save the output_dict as a JSON at relativepath with name as filename
def save_json(relativepath, filename, output_dict):
    with open(os.path.join(relativepath, filename), 'w') as outfile:
        json.dump(output_dict, outfile, indent=4,
                    sort_keys=True, cls=NumpyEncoder)

def save_list(relativepath, filename, output_list):
    with open(os.path.join(relativepath, filename), 'w') as f:
        for line in output_list:
            f.write(line + '\n')
