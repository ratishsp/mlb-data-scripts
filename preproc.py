import codecs
import json
import os
import argparse
from datetime import datetime


def sort_files_key(x):
    date_string = x[-15:-5]  # get the date portion of file name
    datetime_object = datetime.strptime(date_string, "%Y-%m-%d")
    return datetime_object

def prep_mlb(mlb_split_keys, input, output):
    with codecs.open(mlb_split_keys, "r", "utf-8") as f:
        lines = f.readlines()
        ntrain, nval, ntest = lines[0].split()
        ntrain, nval, ntest = int(ntrain), int(nval), int(ntest)
        lines = lines[1:]
        trkeys = set([thing.strip() for thing in lines[:ntrain]])
        valkeys = set([thing.strip() for thing in lines[ntrain:ntrain + nval]])
        testkeys = set([thing.strip() for thing in lines[ntrain + nval:]])
    train_index = 0
    train, val, test = [], [], []
    train_keys_order, val_keys_order, test_keys_order = [], [], []
    file_list = os.listdir(input)
    sorted_file_list = sorted(file_list, key= sort_files_key)
    for filename in sorted_file_list:
        with open(input + filename) as json_data:
            data = json.load(json_data)
        json_data.close()

        two_same_name_player_in_same_match = False
        for thing in data:
            # there are instances where there are two players of the same name in the same game; excluding such instances
            for player_name in ['Chris Young', 'Daniel Robertson', 'Miguel Gonzalez', 'Andy Phillips', 'Francisco Rodriguez', 'Mike Jacobs', 'Edgar Gonzalez', 'Mike Carp', 'Chris Smith']:
                if player_name in set(thing['box_score']['full_name'].values()):
                    if len([i for i, x in enumerate(thing['box_score']['full_name'].values()) if x == player_name]) > 1:
                        two_same_name_player_in_same_match = True
                        print 'player_name', player_name
            if two_same_name_player_in_same_match:
                two_same_name_player_in_same_match = False
                continue
            # get the key
            key = []
            key.append(thing['day'])
            key.append(thing['vis_name'])
            key.append(thing['home_name'])
            key.append(str(thing['vis_line']['team_runs']))
            key.append(str(thing['home_line']['team_runs']))
            key = "-".join(key)
            if key in testkeys:
                test.append(thing)
                test_keys_order.append(key)
            elif key in valkeys:
                val.append(thing)
                val_keys_order.append(key)
            else:
                train.append(thing)
                train_keys_order.append(key)
                if len(train) == 1000:
                    with codecs.open(output + "train"+str(train_index)+".json", "w+", "utf-8") as f:
                        json.dump(train, f)
                    train_index += 1
                    train = []
        print filename

    with codecs.open(output+"train"+str(train_index)+".json", "w+", "utf-8") as f:
        json.dump(train, f)
    with codecs.open(output+"valid.json", "w+", "utf-8") as f:
        json.dump(val, f)
    with codecs.open(output+"test.json", "w+", "utf-8") as f:
        json.dump(test, f)
    write_keys(test_keys_order, train_keys_order, val_keys_order, output)


def write_keys(test_keys_order, train_keys_order, val_keys_order, output):
    output_keys = ["Train"]
    output_keys.extend(train_keys_order)
    output_keys.append("Val")
    output_keys.extend(val_keys_order)
    output_keys.append("Test")
    output_keys.extend(test_keys_order)
    with codecs.open(output + "keys_file", "w+", "utf-8") as f:
        f.write("\n".join(output_keys))
        f.write("\n")
    f.close()


parser = argparse.ArgumentParser(description='Preprocessing to generate train, valid and test json')
parser.add_argument('-mlb_split_keys',type=str,
                    help='split for generating the train/valid/test files')
parser.add_argument('-input',type=str, help='input directory')
parser.add_argument('-output',type=str, help='output directory')
args = parser.parse_args()
mlb_split_keys = args.mlb_split_keys
input = args.input
output = args.output
prep_mlb(mlb_split_keys, input, output)
