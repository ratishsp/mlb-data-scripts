# -*- coding: utf-8 -*-
import os
import json
import codecs
import argparse

N_A = "N/A"
batting_attrib = ["first_name", "last_name", "a","ab","avg","bb","cs","e","h","hbp","hr",
                  "obp","po","pos","r","rbi","sb","sf","slg","so"]
pitching_attrib = ["bb","er","era","h","hr","l","loss","s", "np","r",
                   "save","so","sv","w","win", "ip1", "ip2"]
pitching_attrib = ["p_"+entry for entry in pitching_attrib]
total_attrib = batting_attrib
total_attrib.extend(pitching_attrib)
ls_keys = ["team_hits", "team_errors", "result", "team_name", "team_city"]
NUM_PLAYERS  = 46
HOME = "HOME"
AWAY = "AWAY"
DELIM = u"￨"


def box_preproc(entry):
    records = []
    home_players = range(46)
    vis_players = range(46,92)
    for ii, player_list in enumerate([home_players, vis_players]):
        for j in xrange(NUM_PLAYERS):
            player_key = str(player_list[j])
            player_name = entry['box_score']['full_name'][player_key]
            for k, key in enumerate(total_attrib):
                rulkey = key
                if player_key not in entry["box_score"][rulkey]:
                    continue
                val = entry["box_score"][rulkey][player_key]
                if val == 'N/A':
                    continue
                if key in ['sb', 'sf', 'e', 'po', 'a', 'cs', 'hbp', 'hr', 'p_hr', 'so', 'bb'] and int(val) == 0:
                    continue
                if key in ['avg', 'slg', 'obp'] and val == ".000":
                    continue
                record = []
                record.append(val.replace(" ","_"))  # format is val|entity|record_type|inning|home/away|play_id
                record.append(player_name.replace(" ","_"))
                record.append(rulkey)
                record.append("-1")
                record.append(HOME if ii == 0 else AWAY)
                record.append("-1")
                records.append(DELIM.join(record))

    for k, key in enumerate(ls_keys):
        record = []
        record.append(str(entry["home_line"][key]).replace(" ","_"))
        record.append(entry["home_name"].replace(" ","_"))
        record.append(key)
        record.append("-1")
        record.append(HOME)
        record.append("-1")
        records.append(DELIM.join(record))

    for k, key in enumerate(ls_keys):
        record = []
        record.append(str(entry["vis_line"][key]).replace(" ","_"))
        record.append(entry["vis_name"].replace(" ","_"))
        record.append(key)
        record.append("-1")
        record.append(AWAY)
        record.append("-1")
        records.append(DELIM.join(record))

    return records


def get_play_by_play(entry):
    play_by_play_records = []

    plays = entry["play_by_play"]
    play_index = 0
    for inning in range(1, len(entry['home_line']['innings'])+1):
        for top_bottom in ["top", "bottom"]:
            inning_plays = plays[str(inning)][top_bottom]
            for inning_play in inning_plays:
                if inning_play["runs"] == 0:
                    continue
                play_key = "play_"+str(play_index)
                if "batter" in inning_play:
                    append_play_by_play_with_key_as_val("batter", inning, inning_play, play_by_play_records, play_index,
                                                        top_bottom, inning_play["batter"])
                if "pitcher" in inning_play:
                    append_play_by_play_with_key_as_val("pitcher", inning, inning_play, play_by_play_records,
                                                        play_index, top_bottom, inning_play["pitcher"])

                append_play_by_play("o", inning, inning_play, play_by_play_records, play_index, top_bottom,
                                    play_key, greater_than_zero=True)
                append_play_by_play("b", inning, inning_play, play_by_play_records, play_index, top_bottom,
                                    play_key, greater_than_zero=True)
                append_play_by_play("s", inning, inning_play, play_by_play_records, play_index, top_bottom,
                                    play_key, greater_than_zero=True)
                append_play_by_play("home_team_runs", inning, inning_play, play_by_play_records, play_index, top_bottom,
                                    entry["home_name"])
                append_play_by_play("away_team_runs", inning, inning_play, play_by_play_records, play_index, top_bottom,
                                    entry["vis_name"])
                for baserunner_key in ["b1", "b2", "b3"]:
                    if baserunner_key in inning_play and len(inning_play[baserunner_key])>0 and inning_play[baserunner_key][0] != N_A:
                        for baserunner_instance in inning_play[baserunner_key]:
                            append_play_by_play_with_key_as_val(baserunner_key, inning, inning_play,
                                                                play_by_play_records, play_index, top_bottom,
                                                                baserunner_instance)
                if 'event2' in inning_play and inning_play['event2'] == 'Error' and 'fielder_error' in inning_play:
                    append_play_by_play_with_key_as_val(inning_play["event2"].lower(), inning, inning_play,
                                                        play_by_play_records, play_index, top_bottom,
                                                        inning_play["fielder_error"], subtype="_fielder")
                elif inning_play["event"]=='Field Error' and 'fielder_error' in inning_play :
                    append_play_by_play_with_key_as_val(inning_play["event"].lower(), inning, inning_play,
                                                        play_by_play_records, play_index, top_bottom,
                                                        inning_play["fielder_error"], subtype="_fielder")
                elif 'fielder_error' in inning_play :
                    append_play_by_play_with_key_as_val(inning_play["event"].lower(), inning, inning_play,
                                                        play_by_play_records, play_index, top_bottom,
                                                        inning_play["fielder_error"], subtype="_fielder")
                else:
                    append_play_by_play_with_key_as_val(inning_play["event"].lower(), inning, inning_play,
                                                        play_by_play_records, play_index, top_bottom, play_key)
                    if "event2" in inning_play:
                        append_play_by_play_with_key_as_val(inning_play["event2"].lower(), inning, inning_play,
                                                            play_by_play_records, play_index, top_bottom,
                                                            play_key)
                if "scorers" in inning_play and len(inning_play["scorers"])>0:
                    for scorer in inning_play["scorers"]:
                        append_play_by_play_with_key_as_val("scorer", inning, inning_play, play_by_play_records,
                                                            play_index, top_bottom, scorer)

                append_play_by_play("rbi", inning, inning_play, play_by_play_records, play_index, top_bottom,
                                    play_key, greater_than_zero=True)
                play_by_play_records.append(DELIM.join([str(inning),
                                                        play_key, "pl_inning",
                                                        str(inning), top_bottom, str(play_index)]))
                append_play_by_play("runs", inning, inning_play, play_by_play_records, play_index, top_bottom,
                                    play_key,
                                    greater_than_zero=True)
                append_play_by_play("error_runs", inning, inning_play, play_by_play_records, play_index, top_bottom,
                                    play_key,
                                    greater_than_zero=True)
                play_index += 1
    return play_by_play_records


def append_play_by_play(key, inning, inning_play, play_by_play_records, play_index, top_bottom, entity,
                        greater_than_zero=False):
    #val, type, name, inning, top/bottom, play_index
    if key in inning_play and ((greater_than_zero and int(inning_play[key]) > 0) or not greater_than_zero):
        play_by_play_records.append(DELIM.join(
            [str(inning_play[key]), entity.replace(" ", "_"), "pl_"+key, str(inning), top_bottom,
             str(play_index)]))


def append_play_by_play_with_key_as_val(key, inning, inning_play, play_by_play_records, play_index, top_bottom, entity,
                                        subtype=""):
    play_by_play_records.append(DELIM.join(
        ["pl_" + key.replace(" ", "_") + subtype, entity.replace(" ", "_"), "pl_" + key.replace(" ", "_") + subtype,
         str(inning), top_bottom, str(play_index)]))


def process(input_folder, output_src, output_tgt, type):
    output_file = codecs.open(output_src, mode='w', encoding='utf-8')
    target_file = codecs.open(output_tgt, mode='w', encoding='utf-8')
    for filename in os.listdir(input_folder):
        if not type in filename:
            continue
        d = None
        with codecs.open(input_folder+filename, encoding='utf-8') as json_data:
            d = json.load(json_data)
        json_data.close()
        print 'filename', filename
        for entry in d:
            output = box_preproc(entry)
            play_by_play = get_play_by_play(entry)
            output.extend(play_by_play)
            output_file.write(" ".join(output))
            output_file.write("\n")
            target_file.write(" ".join(entry['summary']))
            target_file.write("\n")

    output_file.close()
    target_file.close()


parser = argparse.ArgumentParser(description='Extract summaries from html')
parser.add_argument('-input_folder',type=str,
                    help='input folder')
parser.add_argument('-output_src',type=str,
                    help='output src file')
parser.add_argument('-output_tgt',type=str,
                    help='output tgt file')
parser.add_argument('-type', type=str, default='train',
                    choices=['train', 'valid', 'test'],
                    help='Type of dataset to generate. Options [train|valid|test]')
args = parser.parse_args()

input_folder = args.input_folder
output_src = args.output_src
output_tgt = args.output_tgt
type = args.type
process(input_folder, output_src, output_tgt, type)
