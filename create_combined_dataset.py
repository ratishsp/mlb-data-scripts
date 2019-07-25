import sys
import os
import codecs
import json
import nltk
import json
import glob
import mlbgame
from datetime import datetime
import argparse


def create_json(input_folder, input_summaries, output_folder):
    for filename in os.listdir(input_folder):
        d = None
        with codecs.open(input_folder+filename) as json_data:
            d = json.load(json_data)
        print 'filename',input_folder+filename
        output = []
        for entry in d:
            datetime_object = datetime.strptime(entry['day'], '%m_%d_%y')
            begin_date = mlbgame.important_dates(datetime_object.year).first_date_seas
            begin_date = datetime.strptime(begin_date, '%Y-%m-%dT%H:%M:%S')
            if datetime_object < begin_date:
                print 'datetime_object', datetime_object, filename
                continue

            html_file_name = []
            html_file_name.append(datetime_object.strftime("%Y%m%d"))
            visname_homename = entry['vis_name'].replace(" ", "_") + "-" + entry['home_name'].replace(" ", "_")
            visname_homename = visname_homename.replace('D-backs', 'Diamondbacks')
            html_file_name.append(visname_homename)
            html_file_name.append(str(entry['vis_line']['team_runs']) + "-" + str(entry['home_line']['team_runs']))

            files = glob.glob(input_summaries+"*" +"_".join(html_file_name))
            if len(files) < 1:
                print input_summaries+"*"+"_".join(html_file_name) + " not found"
            elif len(files) > 1:
                print input_summaries + "*" + "_".join(html_file_name) + " multiple found"
            else:
                fname = files[0]
                with codecs.open(fname, encoding='utf-8') as f:
                    content = f.readlines()
                # you may also want to remove whitespace characters like `\n` at the end of each line
                content = [x.strip() for x in content]
                text = " ".join(content)
                words = nltk.word_tokenize(text)
                newtokes = []
                [newtokes.append(toke) if toke[0].isupper() or '-' not in toke
                 else newtokes.extend(toke.replace('-', " - ").split()) for toke in words]
                entry['summary'] = newtokes
                output.append(entry)

        if len(output) > 0:
            with codecs.open(output_folder+'data_'+filename, 'w+') as outfile:
                json.dump(output, outfile)
            outfile.close()


parser = argparse.ArgumentParser(description='Process years')
parser.add_argument('-input_folder',type=str,
                    help='input folder containg box and line score stats')
parser.add_argument('-input_summaries',type=str,
                    help='input folder containing summaries')
parser.add_argument('-output_folder',type=str,
                    help='output folder')
args = parser.parse_args()

input_folder = args.input_folder
input_summaries = args.input_summaries
output_folder = args.output_folder

create_json(input_folder, input_summaries, output_folder)
