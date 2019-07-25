import os, sys
import datetime
import time
import codecs
import re, datetime
import argparse
from bs4 import BeautifulSoup


def extract_summary(input_filenames, output_folder):
    for index, filename in enumerate(input_filenames):
        game_id = filename[:filename.index("_")]
        recap_url = "http://www.espn.com/mlb/recap?"+game_id

        try:
            response = urllib2.urlopen(recap_url)
        except urllib2.HTTPError, e:
            if e.getcode()/100 == 5:
                print 'error3 ',e.getcode()
                continue
            else:
                raise
        html = response.read().decode('utf-8', 'ignore')
        with codecs.open(output_folder + filename, "w+", "utf-8") as f:
            f.write(html)
        soup = BeautifulSoup(html,"lxml")
        article = soup.find('div', attrs={'class': 'article-body'})
        if article == None:
            return
        paras = article.find_all('p')
        out_file = output_folder + filename
        with codecs.open(out_file, "w+", "utf-8") as f:
            for para in paras:
                f.write(para.get_text().replace("\n"," "))
                f.write("\n")
        f.close()
        if index%1000 == 0:
            print index
        time.sleep(1)  # delay between url request


parser = argparse.ArgumentParser(description='Extract summaries from html')
parser.add_argument('-recaps',type=str,
                    help='file containing the names of recaps')
parser.add_argument('-output_folder',type=str,
                    help='output folder')
args = parser.parse_args()

recaps = args.recaps
output_folder = args.output_folder
with codecs.open(recaps) as f:
    content = f.readlines()

input_filenames = [x.strip() for x in content]
extract_summary(input_filenames, output_folder)
