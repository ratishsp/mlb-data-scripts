import os, sys
import codecs


def clean_summaries(input_folder, output_folder):
    for filename in os.listdir(input_folder):
        with codecs.open(input_folder+filename, encoding='utf-8') as f:
            content = f.readlines()
        content = [x.strip() for x in content]

        output = []
        for line in content:
            if len(line) == 0:
                continue
            if line.isupper() or line.lower().strip()=='game notes':  # skip from section containing game notes as it
                # contains not relevant information
                break
            elif "said" in line and line.count("\"")>1:  # skip lines containing utterances by players
                continue
            else:
                output.append(line)

        if len(output) > 0:
            output_file = codecs.open(output_folder+filename, encoding='utf-8', mode='w+')
            output_file.write("\n".join(output))
            output_file.close()

parser = argparse.ArgumentParser(description='Clean summaries')
parser.add_argument('-input_folder',type=str,
                    help='input folder')
parser.add_argument('-output_folder',type=str,
                    help='output folder')
args = parser.parse_args()

input_folder = args.input_folder
output_folder = args.output_folder
clean_summaries(input_folder, output_folder)
