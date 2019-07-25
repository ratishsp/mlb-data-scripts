# mlb-data

This repo contains scripts to create the **MLB** dataset introduced in the paper [**Data-to-text Generation with Entity Modeling**](https://arxiv.org/abs/1906.03221) (Puduppully, R., Dong, L., & Lapata, M.; ACL 2019). 

# Prerequisites
- Install the [mlbgame-api](https://github.com/ratishsp/mlbgame-api)
```
pip install git+https://github.com/ratishsp/mlbgame-api.git
```

# Steps to create the dataset
Run the following scripts in sequence
- [boxscore_data.py](https://github.com/ratishsp/mlb-data-scripts/blob/master/boxscore_data.py). It requires the argument '-year'. The values to be passed are 0, 1, 2..10. For 0 it will collect the records for the year 2018, for 1 the year 2017 and so on.
```
python boxscore_data.py -year 1 -output ~/mlb-data/api-output/  # get the data for year 2017
```
- [extract_summaries_from_recap_html.py](https://github.com/ratishsp/mlb-data-scripts/blob/master/extract_summaries_from_recap_html.py) extracts the recaps from the html. The names of the htmls to be downloaded is available in the file [recap_file_names.txt](https://github.com/ratishsp/mlb-data-scripts/blob/master/recap_file_names.txt)
 ```
python extract_summaries_from_recap_html -recaps ~/mlb-data/recap_file_names.txt -output_folder ~/mlb-data/html-output/
 ```

- [clean_summaries.py](https://github.com/ratishsp/mlb-data-scripts/blob/master/clean_summaries.py) cleans the html of quotations and text incidental to the game.
 ```
python clean_summaries.py -input_folder ~/mlb-data/html-output/ -output_folder ~/mlb-data/html-output-cleaned/
 ```
 - [create_combined_dataset.py](https://github.com/ratishsp/mlb-data-scripts/blob/master/create_combined_dataset.py) results in a dataset with boxscores and summaries.
 ```
python create_combined_dataset.py -input_folder ~/mlb-data/api-output/ -input_summaries ~/mlb-data/html-output-cleaned/ -output_folder ~/mlb-data/combined/
 ```
- [preproc.py](https://github.com/ratishsp/mlb-data-scripts/blob/master/preproc.py) preprocesses the dataset into train, validation and test splits. The splits are defined in the file [mlb_split_keys.txt](https://github.com/ratishsp/mlb-data-scripts/blob/master/mlb_split_keys.txt).
```
python preproc.py -input ~/mlb-data/combined/ -mlb_split_keys ~/mlb-data/mlb_split_keys.txt -output ~/mlb-data/splits/
```
- [data2text_input_formatter.py](https://github.com/ratishsp/mlb-data-scripts/blob/master/data2text_input_formatter.py) formats the dataset to OpenNMT-py format.
```
python data2text_input_formatter.py -input_folder ~/mlb-data/splits/ -output_src ~/mlb-data/formatted/src_train.txt -output_tgt ~/mlb-data/formatted/tgt_train.txt
```

