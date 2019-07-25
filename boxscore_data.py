import re, datetime
import json, sys, os, codecs
import mlbgame
import numpy as np
import nltk
import argparse
from HTMLParser import HTMLParser

NA = 'N/A'

NUMBER_PLAYERS = 46

months = {"January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
          "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12}
batting_attrib = ["team","first_name", "last_name", "full_name", "a","ab","ao","avg","bb","bo","cs","d","e","fldg","go","h","hbp","hr",
                  "lob","obp","ops","po","pos","r","rbi","sac","sb","sf","slg","so","t"]
pitching_attrib = ["bb","bf","bs","er","era","game_score","h","hld","hr","l","loss","np","out","r","s",
                   "save","so","sv","w","win", "ip1", "ip2"]

TITLE_CASE_REGEX = "(?:[A-Z]\.?[a-z]*\s*)+"
def add_to_data(day, game, data):
    try:
        innings = mlbgame.box_score(game.game_id).innings
    except ValueError, e:
        print str(e)
        if str(e) == 'Could not find a game with that id.':
            return
    except KeyError, e:
        print 'key error', e
        if day.month <= 3:
            return

    for i in range(len(innings)):
        assert innings[i]['inning'] == i+1
    try:
        stats = mlbgame.stats.player_stats(game.game_id)
    except ValueError, e:
        print str(e)
        if str(e) == 'Could not find a game with that id.':
            return
    players = mlbgame.game.players(game.game_id)
    first_name = {}
    last_name = {}
    full_name = {}
    team = {}
    id_to_index = {}
    id_to_name = {}
    id_to_firstname = {}
    id_to_lastname = {}
    last_name_to_full_name = {}
    all_ents = set()
    for index, player in enumerate(players["home_team"]["players"]):
        first_name[str(index)] = player["first"]
        last_name[str(index)] = player["last"]
        full_name[str(index)] = player["first"] + " " + player["last"]
        team[str(index)] = game.home_team
        id_to_index[player["id"]] = index
        id_to_name[player["id"]] = player["first"] + " " + player["last"]
        id_to_firstname[player["id"]] = player["first"]
        id_to_lastname[player["id"]] = player["last"]
        last_name_to_full_name[player["last"]] = player["first"] + " " + player["last"]
    for extra in range(index+1, NUMBER_PLAYERS):
        first_name[str(extra)] = NA
        last_name[str(extra)] = NA
        full_name[str(extra)] = NA
        team[str(extra)] = NA

    for index, player in enumerate(players["away_team"]["players"]):
        index_ = index + NUMBER_PLAYERS
        first_name[str(index_)] = player["first"]
        last_name[str(index_)] = player["last"]
        full_name[str(index_)] = player["first"] + " " + player["last"]
        team[str(index_)] = game.away_team
        id_to_index[player["id"]] = index_
        id_to_name[player["id"]] = player["first"] + " " + player["last"]
        id_to_firstname[player["id"]] = player["first"]
        id_to_lastname[player["id"]] = player["last"]
        last_name_to_full_name[player["last"]] = player["first"] + " " + player["last"]

    all_ents.update(id_to_firstname.values())
    all_ents.update(id_to_name.values())
    all_ents.update(id_to_lastname.values())
    for extra in range(index+1, NUMBER_PLAYERS):
        extra_ = extra + NUMBER_PLAYERS
        first_name[str(extra_)] = NA
        last_name[str(extra_)] = NA
        full_name[str(extra_)] = NA
        team[str(extra_)] = NA
    if len(first_name) > 2*NUMBER_PLAYERS:
        print "lot of players", len(first_name)
        return
    #assert len(first_name)==2*NUMBER_PLAYERS

    box_score = {}
    for attrib in batting_attrib:
        box_score[attrib] = {}
    for attrib in pitching_attrib:
        box_score["p_"+attrib] = {}

    for index in range(2*NUMBER_PLAYERS):
        init_batting_record(box_score, str(index))
        init_pitching_record(box_score, str(index))

    box_score["first_name"] = first_name
    box_score["last_name"] = last_name
    box_score["full_name"] = full_name
    box_score["team"] = team

    for batting_record in stats["home_batting"]:
        index = str(id_to_index[batting_record["id"]])
        set_batting_record(batting_record, box_score, index)

    for batting_record in stats["away_batting"]:
        index = str(id_to_index[batting_record["id"]])
        set_batting_record(batting_record, box_score, index)

    for pitching_record in stats["home_pitching"]:
        index = str(id_to_index[pitching_record["id"]])
        set_pitching_record(pitching_record, box_score, index)

    for pitching_record in stats["away_pitching"]:
        index = str(id_to_index[pitching_record["id"]])
        set_pitching_record(pitching_record, box_score, index)

    vis_inning_map = {}
    for inning_index, inning in enumerate(innings):
        vis_inning_map["inn"+str(inning_index + 1)] = inning['away']
    home_inning_map = {}
    for inning_index, inning in enumerate(innings):
        home_inning_map["inn" + str(inning_index + 1)] = inning['home']

    play_by_play = {}
    for i in range(len(innings)):
        play_by_play[str(i+1)] = {'top':[], 'bottom':[]}

    events = mlbgame.events.game_events(game.game_id)
    num = 0
    for i in range(len(innings)):
        for team in ['top', 'bottom']:
            if str(i+1) not in events or team not in events[str(i+1)]:
                break
            for play in events[str(i+1)][team]:
                upd_play = {}
                if 'pitcher' not in play:
                    if "scores" in play['des'] or 'Stolen Base' in play['event'] or play['event'] == "Wild Pitch":
                        upd_play['b'] = play['b']  # balls
                        upd_play['s'] = play['s']  # strikes
                        upd_play['o'] = play['o']  # outs
                        if 'away_team_runs' in play:
                            upd_play['away_team_runs'] = play['away_team_runs']
                        if 'home_team_runs' in play:
                            upd_play['home_team_runs'] = play['home_team_runs']
                        description = nltk.sent_tokenize(play['des'])
                        scorers = get_scorers(description, id_to_lastname, id_to_name, last_name_to_full_name)
                        upd_play['scorers'] = scorers
                        if "scores" in play['des']:
                            if len(scorers) == 0:
                                print "scorer not found in action"
                            #assert len(scorers) > 0
                        upd_play['event'] = play['event']
                        upd_play['runs'] = len(scorers)
                        event2_error = 'event2' in play and play['event2'] == 'Error'
                        set_fielder_error(all_ents, description, event2_error, id_to_name, play, upd_play)
                        set_stolen_base(description, id_to_name, upd_play)
                        set_stolen_third_base(description, id_to_name, upd_play)
                        set_batting(description, id_to_name, upd_play)
                        set_pitching(description, id_to_name, upd_play)
                        set_bases(description, id_to_name, upd_play, "2nd")
                        set_bases(description, id_to_name, upd_play, "3rd")
                        set_passed_ball(description, id_to_name, upd_play)
                        play_by_play[str(i + 1)][team].append(upd_play)
                    continue
                for attrib in ['batter', 'pitcher']:
                    upd_play[attrib] = id_to_name[play[attrib]]
                for attrib in ['b1', 'b2', 'b3']:
                    upd_play[attrib] = []
                    if len(play[attrib])>0:
                        for id in play[attrib].split():
                            upd_play[attrib].append(id_to_name[id])
                    else:
                        upd_play[attrib] = [NA]
                assert type(play['b1']) is str
                assert type(play['b2']) is str
                assert type(play['b3']) is str
                upd_play['event'] = play['event']
                if 'away_team_runs' in play:
                    upd_play['away_team_runs'] = play['away_team_runs']
                if 'home_team_runs' in play:
                    upd_play['home_team_runs'] = play['home_team_runs']
                if 'rbi' in play:
                    upd_play['rbi'] = play['rbi']
                upd_play['b'] = play['b']   # balls
                upd_play['s'] = play['s']   # strikes
                upd_play['o'] = play['o']   # outs
                #assert int(play['num']) == num + 1,
                if int(play['num']) != num + 1:
                    print "expected play['num'] "+play['num']+" and num+1 "+str(num+1)+" to be equal "
                num += 1
                description = nltk.sent_tokenize(play['des'])
                scorers = get_scorers(description, id_to_lastname, id_to_name, last_name_to_full_name)
                upd_play['scorers'] = scorers
                upd_play['runs'] = len(scorers) + 1 if upd_play['event'] == 'Home Run' else len(scorers)

                event2_error = 'event2' in play and play['event2'] == 'Error'
                set_fielder_error(all_ents, description, event2_error, id_to_name, play, upd_play)
                if play['event'] == 'Field Error' or event2_error:
                    if 'rbi' in upd_play:
                        runs_scored_due_to_error = (len(scorers) - int(upd_play['rbi']))
                    else:
                        runs_scored_due_to_error = len(scorers)
                    if runs_scored_due_to_error > 0:
                        upd_play['error_runs'] = runs_scored_due_to_error
                else:
                    if upd_play['event'] != 'Home Run':
                        if len(scorers)>0:
                            if 'rbi' in upd_play:
                                if int(upd_play['rbi']) != len(scorers):
                                    print 'update the stats for rbi'
                            else:
                                print 'rbi not present'#, json.dumps(play, indent=4)
                        #assert len(scorers) == 0 or len(scorers) == int(upd_play['rbi'])
                    if upd_play['event'] == 'Home Run':
                        if len(scorers)+1 != int(upd_play['rbi']):
                            print 'home run scores do not match'
                        #assert len(scorers)+1 == int(upd_play['rbi'])

                play_by_play[str(i+1)][team].append(upd_play)

    if data is not None and hasattr(game, 'w_team'):
        home_city = mlbgame.game.overview(game.game_id)["home_team_city"]
        vis_city = mlbgame.game.overview(game.game_id)["away_team_city"]
        data.append(
            {
                "day" : day.strftime("%0m_%0d_%y"),
                "home_name": game.home_team,
                 "vis_name": game.away_team,
                 "home_city": home_city,
                 "vis_city": vis_city,
                 "vis_line": {
                     "team_errors":game.away_team_errors,
                     "team_hits":game.away_team_hits,
                     "team_runs":game.away_team_runs,
                     "innings": vis_inning_map,
                     "result": "win" if game.w_team == game.away_team else "loss",
                     "team_name": game.away_team,
                     "team_city" : vis_city
                 },
                 "home_line": {
                     "team_errors": game.home_team_errors,
                     "team_hits": game.home_team_hits,
                     "team_runs": game.home_team_runs,
                     "innings": home_inning_map,
                     "result": "win" if game.w_team == game.home_team else "loss",
                     "team_name": game.home_team,
                     "team_city" : home_city
                 },
                 "box_score": box_score,
                "play_by_play": play_by_play
            }
        )


def set_stolen_base(description, id_to_name, upd_play):
    exp = TITLE_CASE_REGEX + " steals \([\d]+\) 2nd base"
    for desc_sent in description:
        if "2nd base" not in desc_sent:
            continue
        match = re.search(exp, desc_sent)
        if match is None:
            continue
        output = match.group(0)
        output = output[:output.index("steals")].strip()
        upd_play['b2'] = []
        if output in id_to_name.values():
            upd_play['b2'].append(output)


def set_stolen_third_base(description, id_to_name, upd_play):
    exp = TITLE_CASE_REGEX + " steals \([\d]+\) 3rd base"
    for desc_sent in description:
        if "3rd base" not in desc_sent:
            continue
        match = re.search(exp, desc_sent)
        if match is None:
            continue
        output = match.group(0)
        output = output[:output.index("steals")].strip()
        upd_play['b3'] = []
        if output in id_to_name.values():
            upd_play['b3'].append(output)


def set_batting(description, id_to_name, upd_play):
    exp = "With "+TITLE_CASE_REGEX + " batting"
    for desc_sent in description:
        if "batting" not in desc_sent:
            continue
        match = re.search(exp, desc_sent)
        if match is None:
            continue
        output = match.group(0)
        output = output[output.index("With") +len("With"):output.index("batting")].strip()
        if output in id_to_name.values():
            upd_play['batter'] = output


def set_pitching(description, id_to_name, upd_play):
    exp = "wild pitch by "+TITLE_CASE_REGEX
    for desc_sent in description:
        if "pitch" not in desc_sent:
            continue
        match = re.search(exp, desc_sent)
        if match is None:
            continue
        output = match.group(0)
        output = output[output.index("by") +len("by"):].strip()
        if output in id_to_name.values():
            upd_play['pitcher'] = output


def set_bases(description, id_to_name, upd_play, key):
    if key == "2nd":
        exp = TITLE_CASE_REGEX + " to 2nd"
        upd_play_key = "b2"
    elif key == "3rd":
        exp = TITLE_CASE_REGEX + " to 3rd"
        upd_play_key = "b3"
    set_base(description, id_to_name, upd_play, key, exp, upd_play_key)


def set_base(description, id_to_name, upd_play, key, exp, upd_play_key):
    for desc_sent in description:
        if key not in desc_sent:
            continue
        match = re.search(exp, desc_sent)
        if match is None:
            continue
        output = match.group(0)
        output = output[:output.index("to "+key)].strip()
        upd_play[upd_play_key] = []
        if output in id_to_name.values():
            upd_play[upd_play_key].append(output)


def set_passed_ball(description, id_to_name, upd_play):
    exp = "passed ball by "+TITLE_CASE_REGEX
    for desc_sent in description:
        if "passed ball" not in desc_sent:
            continue
        match = re.search(exp, desc_sent)
        if match is None:
            continue
        output = match.group(0)
        output = output[output.index("by") +len("by"):].strip()
        if output in id_to_name.values():
            upd_play['fielder_error'] = output


def set_fielder_error(all_ents, description, event2_error, id_to_name, play, upd_play):
    if play['event'] == 'Field Error' or event2_error:
        if event2_error:
            upd_play['event2'] = play['event2']
        for desc_sent in description:
            if "error by" in desc_sent:
                match_position = desc_sent.find("error by") + len("error by")
                to_analyze = desc_sent[match_position:]
                to_analyze_words = nltk.word_tokenize(to_analyze.strip())
                player_found = False
                for word_index in range(len(to_analyze_words)):
                    if to_analyze_words[word_index] in all_ents:
                        second_index = 1  # assume match for full name such as two word Jose Iglesias
                        while word_index + second_index <= len(to_analyze_words):
                            if " ".join(to_analyze_words[
                                        word_index:word_index + second_index + 1]) not in id_to_name.values():
                                second_index += 1  # match for names such as Ivan De Jesus
                            else:
                                player_found = True
                                break
                        break
                if player_found:
                    upd_play['fielder_error'] = " ".join(
                        to_analyze_words[word_index:word_index + second_index + 1])
                    # assert player_found


def get_scorers(description, id_to_lastname, id_to_name, last_name_to_full_name):
    scorers = []
    for desc_sent in description:
        if desc_sent.endswith('scores.'):
            name_candidate = desc_sent[:-len('scores.')].strip()
            last_name_candidate_index = name_candidate.rfind(" ")
            if last_name_candidate_index != -1:
                last_name_candidate = name_candidate[last_name_candidate_index:].strip()
            else:
                last_name_candidate = name_candidate
            scorers_found = True
            if name_candidate in id_to_name.values():
                scorer = name_candidate
            elif re.sub(' +', ' ', name_candidate) in id_to_name.values():  # for cases such as
                # Michael A.   Taylor scores.
                scorer = re.sub(' +', ' ', name_candidate)
            elif re.sub(' +', ' ', name_candidate).replace(" ", "", 1) in id_to_name.values():  # replacing
                #  first space for cases such as A.  J.   Pierzynski scores.
                scorer = re.sub(' +', ' ', name_candidate).replace(" ", "", 1)
            elif last_name_candidate in id_to_lastname.values():
                # handling cases where only last name is present 'Realmuto scores.'
                scorer = last_name_to_full_name[last_name_candidate]
            else:
                print 'absent', desc_sent
                scorers_found = False
                # assert False
            if scorers_found:
                scorers.append(scorer)
    return scorers


def set_pitching_record(pitching_record, box_score, index):
    box_score["pos"][index] = pitching_record["pos"]
    box_score["p_ip1"][index] = str(int(pitching_record["out"])/3)
    box_score["p_ip2"][index] = str(int(pitching_record["out"]) % 3)+"/3" if int(pitching_record["out"]) % 3 > 0 else NA
    for attrib in pitching_attrib:
        if attrib in pitching_record:
            box_score["p_"+attrib][index] = pitching_record[attrib]


def init_pitching_record(box_score, index):
    for attrib in pitching_attrib:
        box_score["p_"+attrib][index] = NA


def set_batting_record(batting_record, box_score, index):
    for attrib in batting_attrib:
        if attrib in batting_record:
            box_score[attrib][index] = batting_record[attrib]


def init_batting_record(box_score, index):
    for attrib in batting_attrib:
        box_score[attrib][index] = NA


def align_mlb(args):
    output_folder = args.output

    #start_day = datetime.date(2003, 1, 1)
    day = datetime.date(2018, 9, 15)
    #day = datetime.date(2016, 5, 16)
    #start_day = datetime.date(2003, 3, 10)
    if args.year == 0:
        day = datetime.date(2018, 9, 15)
        start_day = datetime.date(2018, 1, 1)
    else:
        day = datetime.date(2018-args.year, 12, 31)
        start_day = datetime.date(2018-args.year, 1, 1)
    data = []

    while day >= start_day:
        print day
        begin_date = mlbgame.important_dates(day.year).first_date_seas
        begin_date = datetime.datetime.strptime(begin_date, '%Y-%m-%dT%H:%M:%S')
        if day < begin_date.date() and day>datetime.date(day.year,1,1):
            print 'skip',day
            day = day - datetime.timedelta(days=1)
            continue
        games = mlbgame.games(day.year, day.month, day.day)
        games = mlbgame.combine_games(games)
        for game in games:
            print(game), game.game_id
            try:
                status_ = mlbgame.game.overview(game.game_id)['status']
            except ValueError, e:
                print 'Overview', str(e)
                continue

            if status_ not in ['Postponed', 'Cancelled']:
                try:
                    add_to_data(day, game, data)
                except Exception, e:
                    print 'Overall exception', str(e)
            else:
                print 'Postponed/ Cancelled',(game), game.game_id
            #print data
            #assert False
        #break
        if day.day == 1:
            with codecs.open(output_folder+"mlbaligned"+str(day)+".json", "w+", "utf-8") as f:
                json.dump(data, f)
            data = []
        day = day - datetime.timedelta(days=1)

parser = argparse.ArgumentParser(description='Process years')
parser.add_argument('-year',type=int, default=0,
                    help='an integer for the year')
parser.add_argument('-output',type=str, help='output directory')
args = parser.parse_args()
align_mlb(args)
