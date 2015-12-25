
# coding: utf-8


from __future__ import division
import pandas as pd
import numpy as np
import re
import pickle
from pprint import pprint
import matplotlib.pyplot as plt 
import matplotlib
import statsmodels.api as sm
import statsmodels.formula.api as smf
import requests
from bs4 import BeautifulSoup
import nltk
from textblob import TextBlob
from nltk.util import ngrams
from collections import defaultdict
from operator import itemgetter
from nltk.corpus import stopwords
from difflib import SequenceMatcher
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from nltk.collocations import *
from sklearn.feature_extraction.text import CountVectorizer
import urllib2
import datetime
from datetime import date
from datetime import timedelta
from datetime import datetime
from datetime import time
import json
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import os


date_string = date.today().strftime("%m-%d-%Y")
yesterday_string = (date.today()- timedelta(1)).strftime("%m-%d-%Y")


def move_columns(df, col_name, slot):
    cols = df.columns.tolist()
    index = cols.index(col_name)
    cols.insert(slot, cols.pop(index))
    return cols

def create_master(df_dict, sort_column='GAME_DATE', ascending=True):
    index = df_dict.keys()
    first = df_dict[index[0]]
    dfm = first.sort(sort_column, ascending=ascending)
    for key in index[1:]:
        if type(df_dict[key]) != tuple:
            df = df_dict[key].sort(sort_column, ascending=ascending)
            dfm = pd.concat([dfm, df])
    dfm.sort(sort_column, ascending=ascending, inplace=True)
    dfm.drop_duplicates(inplace=True)
    dfm.reset_index(drop=True, inplace=True)
    cols = move_columns(dfm, 'GAME_DATE', 1)
    dfm = dfm[cols]
    return dfm

def make_df_dict(df, col_name, ordered=True):
    df_dict = {}
    if ordered:
        df = df.sort('GAME_DATE')
    for key in df[col_name].unique():
        df_dict[key] = df[df[col_name]==key].reset_index(drop=True)
    return df_dict

def get_days_box_scores(month=11, day=26, year=2015):
    date = str(month) + "%2F" + str(day) + "%2F" + str(year)
    date_key = date.replace("%2F",'-')
    response = requests.get("http://stats.nba.com/stats/scoreboardV2?DayOffset=0&LeagueID=00&gameDate=" + date)
    game_date_json = response.json()
    return date_key, game_date_json

def get_box_scores(days_back = 5):
    box_score_data = {}
    month, day, year = date.today().strftime("%m %d %Y").split()
    month, day, year = int(month), int(day), int(year)
    daysInMonths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    days_dict = dict(zip(range(1,13), daysInMonths))
    for i in range(day, day-days_back, -1):
        print i
        if i == 0:
            if month == 1:
                year += -1
                month, day = 12, i + 31
            else:
                month += -1
                i = i + days_dict[month]
        elif i < 0:
            i = i + days_dict[month]
            if i == 0:
                month += -1
                i = i + days_dict[month]
            elif i < 0:
                i = i + days_dict[month]

        data = get_days_box_scores(month=month, day=i, year=year)
        box_score_data[data[0]] = data[1]
    return box_score_data

def get_games(box_score_data):
    date_of_games = {}
    for date, data in box_score_data.iteritems():
        game_date_list = []
        for stats in data['resultSets'][0]['rowSet']:
            game_date = dict(zip(data['resultSets'][0]['headers'], stats))
            game_date_list.append(game_date)
        date_of_games[date] = game_date_list
    return date_of_games


def get_ids(box_score_data):
    games_by_date = {}
    date_of_games = get_games(box_score_data)
    for date, game_list in date_of_games.iteritems():
        matchup_dict = {}
        for game in game_list:
            matchup_dict[game['GAME_ID']] = game['GAMECODE'][-6:-3] + ' ' + game['GAMECODE'][-3:]
        games_by_date[date] = matchup_dict
    return games_by_date  


def get_boxscores(box_score_data, player_dict):
    game_id_dict = get_ids(box_score_data)
    box_scores = {}
    stats_by_date = {}
    for date, game_ids in game_id_dict.iteritems():
        games = []
        keys = game_ids.keys()
        for key in keys:
            response = requests.get("http://stats.nba.com/stats/boxscoretraditionalv2?EndPeriod=10&EndRange=28800&RangeType=0&StartPeriod=1&StartRange=0&GameID=" + key)
            game_log_json = response.json()
            box_scores[key] = game_log_json
            stat_lines = {}
            if len(game_log_json['resultSets'][0]['rowSet']) != 0:
                for stats in game_log_json['resultSets'][0]['rowSet']:
                    game_log = {}
                    game_log = dict(zip(game_log_json['resultSets'][0]['headers'], stats))
                    player_dict[game_log[u'PLAYER_ID']] = game_log[u'PLAYER_NAME']
                    stat_lines[game_log[u'PLAYER_NAME']] = game_log
            games.append(stat_lines)
        stats_by_date[date] = games
        print 'day scraped'
    return stats_by_date

def boxscore_dfs(daily_stats_dict, box_score_ids):
    stats_dict = defaultdict(list)
    stats_dfs = {}
    for date, games in daily_stats_dict.iteritems():
        for game in games:
            for player, stats in game.iteritems():
                stats['GAME_DATE'] = datetime.strptime(date, '%m-%d-%Y')
                stats['MATCHUP'] = get_ids(box_score_ids)[date][stats['GAME_ID']]
                stats_dict[player].append(stats)
    for player, games in stats_dict.iteritems():
        df = pd.Series(games[0])
        if len(games) > 1:
            for game in games[1:]:
                row = pd.Series(game)
                df = pd.concat([df, row], axis=1)
        df = pd.DataFrame(df).T
        df.reset_index(drop=True, inplace=True)
        stats_dfs[player] = df
    return stats_dfs


def match_players_official(dfm, daily_projections, official_ids=None):
    if not official_ids:
        official_ids = {}
    for player in daily_projections[yesterday_string]['Player'].unique():
        candidates = dfm['PLAYER_NAME'].tolist()
        official_ids[player] = process.extractOne(player, candidates)[0]
    return official_ids


def get_fd_scores(df, scores_dict=None):
    if not scores_dict:
        scores_dict = {}
    df['MIN'].str.replace('None', np.nan)
    df.dropna(inplace=True)
    df['FD'] = np.round(df.apply(lambda x: x['PTS'] + x['REB']*1.2 + x['AST']*1.5 + x['BLK']*2 + x['STL']*2 - x['TO'], axis=1), 2)
    scores_dict[yesterday_string] = dict(zip(df['PLAYER_NAME'].tolist(), np.round(df['FD'],2).tolist()))
    return scores_dict


def add_results(df, official_ids, scores_dict):
    df['FD'] = df['Player'].apply(lambda x: np.round(scores_dict[yesterday_string][official_ids[x]],2) if official_ids[x] in scores_dict[yesterday_string] else 0)
    return df


def index_results(daily_projections, official_ids, scores_dict, daily_results_dict=None):
    if not daily_results_dict:
        daily_results_dict = {}
    daily_results_dict[yesterday_string] = add_results(daily_projections[yesterday_string], official_ids, scores_dict)
    return daily_results_dict


def main():

    with open('team_city_dict.pkl', 'r') as picklefile:
     team_dict = pickle.load(picklefile)

    with open('daily_projections.pkl', 'r') as picklefile:
         daily_projections = pickle.load(picklefile)

    with open('player_dict2015.pkl', 'r') as picklefile:
        player_dict = pickle.load(picklefile)

    with open('daily_results.pkl', 'r') as picklefile:
         daily_results = pickle.load(picklefile)

    with open('official_ids.pkl', 'r') as picklefile:
         official_ids = pickle.load(picklefile)

    with open('scores_dict.pkl', 'r') as picklefile:
         scores_dict = pickle.load(picklefile)
    
    box_data = get_box_scores(3)
    yesterday = get_boxscores(box_data, player_dict)
    pldfs = boxscore_dfs(yesterday, box_data)
    dfm = create_master(pldfs)
    official_ids = match_players_official(dfm, daily_projections, official_ids)
    scores_dict = get_fd_scores(dfm, scores_dict)
    yest_proj = daily_projections[yesterday_string]
    results = add_results(yest_proj, official_ids, scores_dict)
    daily_results = index_results(daily_projections, official_ids, scores_dict, daily_results)

    print 'daily_results updated @ ' +  datetime.now().strftime('%I:%M %p')
    with open('scores_dict.pkl', 'w') as picklefile:
        pickle.dump(scores_dict, picklefile)
    with open('official_ids.pkl', 'w') as picklefile:
        pickle.dump(official_ids, picklefile)
    with open('daily_results.pkl', 'w') as picklefile:
        pickle.dump(daily_results, picklefile)



if __name__ == '__main__':
    main()
