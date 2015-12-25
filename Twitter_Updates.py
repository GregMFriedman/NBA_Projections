
# coding: utf-8

# In[1]:

import csv
import json
from pymongo import MongoClient
import requests
from requests_oauthlib import OAuth1
import cnfg
import time
from random import randint
import os
import inspect, os
import pandas as pd
import pickle
import cPickle as pickle
import tweepy
from difflib import SequenceMatcher
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from nltk.util import ngrams
from collections import defaultdict
from collections import OrderedDict
from operator import itemgetter
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from bs4 import BeautifulSoup
from datetime import date
from datetime import datetime
from datetime import timedelta
from pprint import pprint
import re


# ###Helper Methods & Variables

with open('player_dict.pkl', 'r') as picklefile:
     player_dict = pickle.load(picklefile)

with open('daily_projections.pkl', 'r') as picklefile:
     daily_projections = pickle.load(picklefile)

with open('injury_dict.pkl', 'r') as picklefile:
     injury_dict = pickle.load(picklefile)

with open('today.pkl', 'r') as picklefile:
     today = pickle.load(picklefile)

with open('depth_dict.pkl', 'r') as picklefile:
     depth_dict = pickle.load(picklefile)

with open('depth_ids.pkl', 'r') as picklefile:
     depth_ids = pickle.load(picklefile)

with open('player_to_team.pkl', 'r') as picklefile:
     player_to_team = pickle.load(picklefile)

with open('depth_to_teams.pkl', 'r') as picklefile:
     depth_to_teams = pickle.load(picklefile)


date_string = date.today().strftime("%m-%d-%Y")

players = player_dict.values()
player_updates = defaultdict(list)

config = cnfg.load("ds/metis/ds5_Greg/projects/05-kojak/.twitter_develop")
oauth = OAuth1(config["consumer_key"],
               config["consumer_secret"],
               config["access_token"],
               config["access_token_secret"])


auth = tweepy.OAuthHandler(config["consumer_key"],
                           config["consumer_secret"])
auth.set_access_token(config["access_token"],
                      config["access_token_secret"])

api = tweepy.API(auth)

emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""

regex_str = [
    emoticons_str,
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs

    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)', # other words
    r'(?:\S)' # anything else
]

tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)

days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
day_nums = [x for x in range(7)]
day_dict = dict(zip(days, day_nums))


# ###First Scrape Reliable Injury Update Website


def soup_url(url):
    site = requests.get(url)
    page = site.text
    soup = BeautifulSoup(page)
    return soup


def scrape_injury_report():
    injury_dict = {}
    soup = soup_url('http://www.donbest.com/nba/injuries/')
    rows1 = soup.find_all('td', class_="otherStatistics_table_alternateRow statistics_cellrightborder")
    rows2 = soup.find_all('td', class_="statistics_table_row statistics_cellrightborder")
    row_types = [rows1, rows2]
    for rows in row_types:
        for i in range(0, len(rows), 5):
            details = {}
            details['update_date'] = datetime.strptime(rows[i].text, '%m/%d/%y')
            details['position'] = rows[i+1].text
            details['injury'] = rows[i+3].text
            details['update'] = rows[i+4].text
            injury_dict[rows[i+2].text] = details

    return injury_dict

def next_weekday(d, weekday):
    days_ahead = weekday - d.weekday()
    if (days_ahead < 0): # Target day already happened this week
        days_ahead += 7
    return d + timedelta(days_ahead)

def parse_injury_report(injury_dictionary):
    for player, injury in injury_dictionary.iteritems():
        predictors = ['"?"', 'probable', 'doubtful', 'miss', 'out']
        words = injury['update'].split()
        for i in range(len(words)):
            for day in days:
                if fuzz.ratio(words[i], day) > 90:
                    injury['status_date'] = next_weekday(injury['update_date'], day_dict[day])
                    injury['status'] = words[i-1]
        for word in words:
            for predictor in predictors:
                if fuzz.ratio(word, predictor) > 95:
                    injury['status'] = word
    return injury_dictionary

# ###Methods for Twitter API calls & Preprocessing

def tokenize(s):
    return tokens_re.findall(s)
 
def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens

def structure_results(results):
    id_list=[tweet.id for tweet in results]
    data=pd.DataFrame(id_list,columns=['id'])
    
    data["text"]= [tweet.text.encode('utf-8') for tweet in results]
    data["datetime"]=[tweet.created_at for tweet in results]
    
    return data

def get_twitter_updates(n=50):
    updates = []
    for tweet in tweepy.Cursor(api.user_timeline, id="FantasyLabsNBA").items(n):
        updates.append(tweet)
    for tweet in tweepy.Cursor(api.user_timeline, id="Rotoworld_BK").items(n):
        updates.append(tweet)
    injuries=structure_results(updates)
    return injuries

def get_ngrams(token, n):
    pairs = []
    if len(token) <= 2:
        pair = (' ').join(token)
        pairs.append(pair)
    else:
        for i in range(len(token) - n+1):
            pair = (' ').join(token[i:i+n])
            pairs.append(pair)
    return pairs

def tokenize_tweets(df):
    tweet_tokens = []
    updates = df['text'].tolist()
    for update in updates:
        tweet_tokens.append(preprocess(update))
    return tweet_tokens

def get_tweet_bigrams(df, n=2):
    tweet_tokens = tokenize_tweets(df)
    doublets = {}
    for i in range(len(tweet_tokens)):
        doublets[i] = get_ngrams(tweet_tokens[i], n)
    return doublets

def get_stemmed_grams(tweet, n=2):
    words = tweet.split()
    stemmed = map(lambda x: stemmer.stem(x), words)
    return get_ngrams(stemmed, n)


#Twitter API Call Methods

def get_player_updates(df, player_updates=None):
    if not player_updates:
        player_updates = defaultdict(list)
    doublets = get_tweet_bigrams(df)
    for number, doublet_list in doublets.iteritems():
        for doublet in doublet_list:
            for player in players:
                if (fuzz.ratio(player, doublet.decode('ascii', 'ignore')) > 90):
                    player_updates[player].append((df['text'].ix[number], df['datetime'].ix[number]))
    return player_updates


def get_injury_details(update_dict):
    injury_details = []
    for player, updates in update_dict.iteritems():
        for update_tuple in updates:
            update = update_tuple[0]
            if '(' in update:
                start = update.index('(')
                end = update.index(')') + 1
                detail = update[start:end]
                if detail not in injury_details:
                    injury_details.append(detail)
    return injury_details


def filter_past(player_updates, update_dict):
    relevant = defaultdict(list)
    for player, updates in player_updates.iteritems():
        tweets = {}
        for update_tuple in updates:
            update = update_tuple[0]
            update_date = update_tuple[1]
            words = update.split()
            for i in range(len(words)):
                for day in days:
                    if (fuzz.ratio(words[i], day) > 90):
                        date = next_weekday(datetime.today(), day_dict[words[i].rstrip('.;,?')])
                        if date < datetime.today()+timedelta(3):
                            relevant[player].append(update_tuple)
                            tweets[update] = date
        update_dict[player] = tweets
    return relevant

def search_phrase(player, status_phrase, update, update_date, update_dict, injury_dict=injury_dict, stemmed=False):
    info = None
    if player in injury_dict:
        position = injury_dict[player]['position']
        injury = injury_dict[player]['injury']
    else:
        position = '?'
        injury = ''
    n = len(status_phrase.split())
    date = update_dict[player][update]
    words = update.split()
    phrases = get_ngrams(words, n)
    if '(' in update and not injury:
        start = update.index('(') + 1
        end = update.index(')')
        injury = update[start:end]
    if stemmed:
        for phrase in phrases:
            if fuzz.ratio(status_phrase, phrase.decode('ascii', 'ignore')) > 95:
                print status_phrase 
                print update 
                print date
                print update_date
                info = {'status_date': date, 'update': update, 'status': status_phrase, 
                        'update_date': update_date, 'position': position, 'injury': injury }
    if status_phrase in phrases:
        print status_phrase
        print update
        print date
        print update_date
        info = {'status_date': date, 'update': update, 'status': status_phrase, 
                        'update_date': update_date, 'position': position, 'injury': injury}
    if info:
        return info

def parse_twitter_updates(player_updates, injury_dict, update_dict=None, twitter_dict=None):
    if not update_dict:
        update_dict = {}
    if not twitter_dict:
        twitter_dict = injury_dict.copy()
    injury_details = get_injury_details(player_updates)
    status_singles = ['"?"', 'probable', 'doubtful', 'questionable', 'inactive', 'sidelined', 'starting'] 
    status_list = ['will start', 'remains the starter', 'on track to play', 'ruled out' , 'expect to play',                    '"?"', 'probable', 'doubtful', 'questionable', 'inactive', 'sidelined', 'starting',                    'unlikely to play'
                  ]
    relevant = filter_past(player_updates, update_dict)
    for player, updates in relevant.iteritems():
        for update_tuple in updates:
            update = update_tuple[0]
            update_date = update_tuple[1]
            words = update.split()
            for status_single in status_singles:
                status = (' ').join([player, 'listed', status_single])
                info = search_phrase(player, status, update, update_date, update_dict, injury_dict=injury_dict)
                status = (' ').join([player, 'officially', status_single])
                info = search_phrase(player, status, update, update_date, update_dict, injury_dict=injury_dict)
                for detail in injury_details:
                    status = (' ').join([player, detail, 'listed', status_single])
                    info = search_phrase(player, status, update, update_date, update_dict, injury_dict=injury_dict)
            for status_phrase in status_list:
                status = (' ').join([player, status_phrase])
                if (status_phrase == 'expect to play'):
                    stem = True
                    info = search_phrase(player, status, update, update_date, update_dict, stemmed=stem, injury_dict=injury_dict)
                else:
                    stem = False
                    info = search_phrase(player, status, update, update_date, update_dict, injury_dict=injury_dict)
                for detail in injury_details:
                    status = (' ').join([player, detail, status_phrase])
                    info = search_phrase(player, status, update, update_date, update_dict, stemmed=stem, injury_dict=injury_dict)
                    if info:
                        if player not in twitter_dict.keys():
                            twitter_dict[player] = info
                        else:
                            if (info['update_date'] > twitter_dict[player]['update_date']):
                                twitter_dict[player] = info
            if info:
                if player not in twitter_dict.keys():
                    twitter_dict[player] = info
                else:
                    if info['update_date'] > twitter_dict[player]['update_date']:
                        twitter_dict[player] = info
        if info:
            if player not in twitter_dict.keys():
                twitter_dict[player] = info
            else:
                if info['update_date'] > twitter_dict[player]['update_date']:
                    twitter_dict[player] = info
    return twitter_dict

def add_teams(twitter_dict, depth_ids, depth_to_teams):
    for player, info in twitter_dict.iteritems():
        try:
            name = depth_ids[player][0]
        except:
            candidates = map(lambda x: x[0], depth_ids.values())
            name = process.extractOne(player, candidates)[0]
        info['team'] = depth_to_teams[name]
    return twitter_dict

#Filter and react to player updates

def get_todays_updates(twitter_dict):
    todays_players = {}
    for player, update_dict in twitter_dict.iteritems():
        if 'status_date' in update_dict.keys():
            #print update_dict['status_date'].date(), datetime.today().date()
            if update_dict['status_date'].date() == datetime.today().date():
                todays_players[player] = update_dict
    return todays_players

def find_players(todays_update_dict, df):
    todays_players = []
    candidates = df['Player'].tolist()
    for player, update in todays_update_dict.iteritems():
        match = process.extractOne(player, candidates, score_cutoff=78)
        if match:
            todays_players.append((match, player))
    return todays_players

def get_info(todays_players, df, twitter_dict):
    headers = ['Team', 'POS', 'Depth', 'Status']
    info = {}
    for player in todays_players:
        name = player[0][0]
        team = df[df['Player'] == name]['Team'].tolist()[0]
        position = df[df['Player'] == name]['POS'].tolist()[0]
        depth = df[df['Player'] == name]['Depth'].tolist()[0]
        status = twitter_dict[player[1]]['status']
        info[name] = dict(zip(headers, [team, position, depth, status]))
    return info

def get_todays_teams(todays_info, df):
    team_dfs = {}
    teams = []
    for player, info in todays_info.iteritems():
        teams.append(info['Team'])
    for team in teams:
        team_dfs[team] = df[df['Team']==team]
    return team_dfs



def react_to_update(todays_info, df, depth_dict):
    injured_list = []
    for player, info in todays_info.iteritems():
        status = info['Status']
        if ('doubtful' in status) and player in (df['Player'].unique()):
            df.ix[df[df['Player'] == player].index, 'G_Model'] = 0
            injured_list.append(player)
            print player + ' not gonna help your team tonight!'
        if ('out' in status) and player in (df['Player'].unique()):
            df.ix[df[df['Player'] == player].index, 'G_Model'] = 0
            injured_list.append(player)
            print player + ' not gonna help your team tonight!'
        if ('miss' in status) and player in (df['Player'].unique()):
            df.ix[df[df['Player'] == player].index, 'G_Model'] = 0
            injured_list.append(player)
            print player + ' not gonna help your team tonight!'
        if ('will start' in status) and player in (df['Player'].unique()):
            if info['Depth'] == 0:
                print "Don't worry, "+player+" is still gonna start!"
            else: 
                team = info['Team']
                position = info['POS']
                downgraded = depth_dict[team]['depth'][position][0]
                print 'Sorry ' + downgraded
                info['Depth'] = 0
                df.ix[df[df['Player'] == player].index, 'Depth'] = 0
                print 'Looks like ' + player+ 'is starting now!'
                df.ix[df[df['Player'] == downgraded].index, 'Depth'] = 1
                if downgraded in todays_info.keys():
                    todays_info[downgraded]['Depth'] = 1
    return df, injured_list

def order_updates(twitter_dict):
    in_order = OrderedDict()
    for key in sorted(twitter_dict):
        in_order[key] = twitter_dict[key]
    return in_order


def move_columns(df, col_name, slot):
    cols = df.columns.tolist()
    index = cols.index(col_name)
    cols.insert(slot, cols.pop(index))
    return cols

def create_master(df_dict):
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
    return dfm

def make_df_dict(df, col_name):
    df_dict = {}
    for key in df[col_name].unique():
        df_dict[key] = df[df[col_name]==key].reset_index(drop=True)
    return df_dict



def main():

    with open('today.pkl', 'r') as picklefile:
        today = pickle.load(picklefile)

    with open('depth_dict.pkl', 'r') as picklefile:
     depth_dict = pickle.load(picklefile)

    injury_dict = scrape_injury_report()
    injury_updates = parse_injury_report(injury_dict)

    # ###Read 100 of the Latest Injury Update Tweets, Parse them & Organize them in a Dictionary

    injuries = get_twitter_updates()
    player_updates = get_player_updates(injuries)
    twitter_dict = parse_twitter_updates(player_updates, injury_dict)
    twitter_dict = add_teams(twitter_dict, depth_ids, depth_to_teams)
    twitter_dict_ordered = order_updates(twitter_dict)

    #Filter only relevant tweets (injuries, starters) about today's games.

    todays_updates = get_todays_updates(twitter_dict)
    todays_players = find_players(todays_updates, today)
    todays_info = get_info(todays_players, today, twitter_dict)
    todays_info_ordered = order_updates(todays_info)
    today, injured_list = react_to_update(todays_info, today, depth_dict)
    team_dfs = get_todays_teams(todays_info, today)

    #Pickle them

    with open('latest_updates.pkl', 'w') as picklefile:
        pickle.dump(twitter_dict, picklefile, protocol=2)

    with open('todays_info.pkl', 'w') as picklefile:
        pickle.dump(todays_info, picklefile, protocol=2)

    with open('today.pkl', 'w') as picklefile:
        pickle.dump(today, picklefile, protocol=2)

    with open('injured_list.pkl', 'w') as picklefile:
        pickle.dump(injured_list, picklefile, protocol=2)

    with open('injury_dict.pkl', 'w') as picklefile:
        pickle.dump(injury_dict, picklefile, protocol=2)

    with open('latest_updates_ordered.pkl', 'w') as picklefile:
        pickle.dump(twitter_dict_ordered, picklefile, protocol=2)

    with open('todays_info_ordered.pkl', 'w') as picklefile:
        pickle.dump(todays_info_ordered, picklefile, protocol=2)

    now = datetime.now().strftime('%I:%M %p')

    print injured_list
    print "Injury tweets current as of " + now


if __name__ == '__main__':
    main()
