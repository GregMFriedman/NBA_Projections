
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


# In[198]:

with open('team_city_dict.pkl', 'r') as picklefile:
     team_dict = pickle.load(picklefile)
with open('proj_list.pkl', 'r') as picklefile:
     proj_list = pickle.load(picklefile)
with open('dict_list.pkl', 'r') as picklefile:
     dict_list = pickle.load(picklefile)
with open('master_ids.pkl', 'r') as picklefile:
     master_ids = pickle.load(picklefile)
with open('daily_projections.pkl', 'r') as picklefile:
     daily_projections = pickle.load(picklefile)
with open('daily_results.pkl', 'r') as picklefile:
     daily_results = pickle.load(picklefile)


# In[199]:

date_string = date.today().strftime("%m-%d-%Y")


def soup_url(url):
    site = requests.get(url)
    page = site.text
    soup = BeautifulSoup(page)
    return soup


def scrape_numberfire(projection_dict=None):
    if not projection_dict:
        projection_dict = {}
    url = 'http://fantasy.usatoday.com/nba/rankings/'
    soup = soup_url(url)
    player_list = []
    projection_list = []
    team_list = []
    players = soup.find_all('td', class_='left')
    projections = soup.find_all('td', class_="rankings-source col-numberfire")
    info = soup.find('table', class_="sports-dynamic-table-scroll").find('tbody').find_all('td')
    index2 = None
    for player in players:
        player_list.append(player.a.text)
    for projection in projections:
        projection_list.append(projection.text)
    for i in range(0, len(info), 5):
        team_list.append(info[i].text)
    if 'S. Curry' in player_list:
        index1 = player_list.index('S. Curry')
        if 'S. Curry' in player_list[index1+1:]:
            index2 = player_list[index1+1:].index('S. Curry')
        if team_list[index1] == u'GSW':
            player_list.pop(index1)
            player_list.insert(index1, u'Stephen Curry')
        if index2 and team_list[index2] == u'GSW':
            player_list.pop(index2)
            player_list.insert(index2, u'Stephen Curry')
    player_projections = dict(zip(player_list,projection_list))
    projection_dict[date_string] = player_projections
    return projection_dict

def scrape_fantasy_pros(projection_dict=None):
    positions = ['PG', 'SG', 'SF', 'PF', 'C']
    if not projection_dict:
        projection_dict = {}
    player_projections = {}
    for pos in positions:
        url = 'http://www.fantasypros.com/nba/fanduel-cheatsheet.php?position='+pos
        soup = soup_url(url)
        info = soup.find_all('tr')[1:]
        for item in info:
            name = item.find('a').text
            if name:
                projection_text = item.find('td', class_='points').text.split()
                if len(projection_text) == 0:
                    projection = 0
                else:
                    projection = float(projection_text[0])
                gametime_text = item.find('td', class_='to').text.split()
                if len(gametime_text) == 0:
                    gametime = 0
                else:
                    gametime = gametime_text[1]
                player_projections[name] = projection, gametime, pos
            else:
                continue
        projection_dict[date_string] = player_projections
    return projection_dict

def scrape_roto_wire(projection_dict=None):
    if not projection_dict:
        projection_dict = {}
    player_projection = {}
    url = 'http://www.rotowire.com/daily/nba/optimizer.htm'
    soup = soup_url(url)
    players = soup.find_all('tr', class_='playerSet')
    for player in players:
        name_rev = player.a.text
        names = map(lambda x: x.rstrip(','), name_rev.split())
        name = reduce(lambda x, y: (' ').join([y,x]), names)
        minutes = int(player.find('td', class_="lineupopt-minutes").text)
        projection = float(player.find('td', class_="lineupopt-points").text)
        team = player.find('td', class_="lineupopt-team").text
        player_projection[name] = projection, minutes, team
    projection_dict[date_string] = player_projection
    return projection_dict


def scrape_rg(projection_dict=None):
    if not projection_dict:
        projection_dict = {}
    player_projections = {}
    url = 'https://rotogrinders.com/projected-stats/nba?site=fanduel'
    soup = soup_url(url)
    players = soup.find_all('td')
    for i in range(6,len(players), 6):
        name = players[i].a.text
        salary = players[i+3].text.replace('$', '').replace('K','').strip()
        if salary == 'N/A':
            salary = 0
        salary = float(salary)
        projection = float(players[i+4].text)
        player_projections[name] = projection, salary
    projection_dict[date_string] = player_projections
    return projection_dict


def scrape_dfsr(projection_dict=None):
    if not projection_dict:
        projection_dict = {}
    player_projections = {}
    url = 'http://www.dailyfantasysportsrankings.com/lineup-tools/fanduel-basketball-tool/'
    soup = soup_url(url)
    players = soup.find_all('tr')
    players = players[1:]
    for player in players:
        try:
            name = player.find(class_='playerName').text
        except:
            break
        try:
            projection = float(player.find(class_="projPts narrow-column").text)
        except:
            projection = 0
        try:
            minutes = int(player.find(class_='minutes narrow-column').text)
        except:
            minutes = 0
        player_projections[name] = projection, minutes
    projection_dict[date_string] = player_projections
    return projection_dict        


def rg_to_rw_dict(rg_dict, rw_dict, rg_to_rw=None):
    if rg_to_rw==None:
        rg_to_rw = {}
    for key in rg_dict[date_string].keys():
        if key not in rg_to_rw.keys():
            rg_to_rw[key] = process.extractOne(key, rw_dict[date_string].keys())
    return rg_to_rw


def rg_to_nf_dict(rg_dict, nf_dict, rg_to_nf=None):
    if rg_to_nf==None:
        rg_to_nf = {}
    for key in rg_dict[date_string].keys():
        if key not in rg_to_nf.keys():
            candidates = filter(lambda x: x[0] == key[0], nf_dict[date_string].keys())
            rg_to_nf[key] = process.extractOne(key, candidates)
    return rg_to_nf

def rg_to_dfsr_dict(rg_dict, dfsr_dict, rg_to_dfsr=None):
    if rg_to_dfsr==None:
        rg_to_dfsr = {}
    for key in rg_dict[date_string].keys():
        if key not in rg_to_dfsr.keys():
            candidates = filter(lambda x: x[0] == key[0], dfsr_dict[date_string].keys())
            rg_to_dfsr[key] = process.extractOne(key, candidates, score_cutoff=70)
    return rg_to_dfsr

def rg_to_fp_dict(rg_dict, fp_dict, rg_to_fp=None):
    if rg_to_fp==None:
        rg_to_fp = {}
    for key in rg_dict[date_string].keys():
        if key not in rg_to_fp.keys():
            candidates = filter(lambda x: x[0]==key[0], fp_dict[date_string].keys())
            rg_to_fp[key] = process.extractOne(key, candidates)
    return rg_to_fp

def create_master_dict(rg_to_rw, dict_list, rg_to_fp, master_ids=None):
    if not master_ids:
        master_ids = {}
    day_master = {}
    abbs = ['rg', 'rw', 'fp', 'nf']
    for player in rg_to_rw.iterkeys():
        if player not in master_ids:
            names = [(player, 100)]
            for d in dict_list:
                names.append(d[player])
            name_dict = dict(zip(abbs, names))
            master_ids[player] = name_dict
        else:
            master_ids[player]['fp'] = rg_to_fp[player]
    return master_ids


def aggregate_projections(master_ids, proj_list, rg_dict, player_to_team):
    rows = []
    rg, fp, nf, rw = map(lambda x: x[date_string], proj_list)
    for player, ids in rg_dict[date_string].iteritems():
        if player in master_ids.keys():
            ids = master_ids[player]
            if (ids['fp'] and ids['nf'] and ids['rw']):
                p_fp, p_nf, p_rw = master_ids[player]['fp'][0], master_ids[player]['nf'][0], master_ids[player]['rw'][0]
                time = datetime.now().strftime('%I:%M %p')
                if player not in rg:
                    print player, 'rg'
                    continue
                if p_fp not in fp:
                    try:
                        fp[p_fp] = 0, '', pos_dict[ids['depth']]
                    except:
                        print p_fp, 'fp'
                        continue
                if p_nf not in nf:
                    try:
                        nf[p_nf] = 0
                        print p_nf, 'nf proj set to 0'
                    except:
                        continue
                if p_rw not in rw:
                    try:
                        rw[p_rw] = 0, 0, player_to_team[player]
                    except:
                        print p_rw, 'rw'
                        continue
                row = [player, rw[p_rw][2], date_string, time, fp[p_fp][1], rg[player][1], rg[player][0], fp[p_fp][0], nf[p_nf], rw[p_rw][0]]
                rows.append(row)
            else:
                continue
    df = pd.DataFrame(rows, columns=['Player', 'Team', 'Date', 'Time', 'Gametime', 'Salary', 'RG', 'FP', 'NF', 'RW'])
    return df


def index_data(master_ids, proj_list, rg_dict, player_to_team, daily_projections=None):
    if not daily_projections:
        daily_projections = {}
    daily_projections[date_string] = aggregate_projections(master_ids, proj_list, rg_dict, player_to_team)
    return daily_projections

def make_depth_dict():
    positions = ['PG', 'SG', 'SF', 'PF', 'C']
    url = 'http://basketball.realgm.com/nba/depth-charts'
    soup = soup_url(url)
    keys = []
    options = soup.find_all(class_='ddl')
    teams = options[1].find_all('option')[1:]
    i = 0
    for team in teams:
        temp = process.extractOne(team.text, team_dict.keys())
        key = team_dict[temp[0]]
        keys.append(key)
    data = soup.find_all('table', class_="basketball")
    depth_dict = {}
    for datum in data:
        starters = datum.find_all(class_='depth_starters')
        starter_links = starters[0].find_all('a')
        starting5 = []
        roster = defaultdict(str)
        for starter in starter_links:
            starting5.append([starter.text])
        depth = dict(zip(positions, starting5))
        starting5 = np.ravel(starting5)
        starting_lineup = dict(zip(starting5, positions))
        subs = datum.find_all(class_='depth_rotation')
        rotation = defaultdict(str)
        for sub in subs:
            for pos in positions:
                links = sub.find_all('td', {'data-th': pos})
                for link in links:
                    if link.find('a'):
                        depth[pos].append(link.find('a').text)
                        rotation[link.find('a').text] = pos
                        
        scrubs = datum.find_all(class_="depth_limpt")
        scrub_dict = defaultdict(str)
        for scrub in scrubs:
            for pos in positions:
                    links = scrub.find_all('td', {'data-th': pos})
                    for link in links:
                        if link.find('a'):
                            depth[pos].append(link.find('a').text)
                            scrub_dict[link.find('a').text] = pos
        for player, position in starting_lineup.iteritems():
            roster[player] = position
        for player, position in rotation.iteritems():
            roster[player] = position
        for player, position in scrub_dict.iteritems():
            roster[player] = position
        depth_dict[keys[i]] = {'roster': roster,  'depth': depth, 'starters': starting_lineup, 'rotation': rotation, 'scrubs': scrub_dict}
        i += 1
    depth_dict['PHO'] = depth_dict['PHX']
    return depth_dict


def match_players_depth(master_ids, rw_dict, depth_dict, ids=None):
    if not ids:
        ids = {}
    for player, dicts in master_ids.iteritems():
        if player not in ids:
            candidates = []
            name = dicts['rw'][0]
            if name in rw_dict[date_string].keys():
                team = rw_dict[date_string][name][2]
                for pos, people in depth_dict[team]['depth'].iteritems():
                    candidates += people
                if team in depth_dict:
                    ids[player] = process.extractOne(player, candidates)
            else:
                continue
    return ids

def make_depth_ids(df_dict, depth_dict):
    team_depth_ids = {}
    depth_ids = {}
    for team, df in df_dict.iteritems():
        team_depth_ids[team] = match_players_depth(df, depth_dict, team)
    for team, id_dicts in team_depth_ids.iteritems():
        for player, depth_name in id_dicts.iteritems():
            if player not in depth_ids:
                depth_ids[player] = depth_name
            else:
                if depth_name[1] > depth_ids[player][1]:
                    depth_ids[player] = depth_name
    return depth_ids

def make_team_depth_ids(df_dict, depth_dict):
    team_depth_ids = {}
    for team, df in df_dict.iteritems():
        team_depth_ids[team] = match_players_depth(df, depth_dict, team)
    return team_depth_ids

def make_position_dict(depth_dict, depth_ids):
    position_dict = {}
    for team, depths in depth_dict.iteritems():
        for player, pos in depths['starters'].iteritems():
            if player in depth_ids.keys():
                name = depth_ids[player][0]
                position_dict[name] = pos
            else:
                name = player
                position_dict[name] = pos
        for player, pos in depths['rotation'].iteritems():
            if player in depth_ids.keys():
                name = depth_ids[player][0]
                position_dict[name] = pos
            else:
                name = player
                position_dict[name] = pos
        for player, pos in depths['scrubs'].iteritems():
            if player in depth_ids.keys():
                name = depth_ids[player][0]
                position_dict[name] = pos
            else:
                name = player
                position_dict[name] = pos
    return position_dict

def make_reverse_pos_dict(depth_ids):
    rev_dict = {}
    for name, alias in depth_ids.iteritems():
        rev_dict[alias[0]] = name
    return rev_dict

def rg_to_pos_dict(df, pos_dict, rg_to_pos=None):
    if rg_to_pos==None:
        rg_to_pos = {}
    daily_dict = {}
    players = df['Player'].tolist()
    for key in players:
        candidates = filter(lambda x: x[0] == key[0], pos_dict.keys())
        daily_dict[key] = process.extractOne(key, candidates)[0]
    rg_to_pos[date_string] = daily_dict
    return rg_to_pos

def get_position_dict(df, depth_ids, player_to_team, depth_dict, pos_dict=None):
    if not pos_dict:
        pos_dict = {}
    for player in df['Player'].unique():
        name = depth_ids[player][0]
        team = player_to_team[player]
        pos_dict[player] = depth_dict[team]['roster'][name]
    return pos_dict

def get_position(df, pos_dict):
    df['POS'] = df['Player'].apply(lambda x: pos_dict[x])
    return df

def get_team_dict(df, team_dict, master_ids, rw_dict, depth_ids, player_to_team=None):
    if not player_to_team:
        player_to_team = {}
    for player in df['Player'].unique():
        if player not in player_to_team:
            name = master_ids[player]['rw'][0]
            try:
                player_to_team[player] = rw_dict[date_string][name][2]
            except:
                try:
                    player_to_team[player] = rw_dict[yesterday_string][name][2]
                except:
                    for team, dicts in depth_dict.iteritems():
                        if depth_ids[player][0] in dicts['roster'].keys():
                            player_to_team[player] = team
    return player_to_team

def get_team(df, player_to_team):
    df['Team'] = df['Player'].apply(lambda x: player_to_team[x])
    return df

def get_depth_by_player(df, depth_ids, depth_dict, depth_by_date=None):
    if not depth_by_date:
        depth_by_date = {}
    depths = {}
    for player in df['Player'].unique():
        try:
            name = depth_ids[player][0]
            position = df.ix[df['Player']==player, 'POS'].tolist()[0]
            team = df.ix[df['Player']==player, 'Team'].tolist()[0]
            depth = depth_dict[team]['depth'][position].index(name)
            depths[player] = depth
        except:
            depths[player] = 2
    depth_by_date[date_string] = depths
    return depth_by_date


def add_depth(df, depth_by_date):
    df['Depth'] = df['Player'].apply(lambda x: depth_by_date[date_string][x])
    return df

def make_df_dict(df, column):
    df_dict = {}
    i = 0
    for element in df[column].unique():
        df_dict[element] = df[df[column] == element].reset_index(drop=True)
        df_dict[element]['Team_Index'] = np.repeat(i, len(df_dict[element]))
        i += 1
    return df_dict

def index_teams(df_dict):
    team_indexes = {}
    for team, df in df_dict.iteritems():
        team_indexes[team] = df['Team_Index'].ix[0]
    return team_indexes

def prep_df(df):
    df['RG'] = df['RG'].astype(float)
    df['FP'] = df['FP'].astype(float)
    df['NF'] = df['NF'].astype(float)
    df['RW'] = df['RW'].astype(float)
    df['Date'] = pd.to_datetime(df['Date'])
    df['G_Model'] = df.apply(lambda x: np.round((x['RG'] + x['FP'] + x['NF'] + x['RW'])/4.0, 2), axis=1)
    df['PPD'] = df.apply(lambda x: 0 if x['Salary']==0 else np.round(x['G_Model']/x['Salary'],2), axis=1)
    return df

def eliminate_zeros(df):
    df['zeros'] = df.apply(lambda x: 1 if ((x['NF']==0) or (x['RG']==0) or (x['RW']==0)) else 0, axis=1 )
    df = df[df['zeros'] ==0]
    df.drop('zeros', 1, inplace=True)
    return df

def create_master(df_dict):
    index = df_dict.keys()
    first = df_dict[index[0]]
    dfm = first
    for key in index[1:]:
        df = df_dict[key]
        dfm = pd.concat([dfm, df])
    dfm.drop_duplicates(inplace=True)
    dfm.reset_index(drop=True, inplace=True)
    return dfm

def unify_dfs(df_dict):
    unified_dict = {}
    common = list(set(df_dict['12-08-2015'].columns.tolist()) & set(df_dict['12-11-2015'].columns.tolist()))
    print common
    for date, df in df_dict.iteritems():
        columns = df.columns.tolist()
        for column in columns:
            if column not in common:
                df = df.drop(column, axis=1)
                unified_dict[date] = df
    return unified_dict

def make_g_model(daily_results, daily_projections):
    daily_results_common = unify_dfs(daily_results)
    dfm = create_master(daily_results_common)
    dfm['NF'] = dfm['NF'].astype(float)
    dfm = eliminate_zeros(dfm)
    X = pd.get_dummies(dfm[['Salary', 'RG', 'NF', 'RW', 'POS', 'Depth']])
    X = pd.concat([X.drop('Depth',1), pd.get_dummies(X['Depth'])], 1)
    if 'POS_' in X.columns:
        X.drop('POS_', axis=1, inplace=True)
    #if 3 in X.columns:
        #X.drop(3, axis=1, inplace=True)
    print X.columns
    X = sm.add_constant(X)
    info = dfm[['Player', 'Date', 'Time']]
    y = dfm['FD']
    model=sm.OLS(y, X).fit()
    today = daily_projections[date_string]
    X = pd.get_dummies(today[['Salary', 'RG', 'NF', 'RW', 'POS', 'Depth']])
    X = pd.concat([X.drop('Depth',1), pd.get_dummies(X['Depth'])], 1)
    print X.columns
    if 'POS_' in X.columns:
        X.drop('POS_', axis=1, inplace=True)
    X = sm.add_constant(X)
    g_model = model.predict(X)
    return g_model  


def main():

    with open('team_city_dict.pkl', 'r') as picklefile:
     team_dict = pickle.load(picklefile)
    with open('proj_list.pkl', 'r') as picklefile:
         proj_list = pickle.load(picklefile)
    with open('dict_list.pkl', 'r') as picklefile:
         dict_list = pickle.load(picklefile)
    with open('master_ids.pkl', 'r') as picklefile:
         master_ids = pickle.load(picklefile)
    with open('daily_projections.pkl', 'r') as picklefile:
         daily_projections = pickle.load(picklefile)
    with open('player_to_team.pkl', 'r') as picklefile:
         player_to_team = pickle.load(picklefile)
    with open('depth_by_date.pkl', 'r') as picklefile:
         depth_by_date = pickle.load(picklefile)
    with open('depth_ids.pkl', 'r') as picklefile:
         depth_ids = pickle.load(picklefile)

    rg_dict, fp_dict, nf_dict, rw_dict = proj_list
    rg_dict, fp_dict, nf_dict, rw_dict = scrape_rg(rg_dict), scrape_fantasy_pros(fp_dict), scrape_numberfire(nf_dict), scrape_roto_wire(rw_dict)
    proj_list = [rg_dict, fp_dict, nf_dict, rw_dict]

    rg_to_rw, rg_to_fp, rg_to_nf = dict_list
    rg_to_rw, rg_to_fp, rg_to_nf = rg_to_rw_dict(rg_dict, rw_dict, rg_to_rw), rg_to_fp_dict(rg_dict, fp_dict, rg_to_fp), rg_to_nf_dict(rg_dict, nf_dict, rg_to_nf)
    dict_list = [rg_to_rw, rg_to_fp, rg_to_nf]

    master_ids = create_master_dict(rg_to_rw, dict_list, rg_to_fp, master_ids)
    daily_projections = index_data(master_ids, proj_list, rg_dict, player_to_team, daily_projections)

    today = daily_projections[date_string]
    depth_dict = make_depth_dict()
    depth_ids = match_players_depth(master_ids, rw_dict, depth_dict, depth_ids)
    player_to_team = get_team_dict(today, team_dict, master_ids, rw_dict, depth_ids, player_to_team)
    today = get_team(today, player_to_team)
    pos_dict = get_position_dict(today, depth_ids, player_to_team, depth_dict)
    today = get_position(today, pos_dict)
    depth_by_date = get_depth_by_player(today, depth_ids, depth_dict, depth_by_date)
    today = add_depth(today, depth_by_date)
    today = prep_df(today)
    g_model = make_g_model(daily_results, daily_projections)
    today['G_Model'] = g_model
    team_dfs = make_df_dict(today, 'Team')
    team_indexes = index_teams(team_dfs)
    today['Team_Index'] = today['Team'].apply(lambda x: team_indexes[x])
    daily_projections[date_string] = today

    with open('proj_list.pkl', 'w') as picklefile:
        pickle.dump(proj_list, picklefile)
    with open('dict_list.pkl', 'w') as picklefile:
        pickle.dump(dict_list, picklefile)
    with open('master_ids.pkl', 'w') as picklefile:
        pickle.dump(master_ids, picklefile)
    with open('daily_projections.pkl', 'w') as picklefile:
        pickle.dump(daily_projections, picklefile)
    with open('depth_by_date.pkl', 'w') as picklefile:
        pickle.dump(depth_by_date, picklefile)
    with open('depth_ids.pkl', 'w') as picklefile:
        pickle.dump(depth_ids, picklefile)
    with open('player_to_team.pkl', 'w') as picklefile:
        pickle.dump(player_to_team, picklefile)
    with open('today.pkl', 'w') as picklefile:
        pickle.dump(today, picklefile)

    print 'Projections Updated @ ' +  datetime.now().strftime('%I:%M %p')

if __name__ == '__main__':
    main()
