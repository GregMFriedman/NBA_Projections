
# coding: utf-8

import os
import sys

from sqlalchemy import Column, ForeignKey, Integer, String, Date, Float

from sqlalchemy.ext.declarative import declarative_base

from sqlalchemy.orm import relationship

from sqlalchemy import create_engine

from sqlalchemy.orm import mapper, sessionmaker

from sqlalchemy import UniqueConstraint

from sqlalchemy.schema import MetaData

from datetime import date, time, datetime, timedelta

import numpy as np
import pandas as pd
import pickle
import requests
from requests_oauthlib import OAuth1
import cnfg
import time
from random import randint
import inspect


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
with open('player_dict.pkl', 'r') as picklefile:
     player_dict = pickle.load(picklefile)


date_string = date.today().strftime("%m-%d-%Y")
yesterday_string = (date.today() - timedelta(1)).strftime("%m-%d-%Y")
now = datetime.now()
latest = date_string if now.hour > 11 else yesterday_string
today = daily_projections[latest]

config_db = cnfg.load(".psql_config")
connection = config_db['connection']
Base = declarative_base()


class Player(Base):

    # __tablename__ = 'Player_Proj_' + '12-11-2015'
    __tablename__ = 'Player_Projections_' + date_string

    
    Index = Column(Integer, primary_key = True)
    Player = Column(String)
    Team = Column(String)
    Date = Column(Date)
    Time = Column(String)
    GameTime = Column(String)
    Salary = Column(Float)
    RG = Column(Float)
    FP = Column(Float)
    NF = Column(Float)
    RW = Column(Float)
    POS = Column(String)
    Depth = Column(Integer)
    G_Model = Column(Float)
    PPD = Column(Float)
    Team_Index = Column(Integer)


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


def main():

    date_string = date.today().strftime("%m-%d-%Y")
    yesterday_string = (date.today() - timedelta(1)).strftime("%m-%d-%Y")
    now = datetime.now()
    latest = date_string if now.hour > 11 else yesterday_string
    today = daily_projections[date_string]

    
    # team_dfs = make_df_dict(today, 'Team')
    # team_indexes = index_teams(team_dfs)

    # today['Team_Index'] = today['Team'].apply(lambda x: team_indexes[x])
    engine = create_engine(connection, echo=True)
    Base.metadata.create_all(engine)
    Base.metadata.bind = engine
    Session = sessionmaker(bind=engine)
    session = Session()
    today.sort('Player', inplace=True)
    today.reset_index(drop=True, inplace=True)
    today['G_Model'] = np.round(today['G_Model'], 2)
    todays_ids = {}
    for i in range(len(today)):
        name = today['Player'].ix[i]
        todays_ids[name] = i
    with open('todays_ids.pkl', 'w') as picklefile:
        pickle.dump(todays_ids, picklefile) 
    p = today.ix[0]
    player =  Player(Index=0, Player= p['Player'], Team= p['Team'], Date = p['Date'], Time = p['Time'], GameTime= p['Gametime'], Salary = p['Salary'], RG=p['RG'], FP= p['FP'], NF=p['NF'], RW= p['RW'], POS=p['POS'], Depth = p['Depth'], G_Model = p['G_Model'], PPD = p['PPD'], Team_Index = p['Team_Index'])
    session.add(player)
    session.commit()
    for i in range(1, len(today)):
        p = today.ix[i]
        player = Player(Index=i, Player= p['Player'], Team= p['Team'], Date = p['Date'], Time = p['Time'], GameTime= p['Gametime'], Salary = p['Salary'], RG=p['RG'], FP= p['FP'], NF =p['NF'], RW = p['RW'], POS=p['POS'], Depth = p['Depth'], G_Model = p['G_Model'], PPD = p['PPD'], Team_Index = p['Team_Index'])
        
        session.add(player)
        session.commit()

    with open('today.pkl', 'w') as picklefile:
        pickle.dump(today, picklefile) 


if __name__ == "__main__":
    main()
