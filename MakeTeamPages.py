
# coding: utf-8

# In[1]:

from flask import Flask, render_template, request, redirect, url_for, jsonify
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pickle
import cPickle as pickle
import numpy as np
import pandas as pd
import cnfg
from sqlalchemy import Column, ForeignKey, Integer, String
import requests
from requests_oauthlib import OAuth1
import cnfg
import time
from random import randint
import os
import inspect
import decimal, datetime
from datetime import date
import json

from sqlalchemy.ext.declarative import declarative_base

from sqlalchemy.orm import relationship

from sqlalchemy import create_engine

from Flask_Classes import Player




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
with open('today.pkl', 'r') as picklefile:
     today = pickle.load(picklefile)


app = Flask(__name__)
config_db = cnfg.load(".psql_config")
connection = config_db['connection']
Base = declarative_base()
date_string = date.today().strftime("%m-%d-%Y")

engine = create_engine(connection)
Base.metadata.bind = engine

DBSession = sessionmaker(bind=engine)
session = DBSession()


teams = today['Team'].unique().tolist()
positions = today['POS'].unique().tolist()
headers = today.columns.tolist()

# def alchemyencoder(obj):
#     """JSON encoder function for SQLAlchemy special classes."""
#     if isinstance(obj, datetime.date):
#         return obj.isoformat()
#     elif isinstance(obj, decimal.Decimal):
#         return float(obj)

@app.route('/players/', methods=['GET', 'POST'])
def graphAll():
    players = session.query(Player).all()
    if request.method == 'POST':
        print 'Post Request Received!'
        with open('injured_list.pkl', 'r') as picklefile:
            injured_list = pickle.load(picklefile)
        with open('todays_info.pkl', 'r') as picklefile:
            todays_info = pickle.load(picklefile)
        print injured_list
        indexes = []
        if injured_list:
            for player in injured_list:
                update_player = session.query(Player).filter_by(Player=player).one()
                print player, update_player.G_Model
                update_player.G_Model = 0
                indexes.append(update_player.Index)
                print player, update_player.G_Model
                session.add(update_player)
                session.commit()
            
            with open('indexes.pkl', 'w') as picklefile:
                pickle.dump(indexes, picklefile)
        else:
            print 'Everybody seems healthy!'

    players = session.query(Player).order_by(Player.Player).all()

    return render_template('players.html', players=players)

@app.route('/players/updates/', methods=['GET', 'POST'])
def updateProjections():
    with open('injured_list.pkl', 'r') as picklefile:
        injured_list = pickle.load(picklefile)
    if injured_list:
        for player in injured_list:
            update_player = session.query(Player).filter_by(Player=player).one()
            print player, update_player.G_Model
            update_player.G_Model = 0
            print player, update_player.G_Model
            session.add(update_player)
            session.commit()
    players = session.query(Player).order_by(Player.Player).all()


    return render_template('players.html', players=players)





@app.route('/')
@app.route('/position/')
def showPlayers():
    return render_template('positions.html', positions=positions)


@app.route('/position/<position>/')
def playersByPosition(position):
    players = session.query(Player).filter_by(POS=position).order_by(Player.Player).all() 
    return render_template('index.html', players=players, position=position)


@app.route('/player/<int:player_id>/')
def makeProjections(player_id):
    player = session.query(Player).filter_by(Index=player_id).one()
    team = player.Team
    teammates = session.query(Player).filter_by(Team=team).order_by(Player.Salary.desc()).all()
    
    # json_data = dict(zip(headers,today.ix[player_id].tolist()))
    # player_data = jsonify(json_data)
    
    # teammate_df = today[today['Team'] == team]
    # json_data = teammate_df.T.to_dict()
    # team_data = jsonify(json_data)

    return render_template('d3bargraph.html', player=player, teammates=teammates)

@app.route('/projection/')
def showProjections():
    render_template('projections_d3.html')


@app.route('/players/twitter/')
def updateTwitter():
    with open('latest_updates_ordered.pkl', 'r') as picklefile:
        latest_updates = pickle.load(picklefile)
    update_list = []
    for key, value in latest_updates.iteritems():
        latest_updates[key]['player'] = key
        # if 'status_date' in latest_updates[key].keys():
        #     latest_updates[key]['status_date'] = latest_updates[key]['status_date'].strftime("%m-%d-%Y")
        # if 'update_date' in latest_updates[key].keys():
        #     latest_updates[key]['update_date'] = latest_updates[key]['update_date'].strftime("%m-%d-%Y")
        # update_list.append(latest_updates[key])
    #return jsonify(latest_updates)
    return render_template('twitter.html', result=latest_updates)

@app.route('/players/updated/', methods=['POST', 'GET'])
def getUpdates():
    with open('todays_info_ordered.pkl', 'r') as picklefile:
        todays_info = pickle.load(picklefile)

    for key, value in todays_info.iteritems():
        todays_info[key]['player'] = key
        if value['Depth'] == 0:
            value['Rotation'] = 'Starter'
        elif value['Depth'] == 1:
            value['Rotation'] = 'Rotation'
        else:
            value['Rotation'] = 'Scrub'
    #return jsonify(todays_info)
    return render_template('twitter2.html', result=todays_info)

@app.route('/players/injured/', methods=['POST', 'GET'])
def getInjuries():
    with open('injured_list.pkl', 'r') as picklefile:
        injured_list = pickle.load(picklefile)
    injured = json.dumps(injured_list)
    print injured
    return injured

@app.route('/roster/<team>')
def teamPage(team):
    players = session.query(Player).filter_by(Team=team).order_by(Player.Player).all() 
    return render_template('team.html', players=players, updated=None)


@app.route('/players/json/')
def json_stats():
    with open('injured_list.pkl', 'r') as picklefile:
        injured_list = pickle.load(picklefile)
    with open('today.pkl', 'r') as picklefile:
        today = pickle.load(picklefile)
    with open('indexes.pkl', 'r') as picklefile:
        indexes = pickle.load(picklefile)
    with open('todays_ids.pkl', 'r') as picklefile:
        todays_ids = pickle.load(picklefile)

    today.rename(columns={'Player':'playerName', 'Gametime': 'gameTime', 'Salary': 'salary',
                     'POS': 'position', 'Depth': 'depth', 'RG': 'rg', 'FP': 'fp', 
                     'NF': 'nf', 'RW': 'rw', 'G_Model': 'g_model', 'PPD': 'ppd',
                     'Team_Index': 't_index'}, inplace=True)
    stats_json = today.sort('playerName').reset_index(drop=True).T.to_dict()
    for index in indexes:
        stats_json[index]['g_model'] = 0
    for player, value in stats_json.iteritems():
        name = value['playerName']
        value['playerID'] = todays_ids[name]
        value['Date'] = value['Date'].strftime("%m-%d-%Y")
    return jsonify(stats_json)

@app.route('/players/injuries/')
def example():
    
    players = session.query(Player).order_by(Player.Player).all() 

    # use special handler for dates and decimals
    return json.dumps([dict(p) for p in players], default=alchemyencoder)


if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=5000)




    # for player in injured_list:
    #     update_player = session.query(Player).filter_by(Player=player).one()
    #     print player, update_player.G_Model
    #     update_player.G_Model = 0
    #     print player, update_player.G_Model
    #     session.add(update_player)
    #     session.commit()
    # # if len(injured_list) == 1: 
    # #     updated_player = injured_list[0]
    # #     return redirect(url_for('makeProjections', player_id=updated_player.Index))
