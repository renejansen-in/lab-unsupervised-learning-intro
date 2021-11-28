import pandas as pd

song_lake = pd.read_csv('song_lake_clustered.csv')
song_lake = song_lake.drop(['Unnamed: 0'],axis=1)     #;song_lake

hot100 = pd.read_csv('tmp_hot100.csv')                #;hot100

features = pd.read_csv('features_scaled.csv')
features = features.drop(['Unnamed: 0'],axis=1)       #;features

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
secrets_file = open("secrets.txt","r")
string = secrets_file.read()
secrets_dict={}
for line in string.split('\n'):
    if len(line) > 0:
        secrets_dict[line.split(':')[0]]=line.split(':')[1]
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=secrets_dict['cid'],
                                                           client_secret=secrets_dict['csecret']))

import pickle
scaler = pickle.load(open('standardscaler.pkl','rb')) # load the scaling model
kmeans = pickle.load(open('kmeans.pkl','rb'))         # load the clustering model

def features(track, artist):
    track_id = sp.search(q='artist:' + artist + ' track:' + track, type='track')
    uri = track_id["tracks"]["items"][0]['id']
    features = sp.audio_features(uri)
    return features


import random

print('Song recommender | version 1.0')
print('==============================')
song_in = input("What's your most favorite song?  ").title()
artist_in = input('Artist or group: ').title()

collected = sp.search(q=song_in, type='track')
found = collected["tracks"]["total"] 

while found == 0:
    print("Can't find the song, try another")
    song_in = input("Enter song: ").title()
    artist_in = input("Enter artist or group: ").title()
    collected = sp.search(q=song_in, type='track')
    found = collected["tracks"]["total"]

if song_in in list(hot100['song']):
    pick_one_hot = random.choice(list(hot100['song']))
    while pick_one_hot == song_in:
            pick_one_hot = random.choice(list(hot100['song']))
            print('We recommend:', pick_one_hot)
else:
    feature = features(song_in, artist_in)
    column = list(feature[0].keys())
    values = [list(feature[0].values())]
    song_in_df = pd.DataFrame(data = feature, columns = column)
    song_in_df = song_in_df.drop(['type','id','uri','track_href','analysis_url','time_signature'],axis=1)
    song_in_scaled = scaler.transform(song_in_df)
    new_clust = kmeans.predict(song_in_scaled)
    df_clust = song_lake[song_lake['clust'] == list(new_clust)[0]]

    pick_one = random.choice(list(df_clust['trackid']))
    #url = df_clust['url'][df_clust['trackid'] == pick_one].values[0]

    track_info = sp.track(pick_one)
    print('Our recommendation from the same style', new_clust)
    print('Song name: ',track_info['name'])
    print('Artist name:',track_info['artists'][0]['name'])
    print('Link to the song',track_info['external_urls']['spotify'])
    print('Listen 30 seconds:',track_info['preview_url'])
