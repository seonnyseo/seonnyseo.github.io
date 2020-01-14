---
layout: post
title:  "Retrieving Spotify daily Top 50 charts data by using Spotify API and Python"
date:   2019-12-10 23:59:00 +0100
categories: Python Spotify API PANDAS BS4 data retrieving
---

#### Introduction

Spotify is a digital music service that gives users access to millions of songs and has a huge client base all around the world. Driven by the interest in music and the data, I choose to explore more from the data of Spotify. My goal is to build a “Top 50” music dataset with different categories of songs, including 62 lists of different countries and areas as well as global. The data is created by the information of songs and artists and also based on a daily playlist from Spotify users. I would like my data to show the popularity of certain songs, artists and categories.


You can check out python code for this work on this [Jupyter Notebook](https://github.com/seonnyseo/spotify_top_charts_retrieve/blob/master/spotify_top_charts_retrieve.ipynb).


#### Potential Users

Based on my dataset, there will be some potential users and applications for multiple purposes:

![Potential Users](https://i.imgur.com/KcajQ9U.jpg?1)

1. Record company - The dataset could help record companies decide which singers have unlimited potential and are worth investing in. And for the current singers in their company, this dataset can help the company to decide what type of songs need to be made to open the target foreign market. 
2. Events organizer - The dataset can be used as a reference for companies to know which songs their singers are going to choose on their world tour
3. Analyst - The dataset can be provided to analysts for data analysis
4. Listener - The dataset allows listeners to quickly know about the most popular type of music and singers in other countries

Spotify provides the data I use: information of songs, artists and albums, their available markets and popularity score, the number of followers of artists and so on. I use python and API to access the data, and the raw data I obtained is in json format. I clean the data to extract the information that is exactly I want and preprocess the data to finally build a easily readable data set.

#### How to retrieve data from Spotify

![Spotipy](https://i.imgur.com/MYpwORy.jpg)

Spotify allows users to access their API and also supports neat documentation. Although Spotify only provides Web API officially, they archive the list of 3rd party libraries for integrating with the Spotify Web API using several programming languages and platforms. And I am going to connect Spotify through Spotipy library.

[Spotify API Documentation](https://developer.spotify.com/documentation/web-api/)

[Spotipy Documentation](https://spotipy.readthedocs.io/en/2.6.1/)

#### What kind of information

![Features](https://i.imgur.com/bTDQcpM.jpg)

Here is the features that I collect for each songs.

24 Features

* continent 
* country rank
* song
* artist
* album
* release date
* song id
* artist id
* album id
* genre
* acousticness
* danceability
* energy
* instrumentalness
* liveness
* loudness
* speechiness
* valence
* followers
* artist popularity
* song popularity
* album popularity
* created date

You can check out more explanation on the features from [readme](https://github.com/seonnyseo/spotify_top_charts_retrieve).

The category of songs I get from the dataset is determined by the category of its artist. However, many artists release many different styles of songs and no single style label can represent the entire songs. So limitation occurs when I want to category our songs by its exact style, not the artist type. Spotify provides analyses of each song such as acousticness, danceability, energy, instrumentalness, liveness, loudness, speechiness and tempo. These data can help provide more information for analysts and record companies and I can use this information to help us obtain the artist type.


#### Programming

1. Collecting Name of Countries Serviced

[Where is Spotify Available](https://support.spotify.com/us/using_spotify/getting_started/full-list-of-territories-where-spotify-is-available/)

Top 50 Charts are updated daily by Spotify and my goal is to scrap the official charts as much as possible. I assume Spotify will add other countries charts in the future, so I think it is worth to make a way that covers it. My resolution is to grab a list of names of countries serviced from Spotify, and combine this with 'Top 50' keyword and search playlists that contain the combined word. Also, I have to filter one more time by chart creator ID, 'Spotify Official ID'. This work is in the code. 

2. Add a temporary function on Spotipy to gather data

![Temporarily Add](https://i.imgur.com/Om2qpDV.jpg)

Although Spotipy supports lots of Spotify official API endpoint functions, it does not reflect retrieving playlists' tracks information function yet (Dec 2019). This is an essential information for this work and I have to find a solution. So I analyze the Spotipy code first and decide to write a custom function and add it into Spotipy object by module 'types' temporarily. Then, I am still available to exploit the advantages of using Spotipy.

3. Generator to gather IDs easily and reduece usage of memory

![Generator](https://i.imgur.com/2ioB86n.jpg)

The code retrieves every data based on using each IDs. (It can be a chart, a singer, a song, and an album) Spotify API provides several information on a single request and it only requires to modify the endpoint of the request. However, there is a limited number of information that users can gather at once and it depends on a type of information. (Mostly 50, only 20 for albums) So I made a generator to return IDs based on type and limit numbers. I expect it would save a usage on memory.


#### Result

![Result Table](https://i.imgur.com/McQ1NXJ.jpg)

Final table looks like this. You can check out the whole result from the [jupyter notebook code](https://github.com/seonnyseo/spotify_top_charts_retrieve/blob/master/spotify_top_charts_retrieve.ipynb).



#### Future Works and Limitataions

1. Rank in data charts obtained from website is based on daily play counts in the country, which is not supported in API. I assume the rank as ascending order which is returned by API in its track information. (I have not seen an exception yet)

2. I could not figure out the time that Spotify updates the charts yet and the code cannot check the charts have been updated. (Future Works)

3. The code only creates a single daily charts csv file. Supporting aggretation ways such as reading a sheet and stack new data or creating new sheets everyday are 
cadndidates to improve this work.

