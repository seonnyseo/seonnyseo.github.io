---
layout: post
title:  "Airbnb New Users Booking"
date:   2020-05-06 23:59:00 +0100
categories: DataAnalysis Kaggle Python Pandas Xgboost
---

### 1. Introduction

While I am in quarantine due to the COVID, I would like to develop and sharpen my skills for data analysis and data science. XGBoost, which is a popular framework in the Kaggle, data science field and I have no experience before, was an interesting subject to me and I decided to jump in.

[Airbnb recruiting competition](https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings/overview) which was held in Kaggle about 4years ago looks suitable to me since many participants have shared their works using XGBoost, and the goal of this contest is interesting. Airbnb asked to predict new users' first destination with the users' demographic, summary statistics, and their web session records.

Simply I focused on using and learning the XGBoost framework at first, but my attention has been extended to analyze users' data and adopt funnel analysis(AARRR) on it. 

So I will talk about Explatory data anlaysis includes AARRR, and a predicting model below.

### 2. Data Analysis

Airbnb provides two types of datasets. Users' information and their web session data. Information data includes the date of the account created, date of the first booking, and name of destination country if they have a history, demographic info, signup info, and marketing channel info. This dataset is split to train and test sets. I have to predict users' first destination country in the test set.

Session data includes user ids, action, action types and detail, and secs elapsed. Airbnb does not give detailed information on it, so I assumed the meaning of unclear info such as secs elapsed that I considered it as a time elapsed after the last session.

I concentrated on building time-series graphs because of every row in both datasets hold time information and it provides meaningful information of users action while they were staying at the service.

#### 1. Pre-processing

While the tracing system of the service was able to record robust information for most columns, lack of demographic data like age and gender happened since it requires the participation of the users. The age column is tricky here.

![RAW Age graph](https://i.imgur.com/hli6TFU.png)

Fig. Age Raw Data Distribution

It contains information on users who are older than 1000 years old, which does not make sense. However, most of the unreasonable data are distributed around late 1900. I assumed that the system asked to type the year users were born or it did not prevent mistakes properly before, so I subtracted the age from 2015, which is the year the dataset was released. Then a new graph shows a reasonable distribution. (This graph only displays between age 18 - 100.)

[Fixed Age graph]

In the case of the session dataset, some instances do not include user ids. That means I cannot connect the sessions data to the users' information, so I dropped the instances.

Also splitting date information into the year, month, the day has been processed for convenience in data wrangling. 

[Added Columns]

#### 2. EDA

#### 3. AARRR

##### 1. Acquisition

Since the dataset only includes signed up users' information, it is not available to trace the visitors who only accessed and left the service. However, it provides the routes to the service such as affiliate channels, providers, and the first marketing the user interacted with before the signing up info. The first device used data is also provided.

[Marketing Channel]

From the marketing channel perspective, a direct connection is a top channel among marketing channels. This trend started in the middle of 2011 and the difference between direct and other channels has been increased constantly every year. And the second most popular channel is SEM-Branded until 2014/08. I can assume that most users already heard about Airbnb before they visit the service. (SEO 방문자가 많아지고 있음 -> paid 불필요 ?)

[First Device Year]

[Device by month]

Let's look at the type of the first device. The number of desktop users increased rapidly between 2011-2013 while mobile users incremented gradually. On the contrary, first visiting through mobile devices exploded from 2013. If I break down this figure by month, the number of iPhone users overtook desktop users.

[Mobile & Desktop]

A graph drawn by groups into mobile and desktop reveals more dynamical changes from early 2014 and even more visits were connected through mobile devices later. 

[Web browser]

The increase in mobile application users is expected to help increase the number of mobile users. The graph above displays changes in the 8 widely used web browsers users. Airbnb does not describe the Unknown browser users in their explanation, so I guessed it marks the users flew in the service through mobile applications. It seems reasonable to speculate that the increasing trend of the two graphs is similar.

##### 2. Activation

[Activation]

Since every user has activated their account, it is not meaningful to discuss activation here. Thus I drew the changes the number of created accounts by year and month. Airbnb has grown up every year and it exploded in 2014. 

[Activation Month]

This graph shows that the number of created accounts had increased every year again and the number of new users is the highest in the summer. This could be an effect of summer vacation. 


##### 3. Retention



##### 4. Referral

##### 5. Revenue


### 3. Modeling

### 4. Result

