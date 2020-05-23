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

![RAW Age graph](https://i.imgur.com/IZ1SosX.png)

Fig. Age Raw Data Distribution

It contains information on users who are older than 1000 years old, which does not make sense. However, most of the unreasonable data are distributed around late 1900. I assumed that the system asked to type the year users were born or it did not prevent mistakes properly before, so I subtracted the age from 2015, which is the year the dataset was released. Then a new graph shows a reasonable distribution. (This graph only displays between age 18 - 100.)

![Corrected Age graph](https://i.imgur.com/BKejBd3.png)


In the case of the session dataset, some instances do not include user ids. That means I cannot connect the sessions data to the users' information, so I dropped the instances.

Also splitting date information into the year, month, the day has been processed for convenience in data wrangling. 

#### 2. EDA

#### 3. AARRR

##### 1. Acquisition

Since the dataset only includes signed up users' information, it is not available to trace the visitors who only accessed and left the service. However, it provides the routes to the service such as affiliate channels, providers, and the first marketing the user interacted with before the signing up info. The first device used data is also provided.

![Marketing Channel](https://i.imgur.com/6i6UJrZ.png)

From the marketing channel perspective, a direct connection is a top channel among marketing channels. This trend started in the middle of 2011 and the difference between direct and other channels has been increased constantly every year. And the second most popular channel is SEM-Branded until 2014/08. I can assume that most users already heard about Airbnb before they visit the service. 

![First Device Year](https://i.imgur.com/DCuC0uD.png)

![Device by month](https://i.imgur.com/dpOUx6Z.png)

Let's look at the type of the first device. The number of desktop users increased rapidly between 2011-2013 while mobile users incremented gradually. On the contrary, first visiting through mobile devices exploded from 2013. If I break down this figure by month, the number of iPhone users overtook desktop users.

![Mobile & Desktop](https://i.imgur.com/Kfs3oUu.png)

A graph drawn by groups into mobile and desktop reveals more dynamical changes from early 2014 and even more visits were connected through mobile devices later. 

![Web browser](https://i.imgur.com/6W2wspZ.png)

The increase in mobile application users is expected to help increase the number of mobile users. The graph above displays changes in the 8 widely used web browsers users. Airbnb does not describe the Unknown browser users in their explanation, so I guessed it marks the users flew in the service through mobile applications. It seems reasonable to speculate that the increasing trend of the two graphs is similar.

##### 2. Activation

![Activation](https://i.imgur.com/uVVhRl0.png)

Since every user has activated their account, it is not meaningful to discuss activation here. Thus I drew the changes the number of created accounts by year and month. Airbnb has grown up every year and it exploded in 2014. 

![Activation Month](https://i.imgur.com/jH1wgiL.png)

This graph shows that the number of created accounts had increased every year again and the number of new users is the highest in the summer. This could be an effect of summer vacation. 


##### 3. Retention

![Session](https://i.imgur.com/RyDwWoq.png)


![Sessions Expanded](https://i.imgur.com/bCh4sra.png)

For analyzing users' retention, I manipulated the session dataset. Originally it only contains (id, action, action_type, action_detail, device_type, secs_elapsed) columns. It traces the user's activity while staying in the service, but it does not provide timestamps for the actions. So I accumulated elapsed seconds by id, calculated days, and connected it with accounts created day. So the date of the activity can be estimated at an approximate value.

![Retention Graph](https://i.imgur.com/iCScIRi.png)

Retention calculated the difference between the day the user created the account and the date he or she last accessed it. According to the graph, Top 20% of users retention day is longer than 27 days, while bottom 20% users had not come back after 2.5 days. 

| Top 20%     |            | Bottom 20%  |          |
|-------------|------------|-------------|----------|
| count       | 27809      | count       | 26342    |
| mean        | 177.942285 | mean        | 8.721547 |
| std         | 160.384814 | std         | 7.953145 |
| min         | 3          | min         | 0        |
| 25%         | 80         | 25%         | 3        |
| 50%         | 134        | 50%         | 6        |
| 75%         | 222        | 75%         | 12       |
| max         | 2715       | max         | 134      |

The number of Top 20% of users is 27809, and the bottom 20% is 26342. As can be easily predicted, the difference in teh number of sessions between Top and Bottom is significant. Top 20% of users are tend to click more pages and try more actions.

| Top 20%               |          | Bottom 20%            |          |
|-----------------------|----------|-----------------------|----------|
| Action                | Ratio    | Action                | Ratio    |
|-----------------------|----------|-----------------------|----------|
| show                  | 0.2336425| show                  | 0.1643121|
| index                 | 0.0993014| header_userpic        | 0.0822204|
| search_results        | 0.0848848| active                | 0.0682795|
| personalize           | 0.0844791| index                 | 0.0612957|
| ajax_refresh_subtotal | 0.0591886| create                | 0.0596574|
| search                | 0.047732 | dashboard             | 0.0533352|
| similar_listings      | 0.0455265| personalize           | 0.0493038|
| update                | 0.0406442| search                | 0.0481939|
| social_connections    | 0.0336528| update                | 0.037676 |
| reviews               | 0.0299896| search_results        | 0.0304125|

The table above shows the 10 most common actions that the top and the bottom 20% of users by retention days have done in the service. The second most top column in Topside, 'Index', includes viewing various results such as search results, listing, and wish list. Also, Top 20% of users do a lot of actions that are far from immediate reservations such as personalize, which is updating wish lists and social connections. On the other hand, one-time actions such as update user profile pictures, phone numbers, and creating accounts are the most common actions in Bottom 20% users' actions list. 

![Retention Booked / Unbooked](https://i.imgur.com/s56n98z.png)

There was also a difference between customers who used the service and customers who were not. The return rate of experienced customers tended to be slightly higher. (This graph reflects only users data from the  training dataset)

![MAU](https://i.imgur.com/X3VgXza.png)

From the session data, I was able to trace the number of daily users after manipulating and calculating the seconds elapsed column. Because I only have data from both of the demographic and sessions sets between 2014 January to September, the graph falls down after 2014 October.

![DAU](https://i.imgur.com/Q2YKSSg.png)

Conver to Daily Active Users to Monthly Active users graph. 

##### 4. Referral

Airbnb does not provide specific information related to referral. However, I found name of actions that seems related to referral. So I extracted a description of the actions that contain 'referr' in the names. ('ajax_get_referrals_amt' seems the action to check the number of referrals by the users, so the frequency of )

- ajax_get_referrals_amt                  
- ajax_referral_banner_experiment_type     
- ajax_referral_banner_type                
- referrer_status                          
- signup_weibo_referral                      
- weibo_signup_referral_finish             

| Referral Description  |
|-------|---------------|
| count |          9076 |
| sum   |         25019 |
| mean  |          2.75 |
| std   |          2.58 |
| min   |          1.00 |
| 25%   |          2.00 |
| 50%   |          2.00 |
| 75%   |          3.00 |
| max   |         56.00 |

The result reveals that 9076 users made 25019 actions related to the referral. The number of total users in dataset is 275547, so conversion rate from acquisition to referral in this dataset is around 3.2%.

##### 5. Revenue

For estimating the Revenue, I only use the training dataset because the testing set does not include booking information. Accroding to the training data, 88,908 users have history to book accormordations through Airbnb out of 213,451 total users. Nearly 42% of users have used the service. 

It seems the conversion rate is abnormally high and I assumed that Airbnb provides filtered data for the competition since the purpose of it is to predict the users' first destination country. Anyway, I extracted revenue and converted users information as much as I can from the dataset.

![First Destination](https://imgur.com/yeWp2FV.png)

This graph presents the ratio by users' first destination. 'NDF' in here means 'No Destination Found', users had not made a reservation yet. 


2. 성별로 확인해보면 아직 예약하지 않은 유저의 경우 성별을 특정해놓지 않았을 확률이 높음 -> 이건 예약하기 위해서 설정을 하는 것인지, 설정한 사람이 예약할 확률이 높은 지의 문제가 있을 수 있음. 아무튼 lock in 해두면 좋은 거 아닐까 ?
3. 국가별
4. 첫 예약일 - 가입일. 아무래도 필요할 때 가입하는 경향이 높은 듯. 그럼 유입하고 Lock In 할 수 있는 요인을 만들어서 Retention을 높이는 전략이 좋지 않을까 ?

### 3. Modeling

### 4. Result

