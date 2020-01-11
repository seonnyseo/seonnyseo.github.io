---
layout: post
title:  "Text Classification on Social Media Text"
date:   2019-08-24 23:59:00 +0100
categories: Python Twitter API MachineLearning Text Classification
---


### 1.Collecting Tweets through Twitter API

Using Twitter API, we can collect information tweets includes user information, date, and tweets. For this, install and import Twitter API first.


```python
!pip install python-twitter
import twitter
import json
```

    Collecting python-twitter
    [?25l  Downloading https://files.pythonhosted.org/packages/b3/a9/2eb36853d8ca49a70482e2332aa5082e09b3180391671101b1612e3aeaf1/python_twitter-3.5-py2.py3-none-any.whl (67kB)
    [K     |████████████████████████████████| 71kB 4.6MB/s 
    [?25hRequirement already satisfied: future in /usr/local/lib/python3.6/dist-packages (from python-twitter) (0.16.0)
    Requirement already satisfied: requests in /usr/local/lib/python3.6/dist-packages (from python-twitter) (2.21.0)
    Requirement already satisfied: requests-oauthlib in /usr/local/lib/python3.6/dist-packages (from python-twitter) (1.2.0)
    Requirement already satisfied: idna<2.9,>=2.5 in /usr/local/lib/python3.6/dist-packages (from requests->python-twitter) (2.8)
    Requirement already satisfied: urllib3<1.25,>=1.21.1 in /usr/local/lib/python3.6/dist-packages (from requests->python-twitter) (1.24.3)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.6/dist-packages (from requests->python-twitter) (2019.6.16)
    Requirement already satisfied: chardet<3.1.0,>=3.0.2 in /usr/local/lib/python3.6/dist-packages (from requests->python-twitter) (3.0.4)
    Requirement already satisfied: oauthlib>=3.0.0 in /usr/local/lib/python3.6/dist-packages (from requests-oauthlib->python-twitter) (3.1.0)
    Installing collected packages: python-twitter
    Successfully installed python-twitter-3.5


```python
import keys
api = twitter.Api(consumer_key = keys.consumer_key,
                 consumer_secret = keys.consumer_secret,
                 access_token_key = keys.access_token,
                 access_token_secret = keys.access_token_secret)
```

```python
from google.colab import files
uploaded = files.upload()
for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
  name = fn, length = len(uploaded[fn])))
```



     <input type="file" id="files-c423007e-c49b-47fa-866e-6092b97d5ca3" name="files[]" multiple disabled />
     <output id="result-c423007e-c49b-47fa-866e-6092b97d5ca3">
      Upload widget is only available when the cell has been executed in the
      current browser session. Please rerun this cell to enable.
      </output>
      <script src="/nbextensions/google.colab/files.js"></script> 


    Saving drug_safety_data.txt to drug_safety_data.txt
    User uploaded file "drug_safety_data.txt" with length 185065 bytes


Tweets ids and twitter user ids are in 'drug_safety_data.txt' file. 

1. Collect tweets information that we will refer through twitter API
2. The data are stored as json format
3. Text in below shows how it is constructed 

{% highlight html linenos %}
```python
import pandas as pd
drugTweets = pd.read_csv('drug_safety_data.txt', delimiter = '\t', header = None, names = ['tweet_id', 'twitter_user_id', 'abuse'])
drugTweets = drugTweets.drop_duplicates()
drugTweets_text = api.GetStatuses(drugTweets.tweet_id)
txts = []
for tweet in drugTweets_text:
  txts.append(json.loads(json.dumps(tweet._json)))
txts[0]

```
{% endhighlight %}



    {'contributors': None,
     'coordinates': None,
     'created_at': 'Sun May 12 18:08:41 +0000 2013',
     'entities': {'hashtags': [], 'symbols': [], 'urls': [], 'user_mentions': []},
     'favorite_count': 0,
     'favorited': False,
     'geo': None,
     'id': 333644914913079296,
     'id_str': '333644914913079296',
     'in_reply_to_screen_name': None,
     'in_reply_to_status_id': None,
     'in_reply_to_status_id_str': None,
     'in_reply_to_user_id': None,
     'in_reply_to_user_id_str': None,
     'is_quote_status': False,
     'lang': 'en',
     'place': None,
     'retweet_count': 0,
     'retweeted': False,
     'source': '<a href="http://twitter.com/download/iphone" rel="nofollow">Twitter for iPhone</a>',
     'text': 'i know for a FACT that alcohol does not deplete the seroquel levels in your blood, YET HERE WE ARE',
     'truncated': False,
     'user': {'contributors_enabled': False,
      'created_at': 'Fri Dec 24 03:45:56 +0000 2010',
      'default_profile': False,
      'default_profile_image': False,
      'description': 'revive prime suspect (2011 - 2012)',
      'entities': {'description': {'urls': []},
       'url': {'urls': [{'display_url': 'nastyratched.tumblr.com',
          'expanded_url': 'http://nastyratched.tumblr.com',
          'indices': [0, 23],
          'url': 'https://t.co/Ce4IWf7ziQ'}]}},
      'favourites_count': 55542,
      'follow_request_sent': False,
      'followers_count': 178,
      'following': False,
      'friends_count': 570,
      'geo_enabled': True,
      'has_extended_profile': False,
      'id': 230052171,
      'id_str': '230052171',
      'is_translation_enabled': False,
      'is_translator': False,
      'lang': None,
      'listed_count': 4,
      'location': 'perth, western australia',
      'name': 'john s. lithgoat',
      'notifications': False,
      'profile_background_color': 'CCCCCC',
      'profile_background_image_url': 'http://abs.twimg.com/images/themes/theme1/bg.png',
      'profile_background_image_url_https': 'https://abs.twimg.com/images/themes/theme1/bg.png',
      'profile_background_tile': True,
      'profile_banner_url': 'https://pbs.twimg.com/profile_banners/230052171/1516295902',
      'profile_image_url': 'http://pbs.twimg.com/profile_images/1054691338610913280/oINSEFmk_normal.jpg',
      'profile_image_url_https': 'https://pbs.twimg.com/profile_images/1054691338610913280/oINSEFmk_normal.jpg',
      'profile_link_color': '999999',
      'profile_sidebar_border_color': '000000',
      'profile_sidebar_fill_color': '333333',
      'profile_text_color': '666666',
      'profile_use_background_image': False,
      'protected': False,
      'screen_name': 'pants2match',
      'statuses_count': 33777,
      'time_zone': None,
      'translator_type': 'none',
      'url': 'https://t.co/Ce4IWf7ziQ',
      'utc_offset': None,
      'verified': False}}



From the json format, take out necessary information such as user_ids and texts


```python
ids = []
text = []
for line in txts:
  ids.append(line['id'])
  text.append(line['text'])
```

### 2.Text pre-processing

Tweets were written in the informal language in most cases, and included reserved words related to Twitter. To improve machine learning models performance, it is required to clean unnecessary text up to teach models clearly. 


#### 1.Remove Twitter reserved word


```python
!pip install tweet-preprocessor
import preprocessor as p
```

    Collecting tweet-preprocessor
      Downloading https://files.pythonhosted.org/packages/2a/f8/810ec35c31cca89bc4f1a02c14b042b9ec6c19dd21f7ef1876874ef069a6/tweet-preprocessor-0.5.0.tar.gz
    Building wheels for collected packages: tweet-preprocessor
      Building wheel for tweet-preprocessor (setup.py) ... [?25l[?25hdone
      Created wheel for tweet-preprocessor: filename=tweet_preprocessor-0.5.0-cp36-none-any.whl size=7946 sha256=d7bdb1b8e01bb419d1466c4c846844d6f8e02e65bf0986cc5814632499f7228f
      Stored in directory: /root/.cache/pip/wheels/1b/27/cc/49938e98a2470802ebdefae9d2b3f524768e970c1ebbe2dc4a
    Successfully built tweet-preprocessor
    Installing collected packages: tweet-preprocessor
    Successfully installed tweet-preprocessor-0.5.0


tweet-preprocessor library supports to remove these text.


1.   URLs
2.   Hashtags
3.   Mentions
4.   Reserved words (RT, FAV)
5.   Emojis
6.   Smileys
7.   Number


```python
text_clean = []
for line in text:
  text_clean.append(p.clean(line))
print('Before : {}'.format(text[1]))
print('After  : {}'.format(text_clean[1]))
```

    Before : @Scribble_Dragon 50 mg Seroquel with my ‘normal’ 60 mg Lovan and 750 mcg Clonazepam.
    After  : mg Seroquel with my ‘normal’ mg Lovan and mcg Clonazepam.


After cleaning twitter reserved words, put this on the data frame that includes tweet_id, user_id and classfication label information.


```python
tweets_w_text = pd.DataFrame(list(zip(ids, text_clean)), columns = ['tweet_id', 'text_text'])
drugTweets_df = pd.merge(tweets_w_text, drugTweets, on = 'tweet_id', how = 'inner')
```

#### 2.Stopwords / Lowercase / Stemming

Besides removing Twitter words, we can remove stopwords that would not give important information and lowercase every text for avoiding counting same words several times. Also, we decide to apply stemming to reduce inflected words to their word stem.


```python
import nltk
import re
nltk.download('stopwords')
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
```

    [nltk_data] Downloading package stopwords to /root/nltk_data...
    [nltk_data]   Unzipping corpora/stopwords.zip.


'fix_text' function includes removing stopwords, stemming, lowercasing and removing special characters. 


```python
stop_words = set(stopwords.words("english"))
snow = nltk.stem.SnowballStemmer('english')

def fix_Text(text):
	letters_only = re.sub("[^a-zA-Z]"," ", str(text))
	words=letters_only.lower().split()
	meaningful=[snow.stem(word) for word in words if word not in stop_words]
	return(" ".join(meaningful))
```


```python
num_resp = drugTweets_df["text_text"].size
print("Before : {}".format(drugTweets_df['text_text'][2]))
clean_text = []
for i in range(0,num_resp):
	clean_text.append(fix_Text(drugTweets_df["text_text"][i]))

print("After : {}".format(clean_text[2]))
```

    Before : SEVEN missed calls? get you're seroquel mg lowered. you're getting ridiculous
    After : seven miss call get seroquel mg lower get ridicul


Also we found that twitter users use two words for one medicine such and also they type energy drink 'red bull' or 'redbull'. We thought it is reasonable to count these words together. Also, even we removed special characters, broken codes are still in text such as 'amp', 'lt,' and 'gt' because thoese are alphabet characters. So we made a function that change specific words, and remove the broken codes. 


```python
word_list = {'quetiapin' : 'seroquel', 'oxycontin' : 'oxycodone', 'red bull' : 'redbull', 'amp':'', 'lt':'', 'gt':''}

def change_word(text):
  for key in list(word_list.keys()):
    if key in text:
      text = text.replace(key, word_list[key])

  return text
```


```python
print("Before : {}".format(clean_text[5]))
for i in range(num_resp):
  clean_text[i] = change_word(clean_text[i])

print("After : {}".format(clean_text[5]))
```

    Before : antipsychot quetiapin sedat olanzapin risperidon aripiprazol lithium augment agent
    After : antipsychot seroquel sedat olanzapin risperidon aripiprazol lithium augment agent


### 3.Document - Term representation

In order to classify tweets by machine learning models, we need to create a document-term representation. Numbers in the matrix represent how important a word is to a document in a collection or corpus.

#### 1.Term Frequency

Using by CountVectorizer function, we can tokenize and count the frequency of words in tweets. After it, fit_transform module creats a Document-Term matrix. 


```python
from sklearn.feature_extraction.text import CountVectorizer
```

```python
tfVectorizer=CountVectorizer()
tfdtm= tfVectorizer.fit_transform(clean_text)
tfVectorizer.get_feature_names()[0:5]
```




    ['abbi', 'abid', 'abil', 'abilifi', 'abl']



#### 2.Term Frequency - Inverse Document Frequency (Feature Selection)

The frequency of words does not tell us how the words important, because some not important words such as 'I', 'the', 'a' would frequently appear than other terms. Term Frequency - Inverse Document Frequency value represents priority by the number of appearance in the document / the number of occurrence in the corpus. 

TfidfVectorizer module helps create tfidf matrix and set the minimum number of appearance. After creating the counting vector, convert it to data frame that we are going to use it for classification modeling.


```python
from sklearn.feature_extraction.text import TfidfVectorizer
```

We then created 3 datasets where only the terms that appeared a minimum of 20, 30 and 50 times respectively would be included. With the minimum term appearance of 20, 226 features were retained for a minimum of 30 terms, 137 features were retained, and with a minimum of 50 term appearances, 67 features were retained. 

We evaluated the result of each frequency before, and 30 reveals the best performance. Thus, we decide to set minimum frequency of words ad 30 here. 


```python
tfidfVectorizer=TfidfVectorizer(min_df=30)
tfidfdtm = tfidfVectorizer.fit_transform(clean_text)
tfidfVectorizer.get_feature_names()[0:5]
```




    ['actual', 'adderal', 'addict', 'also', 'ask']




```python
tfidf_df = pd.DataFrame(tfidfdtm.toarray(), columns=tfidfVectorizer.get_feature_names())
tfidf_df[:10]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>actual</th>
      <th>adderal</th>
      <th>addict</th>
      <th>also</th>
      <th>ask</th>
      <th>ass</th>
      <th>back</th>
      <th>bad</th>
      <th>bed</th>
      <th>best</th>
      <th>better</th>
      <th>call</th>
      <th>caus</th>
      <th>coffe</th>
      <th>come</th>
      <th>could</th>
      <th>day</th>
      <th>doctor</th>
      <th>done</th>
      <th>dose</th>
      <th>drink</th>
      <th>drug</th>
      <th>eat</th>
      <th>effect</th>
      <th>even</th>
      <th>ever</th>
      <th>everi</th>
      <th>feel</th>
      <th>final</th>
      <th>find</th>
      <th>first</th>
      <th>focus</th>
      <th>friend</th>
      <th>fuck</th>
      <th>gave</th>
      <th>get</th>
      <th>give</th>
      <th>go</th>
      <th>gonna</th>
      <th>good</th>
      <th>...</th>
      <th>shit</th>
      <th>sinc</th>
      <th>sleep</th>
      <th>someon</th>
      <th>someth</th>
      <th>start</th>
      <th>stay</th>
      <th>still</th>
      <th>stop</th>
      <th>studi</th>
      <th>sure</th>
      <th>surgeri</th>
      <th>take</th>
      <th>taken</th>
      <th>talk</th>
      <th>tell</th>
      <th>thank</th>
      <th>thing</th>
      <th>think</th>
      <th>thought</th>
      <th>time</th>
      <th>today</th>
      <th>tomorrow</th>
      <th>tonight</th>
      <th>took</th>
      <th>tri</th>
      <th>two</th>
      <th>use</th>
      <th>wake</th>
      <th>wanna</th>
      <th>want</th>
      <th>way</th>
      <th>week</th>
      <th>well</th>
      <th>without</th>
      <th>work</th>
      <th>would</th>
      <th>xanax</th>
      <th>yeah</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.532974</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.696834</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.69535</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.16467</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.269218</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.448937</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.494662</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.445602</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.747859</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.45356</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.444197</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.46185</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.459702</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.378680</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 134 columns</p>
</div>




```
drugTweets_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tweet_id</th>
      <th>text_text</th>
      <th>twitter_user_id</th>
      <th>abuse</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>333644914913079296</td>
      <td>i know for a FACT that alcohol does not deplet...</td>
      <td>230052171</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>344825926342832128</td>
      <td>mg Seroquel with my ‘normal’ mg Lovan and mcg ...</td>
      <td>179074771</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>333854289023864832</td>
      <td>SEVEN missed calls? get you're seroquel mg low...</td>
      <td>333099736</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>344606561873833985</td>
      <td>there's a fella on my Facebook who is asking t...</td>
      <td>464202509</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>341947615853813761</td>
      <td>you take vyvanse? I was on that stuff for like...</td>
      <td>90304006</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>328655421256654849</td>
      <td>antipsychotics: quetiapine (sedation); olanzap...</td>
      <td>1371093139</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>333735685406224385</td>
      <td>I take quetiapine and it's supposed to just re...</td>
      <td>38919907</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>344937940146855937</td>
      <td>Seroquel is pretty heavy stuff. I would've tho...</td>
      <td>185070700</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>342112352130449409</td>
      <td>Tell me why this kid just gave me six seroquel...</td>
      <td>345065773</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>341680382179160065</td>
      <td>look at the tweet near that one. I refuse to t...</td>
      <td>38971420</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>332223414595117056</td>
      <td>Just about dead, think it's bedtime.. Fuck you...</td>
      <td>52782775</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>333733798787284992</td>
      <td>I may not have weed, but I do have seroquel I ...</td>
      <td>1373158400</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>340530953015402497</td>
      <td>I hate to hear that. Taking seroquel is like s...</td>
      <td>323940583</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>343565496320417792</td>
      <td>Im bout to slip some of my seroquel into her d...</td>
      <td>369114238</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>344277608759975936</td>
      <td>I prescribed quetiapine to my obese patient al...</td>
      <td>46989098</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>341178969535705088</td>
      <td>&amp;lt; Seroquel - at high doses its for psychoti...</td>
      <td>386313872</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>340274216471515136</td>
      <td>but I WOULD use it if I were asleep on quetiap...</td>
      <td>1123283994</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>329007984845926402</td>
      <td>were on the same meds haha serequol and quetia...</td>
      <td>857673764</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>329391569620856833</td>
      <td>they mentioned quetiapine for me ages ago, but...</td>
      <td>586045105</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>342143067169632256</td>
      <td>I been knocked the fuck out yo. Like I took a ...</td>
      <td>190770320</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>343814505706430464</td>
      <td>Right, got a date with Ms Quetiapine and Law A...</td>
      <td>18332340</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>341026211218194432</td>
      <td>am and the quetiapine has failed to sedate me</td>
      <td>1317291756</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>340134027535204352</td>
      <td>"I DONT TAKE MEDS IN THE MORNING! VITAMINS? SE...</td>
      <td>425214012</td>
      <td>0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>341242496321482753</td>
      <td>(Triggering) So... I'm taking quetiapine and t...</td>
      <td>562490916</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>330278165765312512</td>
      <td>I'll be on mg of Quetiapine for the next night...</td>
      <td>273421529</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>344351386542174208</td>
      <td>okay i took the seroquel and i am resisting th...</td>
      <td>15147617</td>
      <td>0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>344359166556639232</td>
      <td>: lemme suck the seroquel residue off your fin...</td>
      <td>31061240</td>
      <td>1</td>
    </tr>
    <tr>
      <th>27</th>
      <td>329560087259672576</td>
      <td>Do you know what Meds are R for bipolar depres...</td>
      <td>158119360</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>340152385051688960</td>
      <td>ur so ignorant im at the hospital right now to...</td>
      <td>1429295593</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>340601194655395840</td>
      <td>Just seeing whats occurring on twitter. .while...</td>
      <td>1449942523</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2945</th>
      <td>541062704102785024</td>
      <td>I tried air before Seroquel but when I found i...</td>
      <td>502126127</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2946</th>
      <td>541855078920228864</td>
      <td>"Because people with ADD/ADHD don't deserve to...</td>
      <td>449961812</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2947</th>
      <td>540597014640095232</td>
      <td>Haloti Ngata got suspended games for taking Ad...</td>
      <td>542355930</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2948</th>
      <td>540681647482368000</td>
      <td>half you niggas be thinking yall popping zans ...</td>
      <td>2325682592</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2949</th>
      <td>540966927993028608</td>
      <td>was reaching for a bottle of oxycodone pills w...</td>
      <td>2904639999</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2950</th>
      <td>541371565003526145</td>
      <td>Officers seized total of Oxycodone pills — -mg...</td>
      <td>942487562</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2951</th>
      <td>540763703340056576</td>
      <td>": I'm going to name my first born oxycodone b...</td>
      <td>34191550</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2952</th>
      <td>540919065385390080</td>
      <td>seroquel is certainly powerful. i took a half ...</td>
      <td>280230553</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2953</th>
      <td>541758583063343104</td>
      <td>AndisGraudins RCT IbuprofPanadol w or w/out co...</td>
      <td>980098682</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2954</th>
      <td>540743596639780864</td>
      <td>I swore this oxycodone would've knocked me out...</td>
      <td>188150281</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2955</th>
      <td>540596388074373120</td>
      <td>Oxycodone also numbs the pain when watford los...</td>
      <td>302174491</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2956</th>
      <td>541739968754761728</td>
      <td>hey hey hey hey hey the oxycodone is startting...</td>
      <td>525375457</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2957</th>
      <td>541131854636916736</td>
      <td>The dreams I have while on Oxycodone are inane...</td>
      <td>1154416777</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2958</th>
      <td>540679886307065857</td>
      <td>killed for holding a bottle of oxycodone pills.</td>
      <td>36519411</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2959</th>
      <td>541547716220698624</td>
      <td>if I take my seroquel I sleep too much and if ...</td>
      <td>421696198</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2960</th>
      <td>541017124559675393</td>
      <td>thank you seroquel for making me sleep for hou...</td>
      <td>1549264729</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2961</th>
      <td>541856612529750018</td>
      <td>fuck that guy. Is Adderall really cheaper than...</td>
      <td>939762660</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2962</th>
      <td>541722116052111360</td>
      <td>": Took adderall to help me focus on school bu...</td>
      <td>40339249</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2963</th>
      <td>540634933018898432</td>
      <td>We still don't know if the pill bottle w/ oxyc...</td>
      <td>28476383</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2964</th>
      <td>541057303781576704</td>
      <td>playing cards against humanity on seroquel can...</td>
      <td>221562659</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2965</th>
      <td>541556536867168256</td>
      <td>I'm too high on oxycodone for my tooth ache to...</td>
      <td>22197146</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2966</th>
      <td>541870151751065600</td>
      <td>The kind of night where I feel like I need Add...</td>
      <td>1686924378</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2967</th>
      <td>541984407578349568</td>
      <td>Lol! Lambert must be on a cocktail of Xanax, S...</td>
      <td>621018967</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2968</th>
      <td>542018327359406080</td>
      <td>What is the most potent form and delivery meth...</td>
      <td>170357723</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2969</th>
      <td>542019095353622528</td>
      <td>Adderall is gonna be my new best friend. Along...</td>
      <td>430195866</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2970</th>
      <td>541858163289784320</td>
      <td>I'm hip, but that's what my other tweet is abo...</td>
      <td>288784938</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2971</th>
      <td>542031072171933697</td>
      <td>My body already hates me for the copious amoun...</td>
      <td>360382242</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2972</th>
      <td>541866635964203009</td>
      <td>Adderall will make you check twitter plus time...</td>
      <td>281287390</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2973</th>
      <td>541998263256485890</td>
      <td>finals week tip: make sure u poop, pee, and ea...</td>
      <td>867182059</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2974</th>
      <td>541886694111604736</td>
      <td>My favorite thing about adderall is how it kee...</td>
      <td>777096356</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>2975 rows × 4 columns</p>
</div>




```python
tfidf_df['abused'] = drugTweets_df.abuse
```

###4.Modeling

Data set for modeling is made through text pre-processing and creating a tf-idf matrix. In this part, we will focus on how to make classfication models.

####1.Import classifiers

Importing classification model packages.

We will train KNN, SVM, Naive Bayes, and Decision tree models, and compare performance.


```python
from sklearn.metrics import confusion_matrix
from sklearn import metrics

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC  
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
```

####2.Load DTM File and split as train & test

Use a tf-idf matrix that we made above as dataset.

1.   Take a label column in dataset as y.
2.   Split dataset as training and test set. (Training 80% / Test 20%)




```python
X = tfidf_df.drop('abused', axis = 1)
y = tfidf_df.abused


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 123)
```

Check how many abuse tweets do we have on our dataset.

0 is not abused one, and 1 is abused. So, we have about 15% of tweets that were labeled as abused.

We can tell it is imbalanced dataset. 


```python
tfidf_df.abused.value_counts()
```




    0    2534
    1     441
    Name: abused, dtype: int64



#### 3.Resampling

If we train models with imbalanced data, the models will be biased to predict majority class which is not aligned to the purpose of this study. In order to resolve this problem, we executed resamplings to make dataset balanced.

We will apply 3 types of resampling and pick the best one.


1. Random Over Sampling
2. Random Under Sampling
3. SMOTE




```python
from sklearn.utils import resample
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
```

Oversampling  (X_train_up, y_train_up)



```python
ros = RandomOverSampler(random_state = 123)
X_train_up, y_train_up = ros.fit_resample(X_train, y_train)
print(sorted(Counter(y_train_up).items()))
```

    [(0, 2029), (1, 2029)]


SMOTE (X_train_SMOTE, y_train_SMOTE)


```python
X_train_SMOTE, y_train_SMOTE = SMOTE().fit_resample(X_train, y_train)
print(sorted(Counter(y_train_SMOTE).items()))
```

    [(0, 2029), (1, 2029)]


Undersampling (X_train_under, y_train_under)


```python
rus = RandomUnderSampler(random_state = 123)
X_train_under, y_train_under = rus.fit_resample(X_train, y_train)
print(sorted(Counter(y_train_under).items()))
```

    [(0, 351), (1, 351)]


#### 4.Train and evaluate 


With oversampled data, we will train models, test on those and compare which classification model shows the best performance. 

Here is three functions that we made to work done easily.


*   fit_models - Training four models(KNN, SVM, Decision Tree, Naive Bayes) with oversampled data.
*   compare_model - Predict abused tweets based on the models trained on fit_models function.
*   print_result - Put training and test result as inputs and compare models peformance comfortably.




```python
def fit_models(X_train, y_train):
  
  knn = KNeighborsClassifier()
  svm = SVC(kernel = 'linear', random_state = 123)
  dt = DecisionTreeClassifier(random_state = 123)
  nb = GaussianNB()

  _models = [knn, svm, dt, nb]
  
  classifiers = []
  for classifier in _models:
    classifier.fit(X = X_train, y = y_train)
    classifiers.append(classifier)
    
  return classifiers
```


```python
def compare_models(classifiers, X, y):

  reports = []
  matrix = []
  
  for _classi in classifiers:
    _predicted = _classi.predict(X = X)
    _report = metrics.classification_report(y, _predicted)
    _matrix = metrics.confusion_matrix(y, _predicted)
    
    reports.append(_report)
    matrix.append(_matrix)
  
  return (reports, matrix)
```


```python
def print_result(models, train_report, test_report):
  
  for model, train, test in zip(models, train_report, test_report):
    print('{:_<112}'.format(model))
    print('{}  {}  {}'.format('train',' ' * 55, 'test'))
    
    train_lines = train.split('\n')
    test_lines = test.split('\n')
    
    for train_line, test_line in zip(train_lines, test_lines):
      print(train_line + ' ' * 5 + test_line)
```


```python
models = ['KNN', 'SVM_Linear', 'DecisionTree','NaiveBayesian']
```

Oversampling Result


```python
_models_up = fit_models(X_train_up, y_train_up)
train_report_up, train_matrix_up = compare_models(_models_up, X_train_up, y_train_up)
test_report_up, test_matrix_up = compare_models(_models_up, X_test, y_test)
```


```python
print_result(models, train_report_up, test_report_up)
```

    KNN_____________________________________________________________________________________________________________
    train                                                           test
                  precision    recall  f1-score   support                   precision    recall  f1-score   support
         
               0       0.96      0.71      0.82      2029                0       0.87      0.63      0.73       505
               1       0.77      0.97      0.86      2029                1       0.19      0.48      0.27        90
         
        accuracy                           0.84      4058         accuracy                           0.61       595
       macro avg       0.87      0.84      0.84      4058        macro avg       0.53      0.55      0.50       595
    weighted avg       0.87      0.84      0.84      4058     weighted avg       0.77      0.61      0.66       595
         
    SVM_Linear______________________________________________________________________________________________________
    train                                                           test
                  precision    recall  f1-score   support                   precision    recall  f1-score   support
         
               0       0.72      0.68      0.70      2029                0       0.92      0.63      0.75       505
               1       0.70      0.74      0.72      2029                1       0.25      0.69      0.36        90
         
        accuracy                           0.71      4058         accuracy                           0.64       595
       macro avg       0.71      0.71      0.71      4058        macro avg       0.58      0.66      0.56       595
    weighted avg       0.71      0.71      0.71      4058     weighted avg       0.82      0.64      0.69       595
         
    DecisionTree____________________________________________________________________________________________________
    train                                                           test
                  precision    recall  f1-score   support                   precision    recall  f1-score   support
         
               0       0.96      0.94      0.95      2029                0       0.87      0.83      0.85       505
               1       0.95      0.96      0.95      2029                1       0.25      0.32      0.28        90
         
        accuracy                           0.95      4058         accuracy                           0.75       595
       macro avg       0.95      0.95      0.95      4058        macro avg       0.56      0.57      0.56       595
    weighted avg       0.95      0.95      0.95      4058     weighted avg       0.78      0.75      0.76       595
         
    NaiveBayesian___________________________________________________________________________________________________
    train                                                           test
                  precision    recall  f1-score   support                   precision    recall  f1-score   support
         
               0       0.78      0.48      0.59      2029                0       0.90      0.44      0.59       505
               1       0.62      0.86      0.72      2029                1       0.18      0.71      0.29        90
         
        accuracy                           0.67      4058         accuracy                           0.48       595
       macro avg       0.70      0.67      0.66      4058        macro avg       0.54      0.58      0.44       595
    weighted avg       0.70      0.67      0.66      4058     weighted avg       0.79      0.48      0.55       595
         


Undersampling Result


```python
_models_under = fit_models(X_train_under, y_train_under)
train_report_under, train_matrix_under = compare_models(_models_under, X_train_under, y_train_under)
test_report_under, test_matrix_under = compare_models(_models_under, X_test, y_test)

print_result(models, train_report_under, test_report_under)
```

    KNN_____________________________________________________________________________________________________________
    train                                                           test
                  precision    recall  f1-score   support                   precision    recall  f1-score   support
         
               0       0.72      0.70      0.71       351                0       0.89      0.51      0.65       505
               1       0.71      0.73      0.72       351                1       0.19      0.66      0.30        90
         
        accuracy                           0.72       702         accuracy                           0.53       595
       macro avg       0.72      0.72      0.72       702        macro avg       0.54      0.58      0.47       595
    weighted avg       0.72      0.72      0.72       702     weighted avg       0.79      0.53      0.59       595
         
    SVM_Linear______________________________________________________________________________________________________
    train                                                           test
                  precision    recall  f1-score   support                   precision    recall  f1-score   support
         
               0       0.78      0.71      0.74       351                0       0.93      0.55      0.69       505
               1       0.73      0.80      0.77       351                1       0.23      0.76      0.35        90
         
        accuracy                           0.75       702         accuracy                           0.58       595
       macro avg       0.76      0.75      0.75       702        macro avg       0.58      0.65      0.52       595
    weighted avg       0.76      0.75      0.75       702     weighted avg       0.82      0.58      0.64       595
         
    DecisionTree____________________________________________________________________________________________________
    train                                                           test
                  precision    recall  f1-score   support                   precision    recall  f1-score   support
         
               0       0.95      0.96      0.96       351                0       0.90      0.58      0.70       505
               1       0.96      0.95      0.96       351                1       0.21      0.63      0.32        90
         
        accuracy                           0.96       702         accuracy                           0.59       595
       macro avg       0.96      0.96      0.96       702        macro avg       0.55      0.61      0.51       595
    weighted avg       0.96      0.96      0.96       702     weighted avg       0.79      0.59      0.65       595
         
    NaiveBayesian___________________________________________________________________________________________________
    train                                                           test
                  precision    recall  f1-score   support                   precision    recall  f1-score   support
         
               0       0.74      0.72      0.73       351                0       0.92      0.56      0.70       505
               1       0.73      0.75      0.74       351                1       0.22      0.71      0.34        90
         
        accuracy                           0.74       702         accuracy                           0.58       595
       macro avg       0.74      0.74      0.74       702        macro avg       0.57      0.64      0.52       595
    weighted avg       0.74      0.74      0.74       702     weighted avg       0.81      0.58      0.64       595
         


SMOTE Result


```python
_models_SMOTE = fit_models(X_train_SMOTE, y_train_SMOTE)
train_report_SMOTE, train_matrix_SMOTE = compare_models(_models_SMOTE, X_train_SMOTE, y_train_SMOTE)
test_report_SMOTE, test_matrix_SMOTE = compare_models(_models_SMOTE, X_test, y_test)


print_result(models, train_report_SMOTE, test_report_SMOTE)
```

    KNN_____________________________________________________________________________________________________________
    train                                                           test
                  precision    recall  f1-score   support                   precision    recall  f1-score   support
         
               0       0.95      0.67      0.78      2029                0       0.89      0.56      0.69       505
               1       0.74      0.97      0.84      2029                1       0.20      0.60      0.30        90
         
        accuracy                           0.82      4058         accuracy                           0.57       595
       macro avg       0.85      0.82      0.81      4058        macro avg       0.54      0.58      0.49       595
    weighted avg       0.85      0.82      0.81      4058     weighted avg       0.78      0.57      0.63       595
         
    SVM_Linear______________________________________________________________________________________________________
    train                                                           test
                  precision    recall  f1-score   support                   precision    recall  f1-score   support
         
               0       0.77      0.69      0.73      2029                0       0.92      0.65      0.76       505
               1       0.72      0.80      0.76      2029                1       0.26      0.68      0.37        90
         
        accuracy                           0.74      4058         accuracy                           0.66       595
       macro avg       0.74      0.74      0.74      4058        macro avg       0.59      0.67      0.57       595
    weighted avg       0.74      0.74      0.74      4058     weighted avg       0.82      0.66      0.70       595
         
    DecisionTree____________________________________________________________________________________________________
    train                                                           test
                  precision    recall  f1-score   support                   precision    recall  f1-score   support
         
               0       0.97      0.96      0.97      2029                0       0.87      0.85      0.86       505
               1       0.96      0.97      0.97      2029                1       0.24      0.27      0.25        90
         
        accuracy                           0.97      4058         accuracy                           0.76       595
       macro avg       0.97      0.97      0.97      4058        macro avg       0.55      0.56      0.56       595
    weighted avg       0.97      0.97      0.97      4058     weighted avg       0.77      0.76      0.77       595
         
    NaiveBayesian___________________________________________________________________________________________________
    train                                                           test
                  precision    recall  f1-score   support                   precision    recall  f1-score   support
         
               0       0.81      0.56      0.66      2029                0       0.89      0.52      0.66       505
               1       0.66      0.87      0.75      2029                1       0.19      0.64      0.30        90
         
        accuracy                           0.71      4058         accuracy                           0.54       595
       macro avg       0.74      0.71      0.71      4058        macro avg       0.54      0.58      0.48       595
    weighted avg       0.74      0.71      0.71      4058     weighted avg       0.79      0.54      0.61       595
         


### 5.Validation

To validate our model, we run cross validation and draw ROC curve.

#### 1.Cross Validation

Since the data is imbalanced, we adopt stratifiedKFold function that keeps class weights on spliiting data for cross validation. Also, we use f1_scorer to compare models by f1_measure, not Accuracy. 


```python
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer

def cross_validation(estimators, folds, X_mat, Y_vec):
   for estimator_name, estimator_object in estimators.items():
      
    f1_scorer = make_scorer(f1_score, pos_label= 1)
    kfolds = StratifiedKFold(n_splits=folds, random_state=123, shuffle=True)
    scores = cross_val_score(estimator=estimator_object, X=X_mat, y=Y_vec, cv=kfolds, scoring = f1_scorer)
    print(f'{estimator_name:>20}: '
          f'mean f1={scores.mean():.2%}; ' +
          f'standard deviation={scores.std():.2%}')
```


```python
estimators = {
    'knn': KNeighborsClassifier(),
    'svm': SVC(kernel='linear', random_state=123, class_weight = 'balanced'),
    'dt': DecisionTreeClassifier(random_state=123),
    'nb': GaussianNB()
}
```


```python
X_train_oversampled = pd.DataFrame(X_train_SMOTE, columns = X_test.columns)
oversampled_x = pd.concat([X_test, X_train_oversampled])
```


```python
y_train_oversampled = pd.Series(y_train_SMOTE)
oversampled_y = y_test.append(y_train_oversampled)
```

First, cross validation on oversampled data(Oversampled_training + test)


```python
cross_validation(estimators, 10, oversampled_x, oversampled_y)
```

                     knn: mean f1=76.73%; standard deviation=2.31%
                     svm: mean f1=71.05%; standard deviation=1.61%
                      dt: mean f1=81.23%; standard deviation=2.36%
                      nb: mean f1=70.92%; standard deviation=1.48%


Second, cross validation on Original data(training + test)


```python
cross_validation(estimators, 10, X, y)
```

                     knn: mean f1=17.79%; standard deviation=6.03%
                     svm: mean f1=34.55%; standard deviation=6.22%
                      dt: mean f1=23.45%; standard deviation=6.04%
                      nb: mean f1=30.07%; standard deviation=2.58%



```python
from sklearn.model_selection import GridSearchCV
```


```python
def svm_param_selection(X, y, nfolds):
  parameter_candidates = [
      {'C':[1,10,100,1000], 'kernel':['linear']}
  ]
  f1_scorer = make_scorer(f1_score, pos_label= 1)
  grid_search = GridSearchCV(estimator = SVC(), param_grid = parameter_candidates, cv = nfolds, scoring = f1_scorer)
  grid_search.fit(X, y)
  grid_search.best_params_
  return grid_search.best_params_

```


```python
_optimized = svm_param_selection(X_train_SMOTE, y_train_SMOTE, 10)
```


```python
_optimized
```




    {'C': 1000, 'kernel': 'linear'}




```python
svm = SVC(kernel = 'linear', C = 1000, random_state = 123)
svm.fit(X_train_SMOTE, y_train_SMOTE)
_predicted = svm.predict(X = X_test)
report = metrics.classification_report(y_test, _predicted)
print(report)

```

                  precision    recall  f1-score   support
    
               0       0.92      0.66      0.77       505
               1       0.26      0.66      0.37        90
    
        accuracy                           0.66       595
       macro avg       0.59      0.66      0.57       595
    weighted avg       0.82      0.66      0.71       595
    


#### 2.ROC curve

We looked at the Area Under the Curve (AUC) on a ROC graph to future compare the model performance. SVM outperforms the other models here as well.



```python
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np

```


```python
classifiers = [KNeighborsClassifier(), 
               SVC(kernel='linear', random_state=123, probability=True),
               DecisionTreeClassifier(random_state=123),
               GaussianNB()]


result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])
names = ['KNN', 'SVM', 'DT','GausianNB']

for idx in range(len(names)):
    model = classifiers[idx].fit(X_train_SMOTE, y_train_SMOTE)
    yproba = model.predict_proba(X_test)[::,1]
    
    fpr, tpr, _ = roc_curve(y_test,  yproba)
    auc = roc_auc_score(y_test, yproba)
    
    result_table = result_table.append({'classifiers':names[idx],
                                        'fpr':fpr, 
                                        'tpr':tpr, 
                                        'auc':auc}, ignore_index=True)

```


```python
print(result_table)
```

      classifiers  ...       auc
    0         KNN  ...  0.608702
    1         SVM  ...  0.715259
    2          DT  ...  0.565886
    3   GausianNB  ...  0.601980
    
    [4 rows x 4 columns]



```python
fig = plt.figure(figsize=(8,6))

for i in result_table.index:
    plt.plot(result_table.loc[i]['fpr'],
             result_table.loc[i]['tpr'], 
             label="{}. AUC={:.3f}".format(result_table.loc[i]['classifiers'] ,result_table.loc[i]['auc']))


plt.plot([0,1], [0,1], color='orange', linestyle='--')

plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("Flase Positive Rate", fontsize=15)

plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)

plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
plt.legend(prop={'size':13}, loc='lower right')

plt.show()
```


![png](project_code_files/project_code_98_0.png)

