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

```python
import keys
api = twitter.Api(consumer_key = 'Consumer-Key',
                 consumer_secret = 'Consumer-Secret-Key',
                 access_token_key = 'Access-Token-Key',
                 access_token_secret = 'Access-Token-Secret-Key')
```

|      Tweet ID     |      User ID   |Abused|
|:-----------------:|:--------------:|:----:|
|332694017852731392 |    187933275   |   1  |
|332778502946426880 |     24377539   |   0  |


Tweets ids and twitter user ids are in 'drug_safety_data.txt' file. 

1. Collect tweets information that we will refer through twitter API
2. The data are stored as json format
3. Text in below shows how it is constructed 

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

I only need 'id' and 'text' from the status.


{
...
'id': 333644914913079296'
'text': 'i know for a FACT that alcohol does not deplete the seroquel levels in your blood, YET HERE WE ARE'
...
}


Full description of Twitter API response can be checked [here](https://developer.twitter.com/en/docs/accounts-and-users/create-manage-lists/api-referenc/get-lists-statuses)



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
```

|      | Text |
|------|------|
|Before|@Scribble_Dragon 50 mg Seroquel with my ‘normal’ 60 mg Lovan and 750 mcg Clonazepam.|
|After |mg Seroquel with my ‘normal’ mg Lovan and mcg Clonazepam.|

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
clean_text = []
for i in range(0,num_resp):
	clean_text.append(fix_Text(drugTweets_df["text_text"][i]))
```

|      | Text |
|------|------|
|Before|SEVEN missed calls? get you're seroquel mg lowered. you're getting ridiculous|
|After |seven miss call get seroquel mg lower get ridicul|


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
for i in range(num_resp):
  clean_text[i] = change_word(clean_text[i])
```

|      | Text |
|------|------|
|Before|antipsychot quetiapin sedat olanzapin risperidon aripiprazol lithium augment agent|
|After |antipsychot seroquel sedat olanzapin risperidon aripiprazol lithium augment agent|



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

|      | actual | adderal | addict | also | ask |
|:----:|:------:|:-------:|:------:|:----:|:---:|
|0     |   0.0  |   0.0   |   0.0  |  0.0 | 0.0 |
|1     |   0.0  |   0.0   |   0.0  |  0.0 | 0.0 |
|2     |   0.0  |   0.0   |0.69535 |  0.0 | 0.0 |





```
drugTweets_df
```

|    |   tweet_id        | text_text                                        | twitter_user_id | abuse |
|:--:|:-----------------:|:------------------------------------------------:|:---------------:|:-----:|
|0   |333644914913079296 | i know for a FACT that alcohol does not deplet...| 2300521710      |   0   |
|1   |344825926342832128 | mg Seroquel with my ‘normal’ mg Lovan and mcg ...| 179074771       |   0   |
|2   |344606561873833985 | there's a fella on my Facebook who is asking t...| 464202509       |   1   |



```python
tfidf_df['abused'] = drugTweets_df.abuse
```

### 4.Modeling

Data set for modeling is made through text pre-processing and creating a tf-idf matrix. In this part, we will focus on how to make classfication models.

#### 1.Import classifiers

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

#### 2.Load DTM File and split as train & test

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

|Abused| Number of Tweets|
|:----:|:---------------:|
|  0   |      2534       |
|  1   |       441       |


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
```


|Abused|Number of Tweets|
|:----:|:--------------:|
|   0  |      2029      |
|   1  |      2029      |


SMOTE (X_train_SMOTE, y_train_SMOTE)


```python
X_train_SMOTE, y_train_SMOTE = SMOTE().fit_resample(X_train, y_train)
```

|Abused|Number of Tweets|
|:----:|:--------------:|
|   0  |      2029      |
|   1  |      2029      |



Undersampling (X_train_under, y_train_under)


```python
rus = RandomUnderSampler(random_state = 123)
X_train_under, y_train_under = rus.fit_resample(X_train, y_train)
```

|Abused|Number of Tweets|
|:----:|:--------------:|
|   0  |       351      |
|   1  |       351      |


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

KNN

| Train | F1-Score | Precision | Recall | Accuracy | Test | F1-Score | Precision | Recall | Accuracy |
|:-----:|:--------:|:---------:|:------:|:--------:|:----:|:--------:|:---------:|:------:|:--------:|
| 0     | 0.82     | 0.96      | 0.71   | 0.84     | 0    | 0.73     | 0.87      | 0.63   | 0.61     |
| 1     | 0.86     | 0.77      | 0.97   |          | 1    | 0.27     | 0.19      | 0.48   |          |

SVM_Linear

| Train | F1-Score | Precision | Recall | Accuracy | Test | F1-Score | Precision | Recall | Accuracy |
|:-----:|:--------:|:---------:|:------:|:--------:|:----:|:--------:|:---------:|:------:|:--------:|
| 0     | 0.70     | 0.72      | 0.68   | 0.71     | 0    | 0.75     | 0.92      | 0.63   | 0.64     |
| 1     | 0.72     | 0.70      | 0.74   |          | 1    | 0.36     | 0.25      | 0.69   |          |

Decision_Tree

| Train | F1-Score | Precision | Recall | Accuracy | Test | F1-Score | Precision | Recall | Accuracy |
|:-----:|:--------:|:---------:|:------:|:--------:|:----:|:--------:|:---------:|:------:|:--------:|
| 0     | 0.95     | 0.96      | 0.94   | 0.95     | 0    | 0.85     | 0.87      | 0.83   | 0.75     |
| 1     | 0.95     | 0.95      | 0.96   |          | 1    | 0.28     | 0.25      | 0.28   |          |

NaiveBayesian

| Train | F1-Score | Precision | Recall | Accuracy | Test | F1-Score | Precision | Recall | Accuracy |
|:-----:|:--------:|:---------:|:------:|:--------:|:----:|:--------:|:---------:|:------:|:--------:|
| 0     | 0.59     | 0.78      | 0.48   | 0.67     | 0    | 0.59     | 0.90      | 0.44   | 0.48     |
| 1     | 0.72     | 0.62      | 0.86   |          | 1    | 0.29     | 0.18      | 0.71   |          |


Undersampling Result


```python
_models_under = fit_models(X_train_under, y_train_under)
train_report_under, train_matrix_under = compare_models(_models_under, X_train_under, y_train_under)
test_report_under, test_matrix_under = compare_models(_models_under, X_test, y_test)
```

KNN

| &nbsp;| F1-Score | Precision | Recall | Accuracy |
|:-----:|:--------:|:---------:|:------:|:--------:|
| Train | &nbsp;   | &nbsp;    |&nbsp;  | &nbsp;   |
| 0     | 0.71     | 0.72      | 0.70   | 0.72     |
| 1     | 0.72     | 0.71      | 0.93   |          |
| Test  | &nbsp;   | &nbsp;    |&nbsp;  | &nbsp;   |
|  0    | 0.65     | 0.89      | 0.51   | 0.53     |
|  1    | 0.30     | 0.19      | 0.66   |          |
SVM_Linear

| Train | F1-Score | Precision | Recall | Accuracy | Test | F1-Score | Precision | Recall | Accuracy |
|:-----:|:--------:|:---------:|:------:|:--------:|:----:|:--------:|:---------:|:------:|:--------:|
| 0     | 0.74     | 0.78      | 0.71   | 0.75     | 0    | 0.69     | 0.92      | 0.55   | 0.58     |
| 1     | 0.77     | 0.73      | 0.80   |          | 1    | 0.35     | 0.23      | 0.76   |          |

Decision_Tree

| Train | F1-Score | Precision | Recall | Accuracy | Test | F1-Score | Precision | Recall | Accuracy |
|:-----:|:--------:|:---------:|:------:|:--------:|:----:|:--------:|:---------:|:------:|:--------:|
| 0     | 0.96     | 0.95      | 0.96   | 0.96     | 0    | 0.70     | 0.90      | 0.58   | 0.59     |
| 1     | 0.96     | 0.96      | 0.95   |          | 1    | 0.32     | 0.21      | 0.63   |          |

NaiveBayesian

| Train | F1-Score | Precision | Recall | Accuracy | Test | F1-Score | Precision | Recall | Accuracy |
|:-----:|:--------:|:---------:|:------:|:--------:|:----:|:--------:|:---------:|:------:|:--------:|
| 0     | 0.73     | 0.74      | 0.72   | 0.74     | 0    | 0.70     | 0.82      | 0.56   | 0.58     |
| 1     | 0.74     | 0.73      | 0.75   |          | 1    | 0.34     | 0.22      | 0.71   |          |


SMOTE Result


```python
_models_SMOTE = fit_models(X_train_SMOTE, y_train_SMOTE)
train_report_SMOTE, train_matrix_SMOTE = compare_models(_models_SMOTE, X_train_SMOTE, y_train_SMOTE)
test_report_SMOTE, test_matrix_SMOTE = compare_models(_models_SMOTE, X_test, y_test)
```

KNN

| Train | F1-Score | Precision | Recall | Accuracy | Test | F1-Score | Precision | Recall | Accuracy |
|:-----:|:--------:|:---------:|:------:|:--------:|:----:|:--------:|:---------:|:------:|:--------:|
| 0     | 0.78     | 0.95      | 0.67   | 0.82     | 0    | 0.69     | 0.89      | 0.56   | 0.57     |
| 1     | 0.84     | 0.74      | 0.97   |          | 1    | 0.30     | 0.20      | 0.60   |          |

SVM_Linear

| Train | F1-Score | Precision | Recall | Accuracy | Test | F1-Score | Precision | Recall | Accuracy |
|:-----:|:--------:|:---------:|:------:|:--------:|:----:|:--------:|:---------:|:------:|:--------:|
| 0     | 0.73     | 0.77      | 0.69   | 0.74     | 0    | 0.76     | 0.92      | 0.65   | 0.66     |
| 1     | 0.76     | 0.72      | 0.80   |          | 1    | 0.37     | 0.26      | 0.68   |          |

Decision_Tree

| Train | F1-Score | Precision | Recall | Accuracy | Test | F1-Score | Precision | Recall | Accuracy |
|:-----:|:--------:|:---------:|:------:|:--------:|:----:|:--------:|:---------:|:------:|:--------:|
| 0     | 0.97     | 0.97      | 0.96   | 0.97     | 0    | 0.86     | 0.87      | 0.85   | 0.76     |
| 1     | 0.97     | 0.96      | 0.97   |          | 1    | 0.25     | 0.24      | 0.27   |          |

NaiveBayesian

| Train | F1-Score | Precision | Recall | Accuracy | Test | F1-Score | Precision | Recall | Accuracy |
|:-----:|:--------:|:---------:|:------:|:--------:|:----:|:--------:|:---------:|:------:|:--------:|
| 0     | 0.66     | 0.81      | 0.56   | 0.71     | 0    | 0.66     | 0.89      | 0.52   | 0.54     |
| 1     | 0.75     | 0.66      | 0.87   |          | 1    | 0.30     | 0.19      | 0.64   |          |


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

```

| Train | F1-Score | Precision | Recall | Accuracy |
|:-----:|:--------:|:---------:|:------:|:--------:|
| 0     | 0.77     | 0.92      | 0.66   | 0.66     |
| 1     | 0.37     | 0.26      | 0.67   |          |


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

|Classifiers   |AUC       |
|:------------:|:--------:|
| KNN          | 0.608702 |
| SVM          | 0.715259 |
| Decision Tree| 0.565886 |
| Gausian NB   | 0.601980 |


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

