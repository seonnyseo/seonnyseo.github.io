---
layout: post
title:  "R Markdown test"
date:   2019-01-11 23:00:40 +0100
categories: r jekyll
---
  
As a group project, I practiced sentiment analysis on music streaming services reviews. Purpose of the project is to find insights from the reviews, and I focused on disclosing relationships between events of the services and user reactions. It can be software updates, offline events, or increasing membership fee. For checking this out, I wanted to see review trends by a word, period and sentiment. The reviews that I handled for the project are from Pandora, Spotify and Amazon music and the total sum of those is about 39,290 reviews

1. Manual Scoring
2. Resampling
3. Extract Data
4. Findings
5. Limitations

### Manual Scoring

For practicing sentiment analysis, I had to decide which lexicon(sentiment dictionary) should I use for the analysis. R provides lexicons such as ‘Bing’, ‘NRC’, and ‘AFINN’. I chose ‘bing’ for the project. This decision is based on comparisons between manual judgments and lexicon scores. The manual judgement datasets are about 1% of total reviews and were selected randomly. Here is the result of the comparison.

Spotify
![Imgur](https://imgur.com/Ik7PDMT.png)

Pandora
![Imgur](https://imgur.com/0h6tjXA.png)

Amazon music
![Imgur](https://imgur.com/t1PQfof.png)


‘Bing’ determined the highest accuracy compared two lexicons on every service. Somehow, the conclusions on negative reviews were shallow, but we couldn’t find out a solution this time. (I assume this is related to the length of reviews and sarcasm) 

Also, another interesting point of the result is neutral reviews. The proportion of neutral reviews are around 14% on every services review. I pondered how to resolve this because 14% of data is not small. Then, I decided to deal this with probability, because the sentiment trend of keywords is the only thing that I want to know. I am not interested in each review. 

### Resampling

I have 1% of manually scored reviews. Now, let’s think this as a population and practice resampling. When we have enough number of samples, we can extract samples from it and infer a normal distribution. So, I extract 50 reviews randomly from the manually scored samples multiple times and deduce proportions of positive and negative in neutral reviews. 

Here is an R code for resampling and a result of Amazon music.


{% highlight r %}
library(tidytext)
library(readr)
library(dplyr)
library(anytime)

# Load manually scored review files here. ("../service_scored.csv")
service <- read_csv("https://raw.githubusercontent.com/seonnyseo/Streaming_Sentiment_Analysis/master/Data/amazon_scored.csv")
  
# I decided to consider ambiguous reviews as negative
service$score <- ifelse(service$score == 0, -1, 1)
  
# Split reviews by each word 
tidy_service <- service %>% unnest_tokens(word, review)
  
bing_sentiments <- tidy_service %>% inner_join(get_sentiments("bing"), by = "word")
bing_sentiments$score <- ifelse(bing_sentiments$sentiment == "negative", -1, 1)
bing_aggregate <- bing_sentiments %>% 
  select(review_id, score) %>% 
  group_by(review_id) %>% 
  summarise(bing_score = sum(score))
  
score_compare_service <- merge(x = service, y = bing_aggregate, all.x = TRUE, by = 'review_id')
score_compare_service[is.na(score_compare_service)] <- 0
  

# score_compare_service(review_id, date, review, score, bing_score)
# Pick 50 each time 
resampling <- function(service){
  
  neutral_average <- 0
  postive_average <- 0
  negative_average <- 0
  
  for(i in c(2000:3000)){
    set.seed(i)
    random_data <- service[sample(nrow(service), 50),]
    
    neutral_count <- sum(random_data$bing_score == 0)
    positive_neutral <- sum(random_data$bing_score == 0 & random_data$score == 1)
    negative_neutral <- sum(random_data$bing_score == 0 & random_data$score == -1)
    
    neutral_average <- neutral_average + neutral_count/50
    postive_average <- postive_average + positive_neutral/neutral_count
    negative_average <- negative_average + negative_neutral/neutral_count
  }
  cat(sprintf("Neutral : %.3f  Positive : %.3f  Negative : %.3f\n", 
              neutral_average/1000, postive_average/1000, negative_average/1000))
}
  
# Run resampling
resampling(score_compare_service)
{% endhighlight %}



{% highlight text %}
## Neutral : 0.140  Positive : 0.710  Negative : 0.291
{% endhighlight %}


This is the results of resampling. (Neutral / (Positive|Neutral) / (Negative|Neutral)
  # Pandora	0.14	0.54	0.46
  # Spotify	0.13	0.59	0.41
  # Amazon	0.14	0.70	0.30

### Extract Data

My purpose is to see a change in trend of sentiment by keywords. Thus, expected outcome is a graph that shows time period in x axis and quantity of reviews by sentiment in y axis. R code is composed of three functions to implement a graph. 

1. Pre-Process
2. Extract reviews
3. Create a graph

#### Pre-Process
{% highlight r %}
pre_process <- function(service)
{
  # PreProcessing
  service$review_id <- 1:nrow(service)
  service$date <- anydate(service$date)
  service <- service[c(3,1,2)]
  
  # Sentiment Score
  tidy_service <- service %>% unnest_tokens(word, review)
  
  # Edit Dictionary, I only add 4 words with sentiment at this time. This can be expanded later. 
  bing_edit <- rbind(get_sentiments("bing"), c("commercial", "negative"))
  bing_edit <- rbind(bing_edit, c("commercials", "negative"))
  bing_edit <- rbind(bing_edit, c("ad", "negative"))
  bing_edit <- rbind(bing_edit, c("ads", "negative"))
  bing_edit <- rbind(bing_edit, c("wish", "negative"))
  
  bing_sentiments <- tidy_service %>% inner_join(bing_edit, by = "word")
  
  bing_sentiments$score <- ifelse(bing_sentiments$sentiment == "negative", -1, 1)
  bing_aggregate <- bing_sentiments %>% select(review_id, score) %>% group_by(review_id) %>% summarise(bing_score = sum(score))
  
  service <- merge(x = service, y = bing_aggregate, all.x = TRUE, by = 'review_id')
  service[is.na(service)] <- 0
  service$bing_judgement <- ifelse(service$bing_score > 0, "positive", 
                                   ifelse(service$bing_score < 0, "negative", "neutral" ))
  
  return(service)
}
{% endhighlight %}


Pre-Process function is called only once for each service dataset. On this step, the function cleans data and judge sentiment of reviews based on bing lexicon. Also, I edited bing dictionary by adding some words are not included in the dictionary such as ‘commercials’, ‘ad’, and ‘wish’. Those words connote negative emotion in many reviews, so I intend to treat these as negative words.

So, dataset acquires sentiment score column after pass this function.


Extract reviews

#### word_data
{% highlight r %}
word_data <-function(service, start, end, word, sentiment){

  word <- tolower(word)
  
  # Filter Data between start date & end date
  extracted <- service[service$date >= start & service$date <= end,]
  # Filter Date that only contains word
  extracted <- extracted[grepl(word, tolower(extracted$review)),]
  
 
  set.seed(101)

  # Neutral / (Positive|Neutral) / (Negative/Neutral)
  # Pandora	0.14	0.54	0.46
  # Spotify	0.13	0.59	0.41
  # Amazon	0.14	0.70	0.30
  
  ifelse(service$service == "pandora", positive_weight <- 0.54,
         ifelse(service$service == "spotify", positive_weight <- 0.59, positive_weight <- 0.70))
    
  neutral_reviews <- extracted[extracted$bing_judgement == "neutral",] %>% select(review_id)
  positive_neutral <- neutral_reviews[sample(nrow(neutral_reviews), nrow(neutral_reviews) * positive_weight),]
  negative_neutral <- neutral_reviews[!(neutral_reviews$review_id %in% positive_neutral),]
    
  extracted$bing_judgement <- ifelse(extracted$review_id %in% positive_neutral, "positive",
                                      ifelse(extracted$review_id %in% negative_neutral, "negative",
                                            extracted$bing_judgement))
  
  
 
  extracted <- extracted[extracted$bing_judgement == sentiment,]
  
  return(extracted)
}
{% endhighlight %}

This function does two operations. This function extracts data by service, period, word, and sentiment from pre-processed raw dataset. As I mentioned above, the extracted dataset should includes around 14% of neutral reviews. So, the function handles neutral reviews with estimated probability from resampling results. It splits neutral reviews as positive or negative randomly follow the weights.

#### frequnecy_month
{% highlight r %}
frequency_month <- function(service, start, end, word, sentiment){
  
  extracted <- word_data(service, start, end, word, sentiment)

  # Make year-month column
  extracted$year_month <- anydate(format(as.Date(extracted$date), "%Y-%m"))

  frequency_df <- extracted %>% group_by(year_month) %>% summarise(frequency = n())
  frequency_df <- frequency_df %>% pad(interval = 'month', start_val = anydate(start), end_val = anydate(end))
  frequency_df[is.na(frequency_df)] <- 0
  return (frequency_df)
}
{% endhighlight %}


 I only want to know frequency of reviews each months. This function counts how many reviews were posted in a certain period by keywords. 

#### word_graph
{% highlight r %}
word_graph <- function(service, word, start, end){

  positive = frequency_month(service, start, end, word, "positive")
  negative = frequency_month(service, start, end, word, "negative")
  
  positive$sentiment <- 'positive'
  negative$sentiment <- 'negative'
  
  frequency_df <- positive %>% full_join(negative)
  
  ret <- ggplot(frequency_df, aes(x = year_month)) +
        geom_line(aes(y = frequency, col = sentiment)) +
        theme(axis.title.x=element_blank())
  
  return(ret)
}
{% endhighlight %}

User calls this function to see results. 

