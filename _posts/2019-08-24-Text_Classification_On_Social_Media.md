---
layout: post
title:  "Text Classification on Social Media Text"
date:   2019-08-24 23:59:00 +0100
categories: Python Twitter API MachineLearning Text Classification
---

This summer, I had a chance to work with one of the biggest pharmacist companies to evaluate the performance of machine learning models on detecting drug abuse in Tweets. Primarily, the company focused on assessing the BERT compared to other machine learning models. It was my first time to work on Natural Language Processing(NLP) and an excellent opportunity to come across new concepts.

I will share the explanation and result of the project below, and you can check out the Python code that I worked from [my github page](https://github.com/seonnyseo/text_classification_on_social_media/blob/master/text_classification_on_social_media.ipynb).

Before moving into the project, I would like to share what I have learned and the points where I should improve on in the future.

- NLP concepts (Lemmatization/Stemming, TF-IDF)
- Reviewed the Tweets before working on the methodology and found that several medications have two names. So I devised the Term replacement to unify the names. Once again, it reminded me of the value of looking at the data first.
- Reason to choose F-measure as a significant performance. In real business, I thought it was important to get the right answers right, but also to make fewer mistakes. In this regard, I thought f-measure with two balance would be suitable as a metric. I had thought about the metrics deeply and become familiar with it through the project.

- Insufficient backgrounds on Neural Network models and also BERT. I studied and experienced a bit of Deep Learning models, Keras, and BERT this time. It's been fun, and will put some effort into it.
- Overfitting on classification models. Although I cross-validated the models and run optimization, it seems the models were overfitted, especially in the case of the Decision Tree model. I will find room for improvement on this issue.

 

### Summary

Social media messaging is a significant resource to collect and understand human behavior. As this data is generated in abundant volumes on a minute basis, evaluating an automated technique to identify these behaviors would be very beneficial.  The goal of this project is to evaluate the performance of Google’s Bidirectional Encoder Representation of Transformers (BERT) in the classification of Social Media Texts. BERT was compared to four different classifiers: K-Nearest Neighbors (KNN), Support Vector Machine (SVM), Naïve Bayes, and Decision Tree (DT); and two neural network models – Recurrent Neural Networks (RNN) and Convolutional Neural Networks(CNN). Twitter posts associated with three commonly abused medications (Adderall, oxycodone, and quetiapine) were collected to classify users into those who are likely to abuse drugs and those who would not. BERT performed the best. Hyper-parameter tuning of BERT was performed to achieve the best results. 


### Introduction

![Intorduction](https://i.imgur.com/DmuiRrD.jpg)

The US is the world’s biggest abusers for prescribed drugs. According to the New York Times, American accounts for 80% of the world’s oxycodone consumption. It kills more people than car accidents every year (Ricks, 2019). The Centers for Disease Control and Prevention classifies it as an epidemic and the world health organization reported it threatens the achievements of modern medicine (Hoffman. J, 2019). The aim of this project is to correctly identify users that are likely to abuse drugs from the Twitter tweets. A total sample of 3017 tweets were collected, with 447 (15%) categorized as “abuse” and 2570 (85%) categorized as “not abuse”. The data was split into training and testing set in the 80:20 ratio respectively. 


### Data Overview

#### Pre-processing

![Pre-processing](https://i.imgur.com/MGpJKMI.jpg)

From the collected tweets no missing data was observed. Prior to dividing the dataset into training and testing, the text was pre-processed. Since the tweets were written in the informal language in most cases, and contained reserved words related to Twitter, we applied several steps to clean the tweets. First, we used python twitter-preprocessor library to erase Twitter reserved words, URLs, and emojis. Second, ‘stop words’ are the words that appear the most in the English vocabulary. These do not provide important meaning and are therefore removed. In our study, we treated  words like “Seroquel” and “seroquel” in the same way therefore to avoid our model from recognizing these two words as two different things, we lowercased our text data. Based on the same idea, we used ‘stemming’ to reduce the number of terms based on grammatical inflections where the end of a words are removed to bring the words to a common base form. Next, upon reading the tweets, we found ‘Seroquel’ and ‘Oxycodone’ are also called ‘Quetiapine’ and ‘Oxycontin’, respectively. These are the same medications differed by generic and brand name, so we replaced the generic names with the brand names. Similarly, we replaced the word “red bull” with “redbull” to maintain consistency among the same words and removed characters such as “&amp” and “&lt” as they did not add meaning to the text.


#### Resampling

![Resampling](https://i.imgur.com/oMHMqQd.jpg)

Since we have an imbalanced dataset with 85% classified as class “not abuse”, and the remaining 15% classified as class “abuse”. To avoid the learning models from being biased towards the majority class, we tried 3 resampling methods – Naïve Random Oversampling and KNN based Synthetic Minority Oversampling Technique(SMOTE) to increase the “abuse” class to match the representation of the “not abuse” class and Naïve Random Undersampling to decrease the “not abuse” class to match the less-represented class, “abuse”. We applied all three resampling methods to the dataset with a minimum of 20-, 30- and 50-word frequency each and then performed the below 4 classification models to each scenario to compare model performance and determine the resampling method and the appropriate minimum frequency (Appendix: Table 1). SMOTE was the chosen method of resampling.

#### Feature Selection

![Feature Selection](https://i.imgur.com/CbfeVxk.jpg)

To analyze our results after performing stemming, we converted our text data into a document-term representation. Followed by tokenizing and counting the frequency of our terms. However, mere frequency count does not equate to feature importance, so we applied an alternate weighting schemes to better the important terms through Term Frequency-Inverse Document Frequency (tf-idf). We then created three datasets where only the terms that appeared a minimum of 20, 30 and 50 times respectively would be included. With the minimum term appearance of 20, 226 features were retained for a minimum of 30 terms, 137 features were retained, and with a minimum of 50 term appearances, 67 features were retained. 


### Classification Models

The data was split into the training and testing set in the 80:20 ratio,  respectively. On the training set, four supervised classification models were performed to determine the best classification technique: KNN, Naïve Bayes, SVM, and Decision Tree. We focused on a multitude of performance metrics to determine the best classification model, mainly the F1-measure, recall value, and the model accuracy of the testing set. The recall is the ratio of correctly predicted positive observations to all the observations in the actual class. F1-measure,  on the other hand, takes both false positives and false negatives into account. This is especially helpful because our class distribution is uneven. And finally, the model accuracy, which is the ratio of correctly predicted observation to the total observations. For the F1-measure and recall measures, we mainly focused on the values for the “abuse” class since we wanted to focus on the model being able to correctly predict and identify the tweets that are likely to be classified as “abuse” over “not abuse.”

#### Models

- KNN

KNN is a non-parametric technique that stores observations and classifies the cases based on the similarities between the observations. Euclidean distance was the chosen metric to measure the similarity, with k=10. Cross-validation was performed to determine the optimal k-value by using an independent dataset to validate the k value. The model achieved a training accuracy of 0.81 (F1-measure: 0.84 and recall: 0.97) and a testing accuracy of 0.56 (F1-measure: .31 and recall: .62). There is a clear issue of overfitting here.

- Naive Bayes

Naive Bayes computes conditional probability for every category and then selects the outcome with the highest probability, where it assumes independence of predictor variables. The model achieved a training accuracy of 0.69 (F1- measure: 0.74 and recall: 0.90) and a testing accuracy of 0.51 (F1- measure: .33 and recall: .75). Overfitting exists here as well.

- SVM

“Linear” kernel was used to search if it would create the optimal hyperplane with the largest margin of separation between the two classes. The model achieved a training accuracy of 0.74 (F1- measure: 0.75 and recall: 0.79) and a testing accuracy of 0.67 (F1- measure: .40 and recall: .69). Overfitting exists here as well.

- Decision Tree

Decision Tree does not require the assumption of linearity and is apt for this case due. The model achieved a training accuracy of 0.96 (F1-measure: 0.96 and recall: 0.97) and a testing accuracy of 0.74 (F1- measure: .23 and recall: .24). Overfitting exists here as well. 


#### Evaluation

![Training](https://i.imgur.com/SSuxsKd.jpg)

![Testing](https://i.imgur.com/VqEq3EF.jpg)

Upon comparing the four classification model (Table 1 below, Appendix: Graph 1) SVM model with the minimum word frequency of 30, resampled with the SMOTE method has the highest F1-measure of 0.4, recall of 0.69, and the test accuracy of 0.67. Out of 98 “abuse” tweets, SVM correctly predicts 47 (48%) as “abuse” and out of 506 “not abuse” tweets, 287 (57%) are correctly predicted. 

![Classification Models ROC Curve](https://imgur.com/lPfaMqJ)

#### Validation and Optimization

We performed 10-fold cross-validation on the SMOTE and original datasets to combat the issue of overfitting and to validate the model. A large difference in the mean score of F1-measure between the original un-resampled dataset (F1- measure mean: .357 and standard deviation: .346) and the SMOTE resampled dataset (F1- measure mean: .721; standard deviation: .233) was observed. This confirms that our resampled dataset performed much better. We further performed a GridSearch to tune the parameters of the SVM model by trying four different values for C (1,10,100,1000) for the kernel ‘linear’, while choosing the Stratified K-fold cross-validation option. The model is optimized with C=1000, with a F1- measure of .37, recall of .68 and accuracy of 0.66. This confirms our optimized model performance. 


### Neural Networks

- Tokenization

In our study, we tokenize the text of drug tweets by performing white space tokenization, removing punctuation, and converting to lowercase. Before tokenization, we set the size of our vocabulary to 5,000.

- Padding

Keras requires that all samples be of the same size. For this reason, we will need to pad some observations and (possibly) truncate others. If padding occurs, 0s are added to either the beginning or the end of the sequence to match the maximum length specified. In our study, we choose to add 0s to the end of the sequence. We set the maximum number of tokens to be the average plus 2 standard deviations, which we saved as an integer and selected 26 tokens at the end.

- Class-Weight

Our dataset is unbalanced. To address this problem, we used the parameter: class_weight = ’balanced” to consider the issue of imbalance. Using this parameter, we give different weights on labels and thus can balance “abuse” and “non-abuse” weight which will help us address the bias problem of our data.

#### Models

- RNN (Recurrent Neural Networks)

RNN are used for processing sequences of data, and they are a popular choice for predictive text input. In our RNN model we create a Sequential model first and added three configured layers: (1) Embedding layer (2) LSTM layer: the units argument is the number of neurons in the layer and we choose 16 first, (3) A total of 3 Dropout Layer and finally the (4) Dense Output layer. To compile the model, we specified the loss function: “Binary cross entropy”, optimizer: “Adam” and metrics as “Accuracy” to assess the goodness of our model and each epoch and overall. We performed various combinations of parameter tuning by changing the number of layers, the embedding size, number of epochs, and the learning rates, for both cased and uncased datasets. The best result on the testing dataset, generated a F1-measure of 0.40, recall of 0.60, and accuracy of 0.81.   

- CNN (Convolutional Neural Networks)

CNN also known as covnet, has been gaining traction in NLP classification. Like RNN, in our CNN model, we first created a Sequential model and then added five configured layers: (1) Embedding layer (2) Convolution layer (3) Pooling layer: A pooling layer, in this case “GlobalMaxPooling” was used to compress the results by discarding features, which helps to increase the generalizability of the model. Finally, like RNN, the (4) Dense Layer (with 100 units) and (5) Dense Output layer were added. Then we compile the model using the same loss function, optimizer and metrics to assess the goodness of our model and each epoch and overall. We fit the model by training the model on a sample of data and make predictions by using the model to generate predictions on new data. A F1-measure of 0.186 with a recall of 0.371was observed on the testing set. 

#### Evaluation

![NN ROC Curve](https://imgur.com/ZTilFJi)

Upon changing multiple parameters for CNN, our result with a batch size of 32 and epoch of 2 has a F1- measure of .391 for the abuse case, with a recall of .454 and accuracy of 0.773. A ROC curve was generated to compare the RNN and CNN results (Appendix A: Graph 3). RNN outperformed with an AUC of .649 in comparison to AUC of .603 for CNN.



### BERT

- Introduction

BERT is a bidirectional model. The Transformer encoder of BERT reads the entire sequence of words at once. The model learns from its surroundings (left and right of the word). Model of the main innovation, points in the pre - "train" method, which is the Masked LM (Mask out k% of the input words, and then predict the masked words, usually k = 15) and the Next Sentence Prediction, two methods respectively to catch phrases and Sentence level representation.

- Pre-trained model

In our study, we used two models-- cased model and uncased model--to pre-train our data, which makes our data well fitted into BERT model. The uncased model processes the data in lowercase format and the cased model leaves the text in its original form, that is a combination of upper- and lower-case words.

- Hyper parameter tuning

We performed a total of 144 combinations of parameters from Batch sizes of [8, 16, 32, 64, 128, 256], Epoch sizes of [3, 4, 5], Learning Rates of [2e-5, 3e-5, 4e-5, 5e-5] and text Case of [‘Uncased’, ‘Cased’]. Our best result, of Batch Size 8, an Epoch of 4, Learning Rate of 2e-5 and Uncased data generated a F1-measure of 0.485, recall of 0.576, accuracy of 0.797 and AUC of 0.708



### Result

![Result Comparison](https://imgur.com/kxjEQxK)

Google’s BERT model outperforms others (Appendix: Graph 4) in identifying users that are likely to abuse drugs based on Twitter posts, with a f1-measure of 0.485, recall of 0.576 and accuracy of 0.797. Our study indicates that social media--Twitter-- can be an important resource for detecting the prevalence of abuse of prescription medications, and that BERT can help us complete the task of identifying potential abuse-indicating user posts. 
