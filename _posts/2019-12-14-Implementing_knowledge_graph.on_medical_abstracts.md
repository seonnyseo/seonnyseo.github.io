---
layout: post
title:  "Implementing Knowledge Graph on Medical Abstracts"
date:   2019-12-14 23:59:00 +0100
categories: Neo4j KnowledgeGraph NLP NoSQL
---

#### Abstract

Extracting relevant data and information from large datasets and research paper can be a complex and time-consuming process. Moreover, being able to narrow down contextual information can be challenging because of the disconnect between various datasets and research findings. In this project, I have explored the usage of knowledge graph in medical context. The enriched knowledge graph I have built is capable of finding relevant relationships among anatomical components such as genes/proteins/diseases with drugs and drug types. Applying queries can help in narrowing down vast number of abstracts into only the relevant ones in a short amount of time. Medical researchers can use these outputs to do further research and detailed analysis.


#### Introduction

The data and research outputs created by the biomedical field are largely fragmented and stored in databases that typically do not connect to each other. To make sense of the data and to generate insights, we have to transform and organize it into knowledge by connecting information. This will be possible through knowledge graphs, which are contextualized data with logical structure imposed on data through entities and relationships. Using knowledge graphs will help retrieve data faster through connected graphs, provide contextual relationships to understand anatomical components and drug interaction better and assist in making research more efficient and economical through interpretation and exploration of insights. The output of this project will help the pharmaceutical company attain these goals.


#### Scope

This project focused on designing knowledge graphs to be used for research and faster retrieval of information and insights. Techniques of natural language processing and concept enrichment have been used to create the most information rich knowledge graphs. To understand and explore its use in research, this project connects and creates insights from research abstracts, available through the PubMed database. The objective of the project is to find relationship among protein targets, genes, disease mechanisms and drugs in these abstracts by using various processes in Neo4j.


#### Methodology

![Methodology](https://i.imgur.com/29PUo58.jpg)

This project utilized Neo4j as the graph database and GraphAware as the framework. The dataset containing 1146 abstracts from the PubMed database has been used to find relationships among the word tags. The ontology document contains focus entities and their codes. This project concentrated mostly on the five groups shown in the table below. The entities document contains phrases that are present in the abstracts and their related entities from the ontology document. In order to connect these data together, we indexed the date. We created Index on Abstract id (data corpus), Code (Ontology), Value (TERMite Response).


#### Annotation

![Annotation](https://i.imgur.com/9bAP6c7.jpg)

Annotation is the first step in building knowledge graph and is the process of breaking down the abstracts by tokenizing the words using the GraphAware NLP framework. Each abstract is separated by sentences and sentences divided into tags (words). During the process all the stop words are removed, and the remaining words are assigned their parts of speech. In this way, we create a hierarchy of three nodes: Annotated Text, Sentence and Tag and two relationships: Contain_sentence and Has_tag.
Annotating the abstracts is important for us to do further graph-based analysis. The simplest form of graph created through this process helps us point out the relevant keywords and sentences present in the vast list of abstracts.


#### Occurs With

![Occurs With](https://i.imgur.com/7hENIbR.jpg)

After the tags are extracted from the abstracts, it is helpful to look at keywords that frequently occur together. Often lot of times, it is possible to build relationships among two keywords by looking at their number of occurrences together. In this process, we create a function that runs through the keywords and counts the number of occurrences of each of the two keywords. For instance, in the sentence: Among genes related to milk fat synthesis and lipid droplet formation, only LPIN1 and dgat1 were upregulated by ad-nsrepb1, LPIN1, dfat, upregulate and ad-nsrepb1 occur together. It will then try to find these occurrences in other extracts.

This process counts the frequency of occurrences and builds the occurs_with relationship. It helps in building the graph and to understand the phrases. The frequencies of these occurrences can also be used to figure out the relevant relationships among keywords.
