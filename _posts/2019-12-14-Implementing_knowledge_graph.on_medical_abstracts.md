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


#### Keyword Extraction

![Keyword Extraction](https://i.imgur.com/gHjF9SF.jpg)

In this process, the words or phrases that best describe the abstract are identified and selected using the Term Frequency – Inverse Document Frequency (TF-IDF). So, in any particular abstract, the term that appears the most can be used as the simple summary of the context of the abstract. The tags that are the most relevant in the abstracts are stored in the node keyword. The relationship describes is created to link the annotated text and keywords.

This process can play an important role in the simplification of the summary for abstracts. It saves time for researchers by narrowing down the abstracts by only looking at the relevant keywords. Further queries can be created as per the need of researchers.


#### Concept Enrichment

![Concept Enrichment](https://i.imgur.com/aHt3AE7.jpg)

To build a knowledge graph, we also need to extract the hidden structure of our textual data and organize it so that a machine can process it. Sometimes, the data that we have is inadequate for making any kind of process. It also might not be the right set of data that is needed. Therefore, the enrichment process allows us to extend the knowledge, thereby introducing external sources. This process allows to gain more insight at the end of the process. We used external knowledge bases
ConceptNet 5 and Microsoft Concept which offers the ability to enrich entities.

For instance, the enrichment can discover that Fibrosis is a degenerative change, but that it is also a chromic complication. We can also create new connections into the graph between the documents. For example, if we take a document with Fibrosis and a document with, nephropathy there is nothing that the entity recognizes that will relate the two. However, by using external knowledge, we can create a new connection with Fibrosis, and nephropathy with Chronicle Complication as link between two. This will automatically enrich our graph, share connected data, and increase knowledge about your documents.


#### TextRank Summarization

![TextRank Summarization](https://i.imgur.com/FG0OzyZ.jpg)

Similar approach to the keyword extraction can be employed to implement simple summarization of an abstract. A densely connected graph of sentences is created, with Sentence-Sentence relationships representing their similarity based on shared words (number of shared words vs sum of logarithms of number of words in a sentence). PageRank is then used as a centrality measure to rank the relative importance of sentences in the document.

For a given annotate id we can get the list of sentences with their relevance ranking. In output below for annotatedText id 247474 sentence 7 and 3 are the most important with text rank .178 and .170 respectively.


#### Cosine Similarity

![Cosine Similarity](https://i.imgur.com/JZ1VSiT.jpg)

Once tags are extracted from all the abstracts, it is possible to compute similarities between them using content-based similarity. During this process, each annotated text is described using the TF- IDF encoding format. TF-IDF. Text documents can be TF-IDF encoded as vectors in a multidimensional Euclidean space. The space dimensions correspond to the tags, previously extracted from the documents. The coordinates of a given document in each dimension are calculated as a product of two sub-measures: term frequency and inverse document frequency.

Similarity between two or more abstracts can be calculated using Cosine similarity. Cosine similarity value range from 0 to 1. Abstracts having similarity cosine value near to 1 are most similar in meaning.


#### Result and Insights

With all of methodology above, we have tried to draw results from two perspectives. First, we want to exploit the methods that we worked on the methodology fully. It means we must find a way to weave the processes and draw insights. Second, we consider the processes in the researcher at the company's view and think how to take an advantage to a graph database. The graph database defines relationships between nodes and supports to search based on the relationships. We think this is an advantage of this database compared with structure database and programming languages.

Neo4j provides visualization of graphs that helps users to explore the database, but it displays complicated visuals in most cases, not efficient to deliver insights and consumes lots of computer resources. In order to avoid this issue, we write two queries that receive keywords or abstract ids as an input and show text as a result. As a result, we realize that the graph database can be used like Wikipedia. Users can surf interesting abstracts by keywords or similar abstracts in our queries. We will discuss how the model works.

##### Keyword to Abstracts

![Keyword to Abstracts](https://i.imgur.com/z4aCaBJ.jpg)

Keyword Searching Concept

![Query](https://i.imgur.com/HMmKnrw.jpg)

This query receives a keyword that user interests and displays a list of most relevant sentences in each abstract and ids of the abstracts that contain the keyword. The query only searches keywords that were extracted from the Keyword Extraction process above, not seeking the entire words in the abstracts. The most relevant sentences are chosen based on the frequency of terms shared in each abstract by TextRank Summarization process. For example, if we search the keyword ’NASH’ in the query, it returns 8 sentences and ids. We expect users save time to search and read articles at this point.


##### Abstract to Abstracts

![Abstract to Abstracgs](https://i.imgur.com/JBWWiC5.jpg)

![Result Table](https://i.imgur.com/wwiqkYJ.jpg)

The other query that we made is for pulling information from similar abstracts that are close to the user's interesting abstract. So, this query receives an abstract id as an input argument. The similarity value is calculated by the Cosine Similarity process, so each abstract has similarity value with every other abstract in the database. According to this reason, we limit the number of presenting abstracts. The reason that we devise this query is we think the researcher wants to surf similar abstracts to gather more information. Thus, the query helps shorten the time to search the abstracts.

![Result Visual](https://i.imgur.com/OhSR2e4.jpg)