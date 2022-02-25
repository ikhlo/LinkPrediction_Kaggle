# Link Prediction Kaggle Challenge 

## Overview

The goal of this challenge was to predict if there exists a citation link between two papers (paper A cites B or paper B cites A). To do so, we had a graph citation network containing about hundreds of thousands papers (as vertices) and more than one million citation (as edges). In addition, for each paper we also had its abstract and a list of authors that have contributed.

## Collaboration

This project have been done in collaboration by [Ikhlass Yaya-Oyé](https://github.com/ikhlo), [Valerii Markin](https://github.com/ValeriyMarkin) and [César Leblanc](https://github.com/CesarLeblanc).

## This project is implemented in Python and gathers tasks of :
* Graph processing and features extraction (nodes embeddings, similarity, index)
* NLP Tasks as abstracts preprocessing (tag filtering, stopwords, stemming)
* Fine-tuning of abstracts embedding using BERT
* Graph Neural Network conception
  

## You will find in this repositery : 
* A [LaTeX report of our work](https://github.com/ikhlo/LinkPrediction_Kaggle/blob/main/ALTEGRAD_Report.pdf)
* The [available data](https://github.com/ikhlo/LinkPrediction_Kaggle/tree/main/data_challenge_2021)
* The [python file about gnn training](https://github.com/ikhlo/LinkPrediction_Kaggle/blob/main/code/train_gnn.py)
* The [main code](https://github.com/ikhlo/LinkPrediction_Kaggle/tree/main/code)
<br> 

## Conclusion : 

Our team, DeVinci, is ranked 4th on the public and private leaderboard as you can see [here](https://www.kaggle.com/c/altegrad-2021/leaderboard).