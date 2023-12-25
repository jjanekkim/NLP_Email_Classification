# Spam Email Classification

## Overview
This repository hosts an email classification project to distinguish between spam and non-spam (ham) emails. The project leverages natural language processing techniques, specifically employing count vectorization and TF-iDF vectorization methodologies.

## Table of Contents
* [Introduction](#introduction)
* [General Info](#general-info)
* [Dependencies](#dependencies)
* [Project Structure](#project-structure)
* [Utilization](#utilization)

## Introduction

The Spam Email Classification Project aims to develop a robust model for identifying and filtering spam emails. In today's digital landscape, emails play a pivotal role in communication, yet distinguishing between important and less relevant content remains a challenge. Leveraging natural language processing (NLP) techniques, this project seeks to create a solution that efficiently filters unwanted emails while preserving essential communications.

## General Info

In this project, I leveraged NLTK libraries to perform a sequence of natural language processing (NLP) techniques, including tokenization, stop-word removal, and lemmatization. Further, I utilized transformative methodologies such as count vectorization and TF-iDF vectorization to convert textual data into numerical representations.

Upon preparing the finalized dataset for model training, I employed logistic regression and XGBoost classifier algorithms to classify spam and ham emails. To streamline the training process, I focused on the top 1000 words from the dataset.

The resulting model showcased commendable performance, achieving an AUC score of 0.93 for the logistic regression model and an impressive 0.95 for the XGBoost classifier. These scores underscore the model's efficacy in effectively discerning between spam and ham emails.

## Dependencies
This project is created with:
- Python version: 3.11.3
- Pandas package version: 2.0.1
- Matplotlib package version: 3.7.1
- Seaborn package version: 0.12.2
- re module version: 2.2.1
- NLTK library version: 3.8.1
- Scikit-learn package version: 1.2.2

## Project Structure
- **Data**: Access the dataset [here](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).
- **SpamEmail_Classification**: This notebook encompasses comprehensive data analysis, preparation, and model training for spam email classification.
- **spam_package**: Python file encapsulates all the essential functions for the pipeline.
- **spamemail_pipeline**: This serves as the project's pipeline, utilizing the functionalities from the 'spam_package.py' file.
  
## Utilization
To utilize this project, please download the dataset from the provided link mentioned above. Subsequently, download the 'spam_package.py' file and execute it in Jupyter Notebook.
