**Project 2: Classifying Messages for Disaster Response**

Pipeline for an ETL process and code for the ML-model to classify messages during a disaster. A web app is also produced to visualise the training data.

**Installation**

This code has a number of required dependencies which have to be pre-installed to work. They include:

- numpy
- pandas
- nltlk
- sklearn
- sqlalchemy
- pickle

The two datasets required are attached in this repository (data/disaster_categories.csv and data/disaster_messages.csv).

**Motivation** 

The motivation behind this project was classify messages sent during a disaster. By doing this, the response can be significantly improved and streamlined, allowing the most vulnreable to be found quicker. 

**Files**

- process_data.py - ETL pipeline to merge and clean the data used in this project.
- train_classifier.py - The code used to build the ML-model.
- run.py - The code to run the web app.

**Usage**

To use the files, run the following commands;
- "python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db"
- "python models/train_classifier.py data/DisasterResponse.db models/classifier"
- "cd app"
- "python run.py"

**Author**

All code created by Rhodri Evans.

**Acknowledegments**

Thanks to UDACITY for helping me develop my data science skill set.
