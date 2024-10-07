import sys
import pandas as pd 
import re
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sqlalchemy import create_engine
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle

def load_data(database_filepath):
    ''' A function for loading the database from sql and setting parameters for the ML-model.
    Inputs:
    database_filepath: Path to database.
    Outputs:
    X : The message column.
    Y: The classications the model will predict.
    category_names: A list of the classifications.
    '''
    # Retrieve the database from SQL.
    engine = create_engine("sqlite:///"+ database_filepath)
    df = pd.read_sql_table(database_filepath, "sqlite:///"+ database_filepath)  
    # Set the X and Y parameters, along with retrieving a list of the classifications.
    X = df.message.values 
    Y = df.drop(columns= ['id', 'message','genre','original']) 
    category_names = Y.columns.tolist() 
    
    return X,Y, category_names

def tokenize(text):
    ''' A function for tokenizing the text.
    Inputs: 
    text - The text to be tokenized.
    Outputs:
    clean_tokens - The processed tokens.
    '''
    # Variable to identify URLs in text.
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+' 
    # Detect and replace URLs. 
    detected_urls= re.findall(url_regex, text) 
    for url in detected_urls:
        text= text.replace(url, "urlplaceholder") 
    # Tokenize the text.    
    tokens= word_tokenize(text) 
    # Set up the Lemmatizer.
    lemmatizer= WordNetLemmatizer() 
    # Clean the tokenized text.
    clean_tokens= [] 
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip() 
        clean_tokens.append(clean_tok)  
  
    return clean_tokens

def build_model():
    ''' A function to build the model.
    Outputs:
    cv- An optimized pipeline object.
    '''
    # Create a pipeline with transformers.
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ]) 
    # Go through parameters to optimize the pipeline with GridSearchCV.
    parameters= {
        'clf__estimator__n_estimators': [50, 100, 200],
        'clf__estimator__min_impurity_decrease':[0.0,0.25,0.5],
        'clf__estimator__max_features': [1,50,100],
        } 
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    ''' A function to output the accuracy of the model.
    Inputs:
    model = The fitted model.
    x_test= The unseen text data.
    y_test = The corresponding classification data.
    Outputs:
    A performace report on each category.
    '''
    # Predict with unseen data.
    y_pred= model.predict(X_test) 
    # Print performance of model for each classification.
    for category in range(len(category_names)):
        true_labels = Y_test.iloc[:, category]
        pred_labels = y_pred[:, category]
        report = classification_report(true_labels, pred_labels) 
        print(f"Category: {category_names[category]}")
        print(report)

def save_model(model, model_filepath):
    '''
    A function for saving the model.
    '''
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main()