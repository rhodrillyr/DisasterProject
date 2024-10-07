import sys
import pandas as pd
import re
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    ''' A function for loading and merging the required dataframes. 
     Inputs: 
     messages_filepath: Contains the messages database.
     categories_filepath : Contains the categories database.
     Outputs:
     df: Merged database.
     '''
    # Read the two csv files as dataframes.
    messages= pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # Merge dataframes.
    df = messages.merge(categories, on="id")
    return df

def clean_data(df):
    ''' A function for cleaning the dataframes prior to further analysis. 
    Inputs:
    df = Merged dataframe from load_data.
    Outputs:
    df= Cleaned dataframe.
    '''
    # Split the category column on each semi colon.
    categories= df["categories"].str.split(";", expand= True)
    # Extract the category names.
    row = categories.iloc[0] 
    # Remove the binary values.
    category_colnames = row.apply(lambda x: x[:-2]) 
    # Re-name columns.
    categories.columns = category_colnames 
    # Sets the row value for each category to be binary.
    for column in categories:
        categories[column] = categories[column].apply(lambda x: re.sub(r'\D', '', str(x)))
        categories[column] = categories[column].astype(int) 
    # Drop original category column, attach new and drop any duplicates.   
    df.drop(columns=["categories"], inplace= True)
    df = pd.concat([df, categories], axis=1) 
    df= df.drop_duplicates() 
    return df

def save_data(df, database_filename):
    ''' A function for saving the dataframe as SQLite.
    Inputs: 
    df= Cleaned dataframe.
    database_filename = Name of new database.
    '''
    # Create engine and save dataframe to location. 
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql(database_filename, engine, index=False, if_exists='replace')   

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')

if __name__ == '__main__':
    main()