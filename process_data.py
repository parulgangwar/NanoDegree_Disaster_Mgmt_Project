import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    This function is to load dataframe from filepaths
    INPUT: messages.csv, categories.csv
    OUTPUT: df
    """
    
    # This step is to load messages dataset
    messages = pd.read_csv(messages_filepath)
    # This step is to load categories dataset
    categories = pd.read_csv(categories_filepath)
    # This step is to merge messages and categories datasets
    df = messages.merge(categories, on="id")
    
    return df

def clean_data(df):
    """
    This function is to include Clean data in DataFrame and to transform categories
    INPUT: df
    OUTPUT: df
    """
    # This step is to create a dataframe of the individual column of categories
    categories = df.categories.str.split(";", expand=True)
    # This step is to get the first row of the categories dataframe
    row = categories.loc[0]

    # This step is to extract a list of new column names for categories.
    category_colnames = row.apply(lambda i: i[:-2])
    # This step is to rename the columns of 'categories'
    categories.columns = category_colnames
    for column in categories:
        # This step is to set each value to be the last character of the string
        categories[column] =  categories[column].str[-1]

        # This step is to convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    # This step is to drop the original categories column from 'df'
    df = df.drop(columns = 'categories') 
    # This step is to concatenate the original dataframe with the new 'categories' dataframe
    df = df.join(categories)
    # This step is to drop duplicates
    df = df.drop_duplicates()
    
    return df


def save_data(df, database_filename):
    """
    This function is to store a data frame in a SQLite database
    Args: df is a data frame
        table_name is name of the table
        database_filename is the name of the SQLite database file
    
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterResponse', engine , index = False, if_exists='replace')



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