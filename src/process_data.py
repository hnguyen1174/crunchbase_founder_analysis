import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

if __name__ == "__main__":

    # Read the dataset
    df = pd.read_csv('../data/final_data.csv')
    df.drop('Unnamed: 0', axis = 1, inplace = True)

    # Replace the headquarter location with the frequency of that headquarter
    df1 = (df['Headquarters Location'].value_counts().reset_index(name='hq_frequency')
                                                    .rename(columns={'index': 'Headquarters Location'}))
    df = pd.merge(df, df1, on=['Headquarters Location'], how='left')

    # df_main contains the main numerical features that will go into our machine learning models
    df_main = df.drop(['Full_Name', 'Primary Job Title', 'Biography', 'Categories', 'Headquarters Location',
                    'Number of Lead Investments_x', 'Number of Partner Investments', 'Last Equity Funding Amount',
                        'Total Equity Funding Amount', 'IPO Status', 'other_dummy', 'Bachelors',
                    'Operating Status','Founded Date', 'Closed Date', 'Company Type', 'Funding Status', 'Last Funding Type',
                    'Last Equity Funding Type','Number of News Articles', 'Number of Employees', 'Acquisition Status',
                    'Last Funding Date', 'IPO Date', 'Number of Events_y'], axis = 1)

    # Gender: 1 is Male and 0 is Female
    df_main['Male'] = np.where(df_main['Gender'] == 'Male', 1, 0)
    df_main.drop('Gender', axis = 1, inplace = True)

    # Process last funding round:
    df_main['Last Funding Amount'] = df_main['Last Funding Amount'].str.extract('([0-9.,]+)')
    df_main['Last Funding Amount'] = df_main['Last Funding Amount'].str.replace(',', '')
    df_main['Last Funding Amount'] = df_main['Last Funding Amount'].astype(float)

    # Total Funding Amount
    df_main['Total Funding Amount'] = df_main['Total Funding Amount'].str.extract('([0-9.,]+)')
    df_main['Total Funding Amount'] = df_main['Total Funding Amount'].str.replace(',', '')
    df_main['Total Funding Amount'] = df_main['Total Funding Amount'].astype(float)

    # Min-Max Scaler
    sc = MinMaxScaler()
    scaled_values = sc.fit_transform(df_main) 
    df_main.loc[:,:] = scaled_values

    df_main.fillna(0,inplace=True)
    df_main.to_csv('../data/df_processed.csv')