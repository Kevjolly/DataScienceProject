import pandas as pd
'''
functions for loading data from csv
and getting basic attributes of data
'''


def csv_to_df(filename):
    df = pd.read_csv(filename, header=None)
    try: 
    	df.set_index(0, inplace=True)
    except:
    	df.set_index('3', inplace=True)	# if column names are used
    return df


def unique_classes(df):
    return df.index.unique().values


def mean_std_of_class(df):
    return df.mean(0), df.std(0)


def min_max_of_df(df):
    return df.min().min(), df.max().max()

