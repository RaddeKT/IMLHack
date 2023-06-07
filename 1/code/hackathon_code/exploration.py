import pandas as pd
import numpy as np
import plotly.express as px
import sklearn
import sys

df=pd.read_csv('datasets/agoda_cancellation_train.csv')
# print(df[filter(lambda x:x.split('_')[0]=='hotel',df.columns,)])


df=df.select_dtypes(float)
dd=df.corr().unstack().abs().sort_values()
print(dd[dd<1])

class Explore():
    def __init__(self,df : pd.DataFrame,result_name : str):
        self.Data = df
        self.res = result_name
        self.Num_Data= df[filter(lambda x:x.split('_')[0]!='cat',df.columns,)]


    def print_corr(self,only_numeric=True):

        if only_numeric:
            df=self.Num_Data
        else:
            df=self.Data
        for i in ['pearson', 'kendall', 'spearman']:
            print(f"**********************{i}********************************")
            print(f"{i} correlation for nomerical data")
            corr=(df.corr(i)[self.res]).sort_values()
            # print correlation of nomerical columns sorted
            print('correation to the resolts columns')
            print(corr)
        print("5 most correlated  features in absolute valies  ")
        all_corrs =df.corr().unstack().abs().sort_values()
        print(all_corrs[all_corrs <1])


    def plot_feats(self,feat1:str,freat2:str):
        """ plots two features"""
        px.scatter(self.Data,feat1,freat2).show()

    def threshold_check(self,feat:str,all:bool):
        pass






