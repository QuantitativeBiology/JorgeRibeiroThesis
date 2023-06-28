import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from scipy.stats import pearsonr, zscore

# Importing the initial data
df = pd.read_csv('/data/benchmarks/clines/proteomics.csv')

#Importing the data with the MVs imputed
df_imputed = pd.read_csv('/home/jorgeribeiro/JorgeRibeiroThesis/results/missForest_10_1000/proteomics_imputed_maxitter10_ntree1000_replaceT_decreasingT.csv')

df_compare = pd.read_csv('/data/benchmarks/clines/proteomics_ccle.csv', index_col=0)

#remove the column original missingness from the imputed dataset
df_imputed = df_imputed.drop(columns=['original_missingness'])

#remove the column Unamed:0 from the imputed dataset
df_imputed = df_imputed.drop(columns=['Unnamed: 0'])


#otain the column GeneSymbol from the original dataset
df_gene = df['GeneSymbol']

#MAKE THE COLUMNS GeneSymbol THE INDEX OF THE DATAFRAME
df_imputed = df_imputed.set_index(df_gene)  
df = df.set_index('GeneSymbol')

#get a list of the overlapping genes between df and df_compare
overlap_proteins = list( df.index.intersection(df_compare.index) )

#get a list of overlapping cell lines between df and df_compare
overlap_cell_lines = list( df.columns.intersection(df_compare.columns) )

#select only the overlapping cell lines and genes
df = df.loc[overlap_proteins, overlap_cell_lines]
df_imputed = df_imputed.loc[overlap_proteins, overlap_cell_lines]
df_compare = df_compare.loc[overlap_proteins, overlap_cell_lines]


# Check if both dataframes have the same shape if not create the intersection of the two dataframes
if df.shape != df_imputed.shape:    
    df = df[df.index.isin(df_imputed.index)]
    df_imputed = df_imputed[df_imputed.index.isin(df.index)]
    print('The dataframes have different shapes')

#calculate the average difference in values between datasets
df_difference = df - df_imputed
df_difference = df_difference.abs()
df_difference = df_difference.mean(axis=1)
print(df_difference)

#get the missingness matrix of the original dataset
df_missingness = df.isnull()
print(df_missingness)

#get a matrix of the different values between the original and imputed dataset
df_difference = df - df_imputed
print(df_difference)

# #print the average absolute value of the difference between the original and imputed dataset
# df_difference = df_difference.abs()
# df_difference = df_difference.mean(axis=1)
# print(df_difference)


#get df_compare from zscore
df_compare=zscore(df_compare, axis=1, nan_policy='omit')
df=zscore(df, axis=1, nan_policy='omit')
df_imputed=zscore(df_imputed, axis=1, nan_policy='omit')


# obtain the correlation between both dataframes
print('by rows, by genes')
df_correlation = df.corrwith(df_compare, axis=1)
print(df_correlation.mean())

print('by columns, by samples')
df_correlation = df.corrwith(df_compare, axis=0)
print(df_correlation.mean())

#obtain the mean correlation between both dataframes 
print('by rows, by genes')
df_correlation = df_imputed.corrwith(df_compare, axis=1)
print(df_correlation.mean())

print('by columns, by samples')
df_correlation = df_imputed.corrwith(df_compare, axis=0)
print(df_correlation.mean())





#get spearman correlation
print('by rows, by genes')
df_correlation = df_imputed.corrwith(df_compare, axis=1, method='spearman')#search for the method
#explorar diferentes metricas de correlacao
#sklearn.metrics.mean_squared_error search this too
#correlation map seaborn cluster map
#calcular mse
#pedir miguel a matriz de proteomics by vae para comparacap

#metrica OOB
#OOB by samples or by genes





