import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from scipy.stats import pearsonr, zscore

# Class of dataframe to be treated for imputation evaluation

class ImputationEvaluation:

    def __init__(self, name, file_path, df):
        self.name=name
        self.file_path = file_path
        self.df = pd.read_csv(file_path, index_col=0)

        #remove the column original missingness from the imputed dataset if there is one
        if 'original_missingness' in self.df.columns:
            self.original_missingness = self.df['original_missingness']
            self.df = self.df.drop(columns=['original_missingness'])
        
        try:
            self.get_overlap(df)
        except:
            self.df.set_index(df.index)
        


    #select only overlapping cell lines and genes
    def get_overlap(self, df_compare):
        overlap_proteins = list( self.df.index.intersection(df_compare.index) )
        overlap_cell_lines = list( self.df.columns.intersection(df_compare.columns) )
        self.df = self.df.loc[overlap_proteins, overlap_cell_lines]
        df_compare = df_compare.loc[overlap_proteins, overlap_cell_lines]
        return df_compare

    #get difference between two dataframes
    def get_difference(self, df_compare):
        df_difference = self.df - df_compare
        df_difference = df_difference.abs()
        return df_difference.mean().mean()
    
    #normalize the data
    def normalization(self):
        self.df_normalized= self.df.apply(zscore, axis=1, nan_policy='omit')
        return self.df_normalized

    #get the correlation between two dataframes
    def get_correlation(self, df_compare, axis_value=1, method_corr='pearson'):
        df_compare = self.get_overlap(df_compare)
        df_compare = df_compare.apply(zscore, axis=1, nan_policy='omit')
        self.normalization()
        correlation = self.df.corrwith(df_compare, method_corr=method,  axis=axis_value, drop=True)
        return correlation.mean()

    #get the mean of the dataframe
    def get_mean(self):
        return self.df.mean().mean()
    
    #get a cluster map of correlations between dataframes
    def get_cluster_map(self, df_compare_array, method_corr='pearson', axis_value=1):        
        #get matrix of correlations between all dataframes in the array
        correlation_array = self.get_correlation_array(df_compare_array, method_corr=method_corr, axis_value=axis_value)
        correlation_array = np.array(correlation_array)
        correlation_array = correlation_array.reshape(len(df_compare_array), len(df_compare_array))
        #get the cluster map of the correlation matrix
        cluster_map = sns.clustermap(correlation_array, cmap='coolwarm', vmin=-1, vmax=1, center=0, annot=True, fmt='.2f')
        return cluster_map
    
    def get_correlation_array(self, df_compare_array, method_corr='pearson', axis_value=1):
        correlation_array = []
        for df_line in df_compare_array:
            for df_compare in df_compare_array:
                correlation = df_line.get_correlation(df_compare, method_corr=method_corr, axis_value=axis_value)
                correlation_array.append(correlation)
        return correlation_array

    #obtain some graphs in the following file path
    def get_graphs(self, file_path, df_compare_array, method_corr='pearson', axis_value=1):
        cluster_map = self.get_cluster_map(file_path)
        cluster_map.savefig(file_path + 'cluster_map.png')

        return 1


# Importing the initial data
df = pd.read_csv('/data/benchmarks/clines/proteomics.csv', index_col=0)
print(df.shape)

#Importing the data with the MVs imputed
df_imputed = pd.read_csv('/home/jorgeribeiro/JorgeRibeiroThesis/results/missForest_10_1000/proteomics_imputed_maxitter10_ntree1000_replaceT_decreasingT.csv',index_col=0)
print(df_imputed.shape)
df_MF_30_1000 = pd.read_csv('/home/jorgeribeiro/JorgeRibeiroThesis/results/missForest_30_1000_nogenesymbol/proteomics_imputed_maxitter30_ntree1000_replaceT_decreasingT.csv',index_col=0)
print(df_MF_30_1000.shape)
df_gain= pd.read_csv('/home/jorgeribeiro/JorgeRibeiroThesis/GAIN_prots/results/imputed_data_proteomics_missrate_0.0_batchsize_128_hintrate_0.9_alpha_100.0_iterations_10000.csv', index_col=0)
print(df_gain.shape)
df_gain_alpha_10= pd.read_csv('/home/jorgeribeiro/JorgeRibeiroThesis/GAIN_prots/results/imputed_data_proteomics_missrate_0.0_batchsize_128_hintrate_0.9_alpha_10.0_iterations_10000.csv', index_col=0)
print(df_gain_alpha_10.shape)
df_gain_alpha_0_1= pd.read_csv('/home/jorgeribeiro/JorgeRibeiroThesis/GAIN_prots/results/imputed_data_proteomics_missrate_0.0_batchsize_128_hintrate_0.9_alpha_0.1_iterations_10000.csv', index_col=0)
print(df_gain_alpha_0_1.shape)
df_gain_alpha_1000= pd.read_csv('/home/jorgeribeiro/JorgeRibeiroThesis/GAIN_prots/results/imputed_data_proteomics_missrate_0.0_batchsize_128_hintrate_0.9_alpha_1000.0_iterations_10000.csv', index_col=0)
df_compare = pd.read_csv('/data/benchmarks/clines/proteomics_ccle.csv', index_col=0)
print(df_compare.shape)



#MAKE THE COLUMNS GeneSymbol THE INDEX OF THE DATAFRAME
df_imputed = df_imputed.set_index(df.index) 
df_MF_30_1000 = df_MF_30_1000.set_index(df.index)

#get a list of the overlapping genes between df and df_compare
overlap_proteins = list( df.index.intersection(df_compare.index) )

#get a list of overlapping cell lines between df and df_compare
overlap_cell_lines = list( df.columns.intersection(df_compare.columns) )

#select only the overlapping cell lines and genes
df = df.loc[overlap_proteins, overlap_cell_lines]
df_imputed = df_imputed.loc[overlap_proteins, overlap_cell_lines]
df_compare = df_compare.loc[overlap_proteins, overlap_cell_lines]
df_gain = df_gain.loc[overlap_proteins, overlap_cell_lines]
df_gain_alpha_10 = df_gain_alpha_10.loc[overlap_proteins, overlap_cell_lines]
df_gain_alpha_0_1 = df_gain_alpha_0_1.loc[overlap_proteins, overlap_cell_lines]
df_gain_alpha_1000 = df_gain_alpha_1000.loc[overlap_proteins, overlap_cell_lines]
df_MF_30_1000 = df_MF_30_1000.loc[overlap_proteins, overlap_cell_lines]



# Check if both dataframes have the same shape if not create the intersection of the two dataframes
# if df.shape != df_imputed.shape:    
#     df = df[df.index.isin(df_imputed.index)]
#     df_imputed = df_imputed[df_imputed.index.isin(df.index)]
#     print('The dataframes have different shapes')

#calculate the average difference in values between datasets
df_difference = df - df_gain
df_difference = df_difference.abs()
df_difference = df_difference.mean(axis=1)
print(df_difference)

# #print the average absolute value of the difference between the original and imputed dataset
# df_difference = df_difference.abs()
# df_difference = df_difference.mean(axis=1)
# print(df_difference)


#get df_compare from zscore
df_compare=zscore(df_compare, axis=1, nan_policy='omit')
df=zscore(df, axis=1, nan_policy='omit')
df_imputed=zscore(df_imputed, axis=1, nan_policy='omit')
df_gain=zscore(df_gain, axis=1, nan_policy='omit')
df_gain_alpha_10=zscore(df_gain_alpha_10, axis=1, nan_policy='omit')
df_gain_alpha_0_1=zscore(df_gain_alpha_0_1, axis=1, nan_policy='omit')
df_gain_alpha_1000=zscore(df_gain_alpha_1000, axis=1, nan_policy='omit')
df_MF_30_1000=zscore(df_MF_30_1000, axis=1, nan_policy='omit')



# obtain the correlation between both dataframes
print('original')
print('by rows, by genes')
df_correlation = df.corrwith(df_compare, axis=1)
print(df_correlation.mean())

print('by columns, by samples')
df_correlation = df.corrwith(df_compare, axis=0)
print(df_correlation.mean())

#obtain the mean correlation between both dataframes 
print('imputed')
print('by rows, by genes')
df_correlation = df_imputed.corrwith(df_compare, axis=1)
print(df_correlation.mean())

print('by columns, by samples')
df_correlation = df_imputed.corrwith(df_compare, axis=0)
print(df_correlation.mean())

#obtain the mean correlation between both dataframes
print('gain')
print('by rows, by genes')
df_correlation = df_gain.corrwith(df_compare, axis=1)
print(df_correlation.mean())
print('by columns, by samples')
df_correlation = df_gain.corrwith(df_compare, axis=0)
print(df_correlation.mean())

#obtain the mean correlation between both dataframes
print('gain alpha 10')
print('by rows, by genes')
df_correlation = df_gain_alpha_10.corrwith(df_compare, axis=1)
print(df_correlation.mean())
print('by columns, by samples')
df_correlation = df_gain_alpha_10.corrwith(df_compare, axis=0)
print(df_correlation.mean())

#obtain the mean correlation between both dataframes
print('gain alpha 0.1')
print('by rows, by genes')
df_correlation = df_gain_alpha_0_1.corrwith(df_compare, axis=1)
print(df_correlation.mean())
print('by columns, by samples')
df_correlation = df_gain_alpha_0_1.corrwith(df_compare, axis=0)
print(df_correlation.mean())

#obtain the mean correlation between both dataframes
print('gain alpha 1000')
print('by rows, by genes')
df_correlation = df_gain_alpha_1000.corrwith(df_compare, axis=1)
print(df_correlation.mean())
print('by columns, by samples')
df_correlation = df_gain_alpha_1000.corrwith(df_compare, axis=0)
print(df_correlation.mean())

#obtain the mean correlation between both dataframes
print('MF 30 1000')
print('by rows, by genes')
df_correlation = df_MF_30_1000.corrwith(df_compare, axis=1)
print(df_correlation.mean())
print('by columns, by samples')
df_correlation = df_MF_30_1000.corrwith(df_compare, axis=0)
print(df_correlation.mean())



#sklearn.metrics.mean_squared_error search this too
#correlation map seaborn cluster map
#calcular mse

#metrica OOB
#OOB by samples or by genes





