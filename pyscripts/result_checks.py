import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from scipy.stats import pearsonr, zscore
import datetime

#import a function from another file
sys.path.insert(0, '/home/jorgeribeiro/JorgeRibeiroThesis/GAIN_prots')
from utils import get_hour_day


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
            self.df.set_index(df.index, inplace=True)

        
    #get difference between two dataframes
    def get_difference(self, df_compare):
        df_compare = self.get_overlap(df_compare)
        df_difference = self.df - df_compare
        df_difference = df_difference.abs()
        return df_difference.mean().mean()
    
    #normalize the data
    def normalization(self):
        self.df_normalized= self.df.apply(zscore, axis=1, nan_policy='omit')
        return self.df_normalized


    #select only overlapping cell lines and genes
    #changes the df attribute (probably not the best way to do it)
    def get_overlap(self, df_compare):
        overlap_proteins = list( self.df.index.intersection(df_compare.index) )
        overlap_cell_lines = list( self.df.columns.intersection(df_compare.columns) )
        self.df = self.df.loc[overlap_proteins, overlap_cell_lines]
        df_compare = df_compare.loc[overlap_proteins, overlap_cell_lines]
        return df_compare

    #get the correlation between two dataframes
    def get_correlation(self, IV_compare, axis_value=0, method_corr='pearson'):
        df_compare = self.get_overlap(IV_compare.df)
        df_compare = df_compare.apply(zscore, axis=1, nan_policy='omit')
        self.normalization()
        correlation = self.df_normalized.corrwith(df_compare, axis=axis_value)
        return correlation.mean()

    #get the mean of the dataframe
    def get_mean(self):
        return self.df.mean().mean()
    
    #get a cluster map of correlations between dataframes
    def get_cluster_map(self, df_compare_array, method_corr='pearson', axis_value=0):        
        #get matrix of correlations between all dataframes in the array
        correlation_array = self.get_correlation_array(df_compare_array, method_corr=method_corr, axis_value=axis_value)
        #correlation_array = np.array(correlation_array)
        #correlation_array = correlation_array.reshape(len(df_compare_array), len(df_compare_array))
        correlation_array = correlation_array.astype(float)
        print(correlation_array)
        #sort the 
        #get the cluster map of the correlation matrix with legend from dataframe correlation_array
        cluster_map = sns.clustermap(correlation_array, cmap="coolwarm", vmin=-1, vmax=1, center=0, annot=True, fmt=".2f", linewidths=.75)
        #cluster_map = sns.clustermap(correlation_array, cmap='mako', vmin=-1, vmax=1, center=0, annot=True, fmt='.2f')
        return cluster_map
    
    def get_correlation_array(self, df_compare_array, method_corr='pearson', axis_value=0):

        df_compare_array.sort(key=lambda x: x.name)
        correlation_array = pd.DataFrame(index=[IV.name for IV in df_compare_array], columns=[IV.name for IV in df_compare_array])

        #get the correlation between all dataframes in the array and input it in the correlation_array
        for IV_line in df_compare_array:
            for IV_compare in df_compare_array:
                correlation = IV_line.get_correlation(IV_compare, method_corr=method_corr, axis_value=axis_value)
                correlation_array.loc[IV_line.name, IV_compare.name] = correlation

        #get the correlation between all dataframes in the array
        # for IV_line in df_compare_array:
        #     for IV_compare in df_compare_array:
        #         correlation = IV_line.get_correlation(IV_compare, method_corr=method_corr, axis_value=axis_value)
        #         correlation_array.append(correlation)
        return correlation_array

    #obtain some graphs in the following file path
    def get_graphs(self, file_path, df_compare_array, method_corr='pearson', axis_value=0):
        cluster_map = self.get_cluster_map(file_path)
        #cluster_map.savefig(file_path + 'cluster_map.png')

        return 1

    def get_transcriptomics_plot(self, transcript_file_path, original_df):

        #create a directory to save the plots if it didnt exist
        now = get_hour_day(datetime.datetime.now())
        directory = '/home/jorgeribeiro/JorgeRibeiroThesis/results/' + now + '_' + self.name
        if not os.path.exists(directory):
            os.makedirs(directory)

        transcriptomics= pd.read_csv(transcript_file_path, index_col=0)

        #select only overlapping cell lines and genes
        overlap_proteins = list( self.df.index.intersection(transcriptomics.index) )
        overlap_cell_lines = list( self.df.columns.intersection(transcriptomics.columns) )
        df = self.df.loc[overlap_proteins, overlap_cell_lines]
        transcriptomics = transcriptomics.loc[overlap_proteins, overlap_cell_lines]
        original_df = original_df.loc[overlap_proteins, overlap_cell_lines]


        #get a plot of the transcriptomics and proteomics values of missing and non missing values in the original dataset
        #get the missingness of the original dataset
        missingness = original_df.isnull()
        missingness = missingness.astype(int)

        #get a dataset with the transcriptomics values of the missing values in the original dataset
        transcriptomics_missing = transcriptomics[missingness==1]
        print(transcriptomics_missing.shape)

        #join the protemics values of the missing values in the original dataset to the transcriptomics_missing dataset
        df_missing = df[missingness==1]
        transcriptomics_missing = pd.concat([transcriptomics_missing, df_missing], axis=1)
        

        #plot the missing values with their transcriptomics values in x axis and proteomics values in y axis

        #save some variables
        transcriptomics_wide = transcriptomics
        df_wide = df
        missingness_wide = missingness


        #transform transcriptomics and df into long format and add a column with the missingness
        transcriptomics = transcriptomics.stack().reset_index()
        transcriptomics.columns = ['GeneSymbol', 'CellLine', 'transcriptomics']
        df = df.stack().reset_index()
        df.columns = ['GeneSymbol', 'CellLine', 'proteomics']
        missingness = missingness.stack().reset_index()
        missingness.columns = ['GeneSymbol', 'CellLine', 'missingness']

        #check if the indexes of the three dataframes are the same
        print(transcriptomics.index.equals(df.index))
        print(transcriptomics.index.equals(missingness.index))

        #add the proteomics column to the transcriptomics dataframe
        transcriptomics['proteomics'] = df['proteomics']
        
        # cut the dataset to only the missing values
        transcriptomics_missing = transcriptomics[missingness['missingness']==1]
        print('number of points',transcriptomics_missing.shape)


        #remove the cell line column
        transcriptomics_missing_mean = transcriptomics_missing.drop(columns=['CellLine'])

        #average the transcriptomics and proteomics values of the missing values per gene
        transcriptomics_missing_mean = transcriptomics_missing_mean.groupby('GeneSymbol').mean()
        transcriptomics_missing_mean = transcriptomics_missing_mean.reset_index()

        
        transcriptomics_non_missing = transcriptomics[missingness['missingness']==0]
        transcriptomics_non_missing_mean = transcriptomics_non_missing.drop(columns=['CellLine'])
        transcriptomics_non_missing_mean = transcriptomics_non_missing_mean.groupby('GeneSymbol').mean()
        transcriptomics_non_missing_mean = transcriptomics_non_missing_mean.reset_index()

        #get an histogram of the transcriptomics values
        plt.hist(transcriptomics_missing_mean['transcriptomics'], bins=20)
        plt.xlabel('transcriptomics')
        plt.ylabel('frequency')
        plt.savefig(directory+'/histogram_transcriptomics'+'.png')
        plt.close()
        plt.clf()

        #get an histogram of the proteomics values
        plt.hist(transcriptomics_missing_mean['proteomics'], bins=20)
        plt.xlabel('proteomics')
        plt.ylabel('frequency')
        plt.savefig(directory+'/histogram_proteomics'+'.png')
        plt.close()
        plt.clf()

        #get boxplots of the proteomic values after dividing the transcriptomics values in 10 bins
        transcriptomics_missing_mean['transcriptomics_bins'] = pd.cut(transcriptomics_missing_mean['transcriptomics'], bins=10)
        sns.boxplot(x='transcriptomics_bins', y='proteomics', data=transcriptomics_missing_mean)
        plt.xticks(rotation=45)
        plt.xlabel('transcriptomics')
        plt.ylabel('proteomics')
        plt.gcf().set_size_inches(15, 10)
        plt.savefig(directory + '/boxplot_missing_'+'.png')
        plt.close()
        plt.clf()

        #get boxplots of the proteomics values after dividing the transcriptmocis values in 10 bins for the non missing values
        transcriptomics_non_missing_mean['transcriptomics_bins'] = pd.cut(transcriptomics_non_missing_mean['transcriptomics'], bins=10)
        sns.boxplot(x='transcriptomics_bins', y='proteomics', data=transcriptomics_non_missing_mean)
        plt.xticks(rotation=45)
        plt.xlabel('transcriptomics')
        plt.ylabel('proteomics')
        plt.gcf().set_size_inches(15, 10)
        plt.savefig(directory + '/boxplot_non_missing_'+'.png')
        plt.close()
        plt.clf()


        #get grouped boxplots of the proteomics values after dividing the transcriptomics values in 10 bins for the missing values and non missin values
        transcriptomics_missing_mean['missingness'] = 'missing'
        transcriptomics_non_missing_mean['missingness'] = 'non_missing'
        transcriptomics_mean = pd.concat([transcriptomics_missing_mean, transcriptomics_non_missing_mean], axis=0)
        transcriptomics_mean['transcriptomics_bins'] = pd.cut(transcriptomics_mean['transcriptomics'], bins=10)
        sns.boxplot(x='transcriptomics_bins', y='proteomics', hue='missingness', data=transcriptomics_mean)
        plt.xticks(rotation=45)
        plt.xlabel('transcriptomics')
        plt.ylabel('proteomics')
        plt.gcf().set_size_inches(15, 10)
        plt.savefig(directory + '/boxplot_missing_non_missing_'+'.png')
        plt.close()
        plt.clf()

        #get a grouped boxplot of the proteomics values after dividing the transcriptomics values in 10 bins for the missing values and non missing values with qcut for the missing values
        transcriptomics_missing_mean['transcriptomics_bins'] = pd.qcut(transcriptomics_missing_mean['transcriptomics'], q=10)
        transcriptomics_mean = pd.concat([transcriptomics_missing_mean, transcriptomics_non_missing_mean], axis=0)
        transcriptomics_mean['transcriptomics_bins'] = pd.qcut(transcriptomics_mean['transcriptomics'], q=10)
        sns.boxplot(x='transcriptomics_bins', y='proteomics', hue='missingness', data=transcriptomics_mean)
        plt.xticks(rotation=45)
        plt.xlabel('transcriptomics')
        plt.ylabel('proteomics')
        plt.gcf().set_size_inches(15, 10)
        plt.savefig(directory + '/boxplot_missing_non_missing_qcut_'+'.png')
        plt.close()
        plt.clf()

        #get a grouped violinplot with split violins of the proteomics values after dividing the transcriptomics values in 10 bins for the missing values and non missing values with qcut for the missing values
        transcriptomics_missing_mean['transcriptomics_bins'] = pd.qcut(transcriptomics_missing_mean['transcriptomics'], q=10)
        transcriptomics_mean = pd.concat([transcriptomics_missing_mean, transcriptomics_non_missing_mean], axis=0)
        transcriptomics_mean['transcriptomics_bins'] = pd.qcut(transcriptomics_mean['transcriptomics'], q=10)
        sns.violinplot(x='transcriptomics_bins', y='proteomics', hue='missingness', data=transcriptomics_mean, split=True, fill= False, inner='quart')
        plt.xticks(rotation=45)
        plt.xlabel('transcriptomics')
        plt.ylabel('proteomics')
        plt.gcf().set_size_inches(15, 10)
        plt.savefig(directory + '/violinplot_missing_non_missing_qcut_'+'.png')
        plt.close()
        plt.clf()









        #plot the mean transcriptomics values with the mean proteomics values pre and post imputation









# Importing the initial data
df = pd.read_csv('/data/benchmarks/clines/proteomics.csv', index_col=0)

#Importing data for ImputationEvaluation class and make an array of that class
df_MF_10_1000 = ImputationEvaluation('df_MF_10_1000', '/home/jorgeribeiro/JorgeRibeiroThesis/results/missForest_10_1000/proteomics_imputed_maxitter10_ntree1000_replaceT_decreasingT.csv', df)
df_MF_20_2000 = ImputationEvaluation('df_MF_20_2000', '/home/jorgeribeiro/JorgeRibeiroThesis/results/missForest_20_2000/proteomics_imputed_maxitter20_ntree2000_replaceT_decreasingT.csv', df)
df_MF_30_1000 = ImputationEvaluation('df_MF_30_1000', '/home/jorgeribeiro/JorgeRibeiroThesis/results/missForest_30_1000_nogenesymbol/proteomics_imputed_maxitter30_ntree1000_replaceT_decreasingT.csv', df)
df_gain= ImputationEvaluation('df_gain', '/home/jorgeribeiro/JorgeRibeiroThesis/GAIN_prots/results/imputed_data_proteomics_missrate_0.0_batchsize_128_hintrate_0.9_alpha_100.0_iterations_10000.csv', df)
df_gain_alpha_10= ImputationEvaluation('df_gain_alpha_10', '/home/jorgeribeiro/JorgeRibeiroThesis/GAIN_prots/results/imputed_data_proteomics_missrate_0.0_batchsize_128_hintrate_0.9_alpha_10.0_iterations_10000.csv', df)
df_gain_alpha_0_1= ImputationEvaluation('df_gain_alpha_0_1', '/home/jorgeribeiro/JorgeRibeiroThesis/GAIN_prots/results/imputed_data_proteomics_missrate_0.0_batchsize_128_hintrate_0.9_alpha_0.1_iterations_10000.csv', df)
df_gain_alpha_1000= ImputationEvaluation('df_gain_alpha_1000', '/home/jorgeribeiro/JorgeRibeiroThesis/GAIN_prots/results/imputed_data_proteomics_missrate_0.0_batchsize_128_hintrate_0.9_alpha_1000.0_iterations_10000.csv', df)
df_compare = ImputationEvaluation('df_compare', '/data/benchmarks/clines/proteomics_ccle.csv', df)
df_original = ImputationEvaluation('df_original', '/data/benchmarks/clines/proteomics.csv', df)
df_gain_hr_0_6_mr_0_1= ImputationEvaluation('df_gain_hr_0_6_mr_0_1','/home/jorgeribeiro/JorgeRibeiroThesis/GAIN_prots/results/imputed_data_proteomics_missrate_0.1_batchsize_128_hintrate_0.6_alpha_100.0_iterations_10000.csv', df)
df_gain_hr_0_6= ImputationEvaluation('df_gain_hr_0_6','/home/jorgeribeiro/JorgeRibeiroThesis/GAIN_prots/results/imputed_16h_14Sep_proteomics_rmse_nan_missrate_0.0_batchsize_128_hintrate_0.6_alpha_100.0_iterations_10000.csv', df)
df_gain_hr_0_6_mr_0_01= ImputationEvaluation('df_gain_hr_0_6_mr_0_01','/home/jorgeribeiro/JorgeRibeiroThesis/GAIN_prots/results/imputed_22h_14Sep_proteomics_rmse_0.0627_missrate_0.01_batchsize_128_hintrate_0.6_alpha_100.0_iterations_10000.csv', df)
df_gain_mr_0_1= ImputationEvaluation('df_gain_mr_0_1','/home/jorgeribeiro/JorgeRibeiroThesis/GAIN_prots/results/imputed_data_proteomics_missrate_0.1_batchsize_128_hintrate_0.9_alpha_100.0_iterations_10000.csv', df)
df_gain_hr_0_0_mr0_01= ImputationEvaluation('df_gain_hr_0_0_mr0_01','/home/jorgeribeiro/JorgeRibeiroThesis/GAIN_prots/results/imputed_0h_18Sep_proteomics_rmse_0.058_missrate_0.01_batchsize_128_hintrate_0.0_alpha_100.0_iterations_10000.csv', df)
df_replace_compare = ImputationEvaluation('df_replace_compare', '/home/jorgeribeiro/JorgeRibeiroThesis/results/MVs_replace_df_compare.csv', df)
df_mean_rows = ImputationEvaluation('df_mean_rows', '/home/jorgeribeiro/JorgeRibeiroThesis/results/mean_rows.csv', df)
df_mean_columns = ImputationEvaluation('df_mean_columns', '/home/jorgeribeiro/JorgeRibeiroThesis/results/mean_column.csv', df)
df_VAE = ImputationEvaluation('df_VAE', '/home/jorgeribeiro/JorgeRibeiroThesis/Proteomics/proteomicsVAE.csv', df)
df_optuna_mr0_1= ImputationEvaluation('df_optuna_mr0_1', '/home/jorgeribeiro/JorgeRibeiroThesis/GAIN_prots/results/imputed_18h_27Sep_proteomics_rmse_0.0461_missrate_0.1_batchsize_512_hintrate_0.2_alpha_1000.0_iterations_1000.csv', df)

df_optuna_17_02= ImputationEvaluation('df_optuna_17_02', '/home/jorgeribeiro/JorgeRibeiroThesis/GAIN_prots/results/imputed_17h_2Oct_proteomics_rmse_nan_missrate_0.0_batchsize_128_hintrate_0.14_alpha_1000.0_iterations_1000.csv', df)

#make an array of the dataframes to be compared
df_compare_array = [df_original, df_gain, df_compare, df_MF_10_1000, df_MF_20_2000, df_MF_30_1000, df_gain_alpha_1000, df_gain_hr_0_6, df_replace_compare, df_mean_rows, df_VAE, df_optuna_17_02] 

#df_compare_array = [df_original, df_gain, df_compare, df_MF_10_1000, df_MF_20_2000, df_MF_30_1000, df_gain_alpha_0_1, df_gain_alpha_10, df_gain_alpha_1000, df_gain_hr_0_6_mr_0_1, df_gain_hr_0_6, df_gain_hr_0_6_mr_0_01, df_gain_hr_0_0_mr0_01, df_gain_mr_0_1, df_replace_compare, df_mean_rows, df_mean_columns, df_VAE, df_optuna_mr0_1, df_optuna_17_02] 
#obtain the overlap between the dataframes in the array to df_compare

# for IV in df_compare_array:
#     IV.get_overlap(df_compare.df)
#     print(IV.name)
#     print(IV.df.shape)

transcript_file_path = '/data/benchmarks/clines/transcriptomics.csv'

#obtain the plot of transcriptomics and proteomics values of missing and non missing values in the original dataset
df_optuna_17_02.get_transcriptomics_plot(transcript_file_path, df)
df_MF_30_1000.get_transcriptomics_plot(transcript_file_path, df)



#obtain the correlation array between the dataframes in the array
# correlation_array = df_original.get_correlation_array(df_compare_array)

# #obtain the cluster map of the correlation array
# cluster_map = df_compare.get_cluster_map(df_compare_array)
# cluster_map.savefig('/home/jorgeribeiro/JorgeRibeiroThesis/results/cluster_map'+ str(get_hour_day(datetime.datetime.now()))+'.png')





#Importing the data with the MVs imputed
# df_imputed = pd.read_csv('/home/jorgeribeiro/JorgeRibeiroThesis/results/missForest_10_1000/proteomics_imputed_maxitter10_ntree1000_replaceT_decreasingT.csv',index_col=0)
# df_MF_30_1000 = pd.read_csv('/home/jorgeribeiro/JorgeRibeiroThesis/results/missForest_30_1000_nogenesymbol/proteomics_imputed_maxitter30_ntree1000_replaceT_decreasingT.csv',index_col=0)
# df_gain= pd.read_csv('/home/jorgeribeiro/JorgeRibeiroThesis/GAIN_prots/results/imputed_data_proteomics_missrate_0.0_batchsize_128_hintrate_0.9_alpha_100.0_iterations_10000.csv', index_col=0)
# df_gain_alpha_10= pd.read_csv('/home/jorgeribeiro/JorgeRibeiroThesis/GAIN_prots/results/imputed_data_proteomics_missrate_0.0_batchsize_128_hintrate_0.9_alpha_10.0_iterations_10000.csv', index_col=0)
# df_gain_alpha_0_1= pd.read_csv('/home/jorgeribeiro/JorgeRibeiroThesis/GAIN_prots/results/imputed_data_proteomics_missrate_0.0_batchsize_128_hintrate_0.9_alpha_0.1_iterations_10000.csv', index_col=0)
# df_gain_alpha_1000= pd.read_csv('/home/jorgeribeiro/JorgeRibeiroThesis/GAIN_prots/results/imputed_data_proteomics_missrate_0.0_batchsize_128_hintrate_0.9_alpha_1000.0_iterations_10000.csv', index_col=0)
# df_compare = pd.read_csv('/data/benchmarks/clines/proteomics_ccle.csv', index_col=0)


# #MAKE THE COLUMNS GeneSymbol THE INDEX OF THE DATAFRAME
# df_imputed = df_imputed.set_index(df.index) 
# df_MF_30_1000 = df_MF_30_1000.set_index(df.index)

# #get a list of the overlapping genes between df and df_compare
# overlap_proteins = list( df.index.intersection(df_compare.index) )

# #get a list of overlapping cell lines between df and df_compare
# overlap_cell_lines = list( df.columns.intersection(df_compare.columns) )


# #select only the overlapping cell lines and genes
# df = df.loc[overlap_proteins, overlap_cell_lines]
# df_imputed = df_imputed.loc[overlap_proteins, overlap_cell_lines]
# df_compare = df_compare.loc[overlap_proteins, overlap_cell_lines]
# df_gain = df_gain.loc[overlap_proteins, overlap_cell_lines]
# df_gain_alpha_10 = df_gain_alpha_10.loc[overlap_proteins, overlap_cell_lines]
# df_gain_alpha_0_1 = df_gain_alpha_0_1.loc[overlap_proteins, overlap_cell_lines]
# df_gain_alpha_1000 = df_gain_alpha_1000.loc[overlap_proteins, overlap_cell_lines]
# df_MF_30_1000 = df_MF_30_1000.loc[overlap_proteins, overlap_cell_lines]


# #calculate the average difference in values between datasets
# df_difference = df - df_compare
# df_difference = df_difference.abs()
# df_difference = df_difference.mean(axis=1)
# print(df_difference.mean())

# #print the average absolute value of the difference between the original and imputed dataset
# df_difference = df_difference.abs()
# df_difference = df_difference.mean(axis=1)
# print(df_difference)


#get df_compare from zscore
# df_compare=zscore(df_compare, axis=1, nan_policy='omit')
# df=zscore(df, axis=1, nan_policy='omit')
# df_imputed=zscore(df_imputed, axis=1, nan_policy='omit')
# df_gain=zscore(df_gain, axis=1, nan_policy='omit')
# df_gain_alpha_10=zscore(df_gain_alpha_10, axis=1, nan_policy='omit')
# df_gain_alpha_0_1=zscore(df_gain_alpha_0_1, axis=1, nan_policy='omit')
# df_gain_alpha_1000=zscore(df_gain_alpha_1000, axis=1, nan_policy='omit')
# df_MF_30_1000=zscore(df_MF_30_1000, axis=1, nan_policy='omit')



# # obtain the correlation between both dataframes
# print('original')
# print('by rows, by genes')
# df_correlation = df.corrwith(df_compare, axis=1)
# print(df_correlation.mean())

# print('by columns, by samples')
# df_correlation = df.corrwith(df_compare, axis=0)
# print(df_correlation.mean())


#sklearn.metrics.mean_squared_error search this too
#calcular mse





