import pandas as pd
import numpy as np
# Assuming your original dataset is stored in a CSV file called 'data.csv'
data = pd.read_csv('/data/benchmarks/clines/proteomics.csv', index_col=0)

#get np array from the data
data_np = data.values

data_m = np.isnan(data_np)

#import imputed dataset
data_mf= pd.read_csv('/home/jorgeribeiro/JorgeRibeiroThesis/results/missForest_30_1000/proteomics_imputed_maxitter30_ntree1000_replaceT_decreasingT.csv', index_col=0)

#remove last column
data_mf = data_mf.iloc[:, :-1]

#import imputed dataset GAIN
data_gain = pd.read_csv('/home/jorgeribeiro/JorgeRibeiroThesis/GAIN_prots/results/imputed_17h_2Oct_proteomics_rmse_nan_missrate_0.0_batchsize_128_hintrate_0.14_alpha_1000.0_iterations_1000.csv', index_col=0)

#import transcriptomics dataset
data_transcriptomics = pd.read_csv('/data/benchmarks/clines/transcriptomics.csv', index_col=0)

#transcriptomics and proteomics have the same indexes, and cell lines, group them together in a dataset with columns GeneSymbol, CellLine, Proteomics, Transcriptomics
 #start by making each dataset a long format GeneSymbol, CellLine, Value
data_mf_long = data_mf.stack().reset_index()
data_mf_long.columns = [ 'GeneSymbol', 'CellLine', 'MF']
data_gain_long = data_gain.stack().reset_index()
data_gain_long.columns = [ 'GeneSymbol', 'CellLine', 'GAIN']
data_transcriptomics_long = data_transcriptomics.stack().reset_index()
data_transcriptomics_long.columns = [ 'GeneSymbol', 'CellLine', 'Transcriptomics']

#do the same for data and keep the lines with missing values
data_long= pd.melt(data.reset_index(), id_vars='GeneSymbol', var_name='CellLine', value_name='Proteomics')

#reorder rows so that they are in the same order
data_long = data_long.sort_values(by=['GeneSymbol', 'CellLine']).reset_index(drop=True)

#Ate aqui nice

#add a column to data_long with true or false if the value is missing
data_long['missing'] = data_m.flatten()

#merge the datasets
data_merged = pd.merge(data_long, data_mf_long, on=['GeneSymbol', 'CellLine'], how='left' )
data_merged = pd.merge(data_merged, data_gain_long, on=['GeneSymbol', 'CellLine'], how='left')
data_merged = pd.merge(data_merged, data_transcriptomics_long, on=['GeneSymbol', 'CellLine'], how='left', suffixes=('', '_transcriptomics'))


#Make a boxplot of the gain proteomics according to bins of transcriptomics
import seaborn as sns
import matplotlib.pyplot as plt

#make bins of transcriptomics with 4 decimal places
data_merged['Transcriptomics_bins'], bin_edges= pd.qcut(data_merged['Transcriptomics'], q=10, retbins=True, precision=4)

bin_edges = np.round(bin_edges, 4)

bin_labels = []
for i in range(len(bin_edges) - 1):
    bin_labels.append(f'{bin_edges[i]} - {bin_edges[i+1]}')

data_merged['Transcriptomics_bins'] = pd.qcut(data_merged['Transcriptomics'], q=10, labels=bin_labels)

#make boxplot
ax=sns.boxplot(x='Transcriptomics_bins', y='GAIN', data=data_merged, hue='missing')
handles, labels = ax.get_legend_handles_labels()
new_labels= ['non MVs', 'MVs']
ax.legend(handles, new_labels, title='Missing', loc='upper right', title_fontsize='large', fontsize='large')
plt.xticks(rotation=90)
plt.xlabel('Transcriptomics', fontsize='x-large')
plt.ylabel('Proteomics GAIN', fontsize='x-large')
plt.legend(title='Missing', title_fontsize='large', fontsize='large', loc='upper right')
#size of the plot
plt.gcf().set_size_inches(20, 15)
plt.savefig('/home/jorgeribeiro/JorgeRibeiroThesis/results/boxplot_transcriptomics_proteomics_gain.png')
plt.clf()


#make boxplot
ax=sns.boxplot(x='Transcriptomics_bins', y='MF', data=data_merged, hue='missing')
handles, labels = ax.get_legend_handles_labels()
new_labels= ['non MVs', 'MVs']
ax.legend(handles, new_labels, title='Missing', loc='upper right', title_fontsize='large', fontsize='large')
plt.xticks(rotation=90)
plt.xlabel('Transcriptomics', fontsize='x-large')
plt.ylabel('Proteomics MF', fontsize='x-large')
plt.legend(title='Missing', loc='upper right', title_fontsize='large', fontsize='large')
#size of the plot
plt.gcf().set_size_inches(20, 15)
plt.savefig('/home/jorgeribeiro/JorgeRibeiroThesis/results/boxplot_transcriptomics_proteomics_mf.png') 
plt.clf()

#make the dataset in long form with a column indicating the method or just proteomics if missing is false
data_merged_long = pd.melt(data_merged, id_vars=['GeneSymbol', 'CellLine', 'missing', 'Transcriptomics_bins'], value_vars=['GAIN', 'MF'], var_name='Method', value_name='Proteomics_method')
print(data_merged_long.head(20))


#make boxplots side by side of the values that have missing true for gain and mf
data_merged_missing = data_merged[data_merged['missing'] == True]

#make the dataset in long form with a column indicating the method
data_merged_missing_long = pd.melt(data_merged_missing, id_vars=['GeneSymbol', 'CellLine', 'missing', 'Transcriptomics_bins'], value_vars=['GAIN', 'MF'], var_name='Method', value_name='Proteomics_method')


#make boxplot
ax=sns.boxplot(x='Transcriptomics_bins', y='Proteomics_method', data=data_merged_missing_long, hue='Method')
plt.xticks(rotation=90)
plt.xlabel('Transcriptomics', fontsize='x-large')
plt.ylabel('Proteomics', fontsize='x-large') 
plt.legend(title='Method', title_fontsize='large', fontsize='large')
#size of the plot
plt.gcf().set_size_inches(20, 15)
plt.savefig('/home/jorgeribeiro/JorgeRibeiroThesis/results/boxplot_transcriptomics_proteomics_missing_next.png')
plt.clf()






