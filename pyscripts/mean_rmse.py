import pandas
import numpy as np
import datetime
import os

#load the MVS_adder and rmse_loss functions from directory
import sys
sys.path.append('/home/jorgeribeiro/JorgeRibeiroThesis/GAIN_prots')
from utils import MVS_adder, rmse_loss

#load the original protein data
data = pandas.read_csv('/data/benchmarks/clines/proteomics.csv', index_col=0)

#check if there are any rows with only missing values
print('before adding MVS number of rows with only missing values')
print(data.isnull().all(axis=1).sum())

#get the mask matrix of augmented missing values
data_m = MVS_adder(data.values, 0.001)

data_copy=data.copy()
data_with_extra_MVS = data_m*data_copy
data_with_extra_MVS[data_m==0] = np.nan

#check if there are any rows with only missing values
print('after adding MVS number of rows with only missing values')
print(data_with_extra_MVS.isnull().all(axis=1).sum())

#replace the MVS with the mean of the row that the MVS is in
data_with_extra_MVS_rows = data_with_extra_MVS.copy()
for index, row in data_with_extra_MVS_rows.iterrows():
    # Find the mean of the row excluding NaN values
    row_mean = np.nanmean(row)
    
    # Replace NaN values in the row with the row mean
    data_with_extra_MVS_rows.loc[index] = row.fillna(row_mean)


#replace the MVS with the mean of the column that the MVS is in
data_with_extra_MVS_columns = data_with_extra_MVS.copy()
data_with_extra_MVS_columns = data_with_extra_MVS_columns.fillna(data_with_extra_MVS_columns.mean())


#save the RMSE in a file in the same directory
file_name = '/home/jorgeribeiro/JorgeRibeiroThesis/results'

#change directory to the one where the file is
os.chdir(file_name)
file_name = file_name + '/rmse_for_mean_datasets_set_seed.txt'



#load the imputed data from file_name
mean_rows = pandas.read_csv('/home/jorgeribeiro/JorgeRibeiroThesis/results/mean_rows.csv', index_col=0)
mean_columns = pandas.read_csv('/home/jorgeribeiro/JorgeRibeiroThesis/results/mean_column.csv', index_col=0)


rmse, rmse_training = rmse_loss(data.values, data_with_extra_MVS_columns.values, data_m)

#print the RMSE
print('RMSE',rmse)
print('RMSE_training',rmse_training)

with open(file_name, 'a') as file:
    file.write('mean_columns\n')
    file.write(f'rmse:{rmse}\n')
    file.write(f'\n')

print('columns done')

#remove the rows with only MVs in data_with_extra_MVS_rows from data, data_with_extra_MVS_rows and data_m
data = data[~data_with_extra_MVS_rows.isnull().all(axis=1)]
data_m = data_m[~data_with_extra_MVS_rows.isnull().all(axis=1)]
data_with_extra_MVS_rows = data_with_extra_MVS_rows[~data_with_extra_MVS_rows.isnull().all(axis=1)]





rmse, rmse_training = rmse_loss(data.values, data_with_extra_MVS_rows.values, data_m)

#print the RMSE
print('RMSE',rmse)
print('RMSE_training',rmse_training)

with open(file_name, 'a') as file:
    file.write('mean_rows\n')
    file.write(f'rmse:{rmse}\n')
    file.write(f'\n')


