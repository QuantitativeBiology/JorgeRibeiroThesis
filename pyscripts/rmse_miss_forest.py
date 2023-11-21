import pandas
import numpy as np
import datetime
import os

#load the MVS_adder and rmse_loss functions from directory
import sys
sys.path.append('/home/jorgeribeiro/JorgeRibeiroThesis/GAIN_prots')
from utils import MVS_adder, rmse_loss

#get file_name from command line
file_name = sys.argv[1]
data_m_name = sys.argv[2]

#load the original protein data
data = pandas.read_csv('/data/benchmarks/clines/proteomics.csv', index_col=0)

#load the imputed data from file_name
imputed_data = pandas.read_csv(file_name, index_col=0)

print(imputed_data.shape)

#remove last column from imputed data
imputed_data = imputed_data.iloc[:, :-1]

#print the shape of the data
print(data.shape)
print(imputed_data.shape)

#get the mask matrix of augmented missing values
data_m = MVS_adder(data.values, 0.1, True)

#import data_m_used from file
data_m_used = pandas.read_csv(data_m_name, index_col=0)

data_m_used = data_m_used.values

#check if the mask matrix is equal to the one used in the imputation
print(np.array_equal(data_m, data_m_used))


#calculate the RMSE
rmse, rmse_training = rmse_loss(data.values, imputed_data.values, data_m)

#print the RMSE
print('RMSE',rmse)
print('RMSE_training',rmse_training)

#save the RMSE in a file in the same directory
file_name = file_name.split('/')[0:-1]
file_name = '/'.join(file_name)

 #change directory to the one where the file is
os.chdir(file_name)


file_name = file_name + '/rmse.txt'



with open(file_name, 'w') as f:
    f.write('RMSE: ' + str(rmse) + '\n')
    f.write('RMSE_training: ' + str(rmse_training) + '\n')
    f.write('file_name: ' + file_name + '\n')
    f.write('data_m_name: ' + data_m_name + '\n')
    f.write('data_m_correspondence: ' + str(np.array_equal(data_m, data_m_used)) + '\n')


    
