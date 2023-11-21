# coding=utf-8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''Utility functions for GAIN.

(1) normalization: MinMax Normalizer
(2) renormalization: Recover the data from normalzied data
(3) rounding: Handlecategorical variables after imputation
(4) rmse_loss: Evaluate imputed data in terms of RMSE
(5) xavier_init: Xavier initialization
(6) binary_sampler: sample binary random variables
(7) uniform_sampler: sample uniform random variables
(8) sample_batch_index: sample random batch index
'''
 
# Necessary packages
import numpy as np
import datetime
#import tensorflow as tf
##IF USING TF 2 use following import to still use TF < 2.0 Functionalities
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def get_hour_day(time):
  '''Get the hour and day of the week from a datetime object
  Args:
    - time: datetime object
  Returns:
    - time: hour_day_month string
  '''
  hour = str(time.hour)
  day = str(time.day)
  month = time.strftime("%B")

  time=hour+'h_'+day+month[:3]

  return time

def normalization (data, parameters=None):
  '''Normalize data in [0, 1] range.
  
  Args:
    - data: original data
  
  Returns:
    - norm_data: normalized data
    - norm_parameters: min_val, max_val for each feature for renormalization
  '''

  # Parameters
  _, dim = data.shape
  norm_data = data.copy()
  
  if parameters is None:
  
    # MixMax normalization
    min_val = np.zeros(dim)
    max_val = np.zeros(dim)
    
    # For each dimension
    for i in range(dim):
      min_val[i] = np.nanmin(norm_data[:,i])
      norm_data[:,i] = norm_data[:,i] - np.nanmin(norm_data[:,i])
      max_val[i] = np.nanmax(norm_data[:,i])
      norm_data[:,i] = norm_data[:,i] / (np.nanmax(norm_data[:,i]) + 1e-6)   
      
    # Return norm_parameters for renormalization
    norm_parameters = {'min_val': min_val,
                       'max_val': max_val}

  else:
    min_val = parameters['min_val']
    max_val = parameters['max_val']
    
    # For each dimension
    for i in range(dim):
      norm_data[:,i] = norm_data[:,i] - min_val[i]
      norm_data[:,i] = norm_data[:,i] / (max_val[i] + 1e-6)  
      
    norm_parameters = parameters    
      
  return norm_data, norm_parameters


def renormalization (norm_data, norm_parameters):
  '''Renormalize data from [0, 1] range to the original range.
  
  Args:
    - norm_data: normalized data
    - norm_parameters: min_val, max_val for each feature for renormalization
  
  Returns:
    - renorm_data: renormalized original data
  '''
  
  min_val = norm_parameters['min_val']
  max_val = norm_parameters['max_val']

  _, dim = norm_data.shape
  renorm_data = norm_data.copy()
    
  for i in range(dim):
    renorm_data[:,i] = renorm_data[:,i] * (max_val[i] + 1e-6)   
    renorm_data[:,i] = renorm_data[:,i] + min_val[i]
    
  return renorm_data


def rounding (imputed_data, data_x):
  '''Round imputed data for categorical variables.
  
  Args:
    - imputed_data: imputed data
    - data_x: original data with missing values
    
  Returns:
    - rounded_data: rounded imputed data
  '''
  
  _, dim = data_x.shape
  rounded_data = imputed_data.copy()
  
  for i in range(dim):
    temp = data_x[~np.isnan(data_x[:, i]), i]
    # Only for the categorical variable
    if len(np.unique(temp)) < 20:
      rounded_data[:, i] = np.round(rounded_data[:, i])
      
  return rounded_data


def rmse_loss (ori_data, imputed_data, data_m, print_option=True):
  '''Compute RMSE loss between ori_data and imputed_data
  
  Args:
    - ori_data: original data without added missing values
    - imputed_data: imputed data
    - data_m: indicator matrix for missingness after added missing values
    
  Returns:
    - rmse: Root Mean Squared Error
  '''
  
  ori_data, norm_parameters = normalization(ori_data)
  imputed_data, _ = normalization(imputed_data, norm_parameters)
  
  # Obtain a matrix of missing  values in ori data with np.isnan
  # and then convert it to a matrix of 0s and 1s with astype(int)
  #is 1 where there is missing data and 0 where there is not
  data_missing_initial= np.isnan(ori_data)
  data_missing_initial = data_missing_initial.astype(int)

  # replace the missing values in the original data with 0s
  # ori_data[data_missing_initial==1] = imputed_data[data_missing_initial==1]
  ori_data[data_missing_initial==1] = 0
  

  
  # Obtain a matrix that is 1 where data_missing_initial is 0 and data_m is 0 (i.e. where there is no missing data in the original data and but there is in the extended missing data)
  data_missing_shared = (data_missing_initial) * (1-data_m)
  data_missing_added = (1-data_missing_initial) * (1-data_m)
  data_training = (1-data_missing_initial) * (data_m)


  #get number of missing values in the original data
  num_missing = np.sum(data_missing_initial)

  #get number of missing values in the extended missing data
  num_missing_extended = np.sum(data_missing_added)

  #get percentage of MVs in the original data and in the extended missing data
  perc_missing = num_missing/(ori_data.shape[0]*ori_data.shape[1])
  perc_missing_extended = num_missing_extended/(ori_data.shape[0]*ori_data.shape[1])
  perc_missing_shared = np.sum(data_missing_shared)/(ori_data.shape[0]*ori_data.shape[1])
  perc_missing_total = (np.sum(1-data_m))/(ori_data.shape[0]*ori_data.shape[1])
  if print_option:
    print("perc_missing_ori_data: ", perc_missing)
    print("perc_missing_data_missing_initial", np.sum(data_missing_initial)/(ori_data.shape[0]*ori_data.shape[1]))
    print("perc_missing_extended: ", perc_missing_extended)
    print("perc_missing_shared: ", perc_missing_shared)
    print("perc_missing_total: ", perc_missing_total)
    
  # Only for missing values
  # nominator = np.sum(((1-data_m) * ori_data - (1-data_m) * imputed_data)**2)
  # denominator = np.sum(1-data_m)
  nominator = np.sum((data_missing_added * ori_data - data_missing_added * imputed_data)**2)
  denominator = np.sum(data_missing_added)
  if denominator == 0:
    if print_option:
      print("No missing values were added, no error can be calculated")
    return np.nan, np.nan
    
  rmse = np.sqrt(nominator/float(denominator))

  #get shared MVs error compared to zero (mean when normalized?) - 
  nominator_shared = np.sum((data_missing_shared * ori_data - data_missing_shared * imputed_data)**2)
  if print_option:
    print("error_shared_MVS_to_0: ", np.sqrt(nominator_shared/float(np.sum(data_missing_shared))))

  #get data training error - should be zero
  nominator_training = np.sum((data_training * ori_data - data_training * imputed_data)**2)
  rmse_training = np.sqrt(nominator_training/float(np.sum(data_training)))
  if print_option:
    print("error_training_data: ", rmse_training)
  
  return rmse, rmse_training


def xavier_init(size):
  '''Xavier initialization.
  
  Args:
    - size: vector size
    
  Returns:
    - initialized random vector.
  '''
  in_dim = size[0]
  xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
  return tf.random_normal(shape = size, stddev = xavier_stddev)
      
def MVS_adder(data_x, perc_MV, set_seed=False):
  '''Add missing values to the original data.
  
  Args:
    - data_x: original data with missing values
    - data_missing_initial: indicator matrix for missingness
    - perc_MV: percentage of MVs to add
    
  Returns:
    - data_x_extended: original data with more missing values
    - data_m_extended: indicator matrix for missingness
  '''
  data_m_extended = 1-np.isnan(data_x)


  #get number of MVs to add
  num_MV_to_add = int(data_x.size*perc_MV)

  # get indices to add MVS besides the ones already present with a set seed
  if set_seed:
    np.random.seed(20)

  idx_to_add = np.where(data_m_extended==1)
  idx_to_add = np.array(idx_to_add)
  idx_to_add = idx_to_add.T
  idx_to_add = idx_to_add.tolist()
  idx_to_add = np.random.permutation(idx_to_add)
  idx_to_add = idx_to_add[:num_MV_to_add]
  idx_to_add = np.array(idx_to_add)
  idx_to_add = idx_to_add.T

  #add MVs to the original data
  np.put(data_m_extended, np.ravel_multi_index(idx_to_add, data_x.shape), 0)

  return data_m_extended



def binary_sampler(p, rows, cols):
  '''Sample binary random variables.
  
  Args:
    - p: probability of 1
    - rows: the number of rows
    - cols: the number of columns
    
  Returns:
    - binary_random_matrix: generated binary random matrix.
  '''
  unif_random_matrix = np.random.uniform(0., 1., size = [rows, cols])
  binary_random_matrix = 1*(unif_random_matrix < p)
  return binary_random_matrix


def uniform_sampler(low, high, rows, cols):
  '''Sample uniform random variables.
  
  Args:
    - low: low limit
    - high: high limit
    - rows: the number of rows
    - cols: the number of columns
    
  Returns:
    - uniform_random_matrix: generated uniform random matrix.
  '''
  return np.random.uniform(low, high, size = [rows, cols])       


def sample_batch_index(total, batch_size):
  '''Sample index of the mini-batch.
  
  Args:
    - total: total number of samples
    - batch_size: batch size
    
  Returns:
    - batch_idx: batch index
  '''
  total_idx = np.random.permutation(total)
  batch_idx = total_idx[:batch_size]
  return batch_idx
  

  
