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

'''Main function for UCI letter and spam datasets.
'''

# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import pandas as pd 
import datetime

from data_loader import data_loader
from gain import gain
from utils import rmse_loss, get_hour_day


def main (args):
  '''Main function for UCI letter and spam datasets.
  
  Args:
    - data_name: letter or spam
    - miss_rate: probability of missing components
    - batch:size: batch size
    - hint_rate: hint rate
    - alpha: hyperparameter
    - iterations: iterations
    
  Returns:
    - imputed_data_x: imputed data
    - rmse: Root Mean Squared Error
  '''
  
  data_name = args.data_name
  miss_rate = args.miss_rate
  
  gain_parameters = {'batch_size': args.batch_size,
                     'hint_rate': args.hint_rate,
                     'alpha': args.alpha,
                     'iterations': args.iterations}
  
  # Load data and introduce missingness
  ori_data_x, miss_data_x, data_m = data_loader(data_name, miss_rate, True)

  #save data_m as a csv file with the time stamp
  time_stamp=get_hour_day( datetime.datetime.now())
  np.savetxt("results/data_m_" +str(time_stamp)+ '_' +args.data_name+"_missrate_"+str(args.miss_rate)+".csv", data_m, delimiter=",")


  # Impute missing data
  imputed_data_x = gain(miss_data_x, gain_parameters, ori_data_x.values)
  
  # Report the RMSE performance
  rmse, rmse_training = rmse_loss (ori_data_x.values, imputed_data_x, data_m)

  
  #obtain a pandas dataframe from the imputed data np array
  imputed_data_df = pd.DataFrame(imputed_data_x)

  #give the columns and rows the same names as the original data
  imputed_data_df.columns = ori_data_x.columns
  imputed_data_df.index = ori_data_x.index

  print()
  print('RMSE Performance: ' + str(np.round(rmse, 4)))
  
  return imputed_data_df, rmse

if __name__ == '__main__':  
  
  # Inputs for the main function
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_name',
      choices=['letter','spam', 'proteomics', 'proteomics_set_seed'],
      default='spam',
      type=str)
  parser.add_argument(
      '--miss_rate',
      help='missing data probability',
      default=0.2,
      type=float)
  parser.add_argument(
      '--batch_size',
      help='the number of samples in mini-batch',
      default=128,
      type=int)
  parser.add_argument(
      '--hint_rate',
      help='hint probability',
      default=0.9,
      type=float)
  parser.add_argument(
      '--alpha',
      help='hyperparameter',
      default=100,
      type=float)
  parser.add_argument(
      '--iterations',
      help='number of training interations',
      default=10000,
      type=int)
  
  args = parser.parse_args() 
  
  # Calls main function  
  imputed_data, rmse = main(args)
  rmse=np.round(rmse, 4)
  #get time stamp
  time_stamp=get_hour_day( datetime.datetime.now())

  #save the imputed data as a csv file
  imputed_data.to_csv("results/imputed_" +str(time_stamp)+ '_' +args.data_name+"_rmse_"+str(rmse)+"_missrate_"+str(args.miss_rate)+"_batchsize_"+str(args.batch_size)+"_hintrate_"+str(args.hint_rate)+"_alpha_"+str(args.alpha)+"_iterations_"+str(args.iterations)+".csv")

  