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

'''Data loader for UCI letter, spam and MNIST datasets.
'''

# Necessary packages
import numpy as np
from utils import binary_sampler, MVS_adder
from keras.datasets import mnist
import pandas as pd


def data_loader (data_name, miss_rate, set_seed=False):
  '''Loads datasets and introduce missingness.
  
  Args:
    - data_name: letter, spam, or mnist
    - miss_rate: the probability of missing components
    
  Returns:
    data_x: original data
    miss_data_x: data with missing values
    data_m: indicator matrix for missing components
    data_indexed: data with missing values and indexed
  '''
  
  # Load data
  if data_name in ['letter', 'spam']:
    file_name = 'data/'+data_name+'.csv'
    data_x = np.loadtxt(file_name, delimiter=",", skiprows=1)
    # Parameters
    no, dim = data_x.shape
  
    # Introduce missing data
    data_m = binary_sampler(1-miss_rate, no, dim)
  elif data_name == 'mnist':
    (data_x, _), _ = mnist.load_data()
    data_x = np.reshape(np.asarray(data_x), [60000, 28*28]).astype(float)

    # Parameters
    no, dim = data_x.shape
    
    # Introduce missing data
    data_m = binary_sampler(1-miss_rate, no, dim)

  #load data from the proteomics dataset
  else:
    file_name = '/data/benchmarks/clines/proteomics.csv'

    data_x=pd.read_csv(file_name, index_col=0)
    #print(data_x.shape) #6671,949 -> 6671 proteins, 949 cell lines

    no, dim = data_x.shape
    data_m= MVS_adder(data_x.values, miss_rate, set_seed=set_seed)

    
    #adaptar python3
    # TODO: alterar para proteomics


  
  miss_data_x = data_x.values.copy()
  miss_data_x[data_m == 0] = np.nan
      
  return data_x, miss_data_x, data_m
