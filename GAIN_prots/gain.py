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

'''GAIN function.
Date: 2020/02/28
Reference: J. Yoon, J. Jordon, M. van der Schaar, "GAIN: Missing Data 
           Imputation using Generative Adversarial Nets," ICML, 2018.
Paper Link: http://proceedings.mlr.press/v80/yoon18a/yoon18a.pdf
Contact: jsyoon0823@gmail.com
'''

# Necessary packages
#import tensorflow as tf
##IF USING TF 2 use following import to still use TF < 2.0 Functionalities
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import datetime
import os

from utils import normalization, renormalization, rounding, rmse_loss
from utils import xavier_init, get_hour_day
from utils import binary_sampler, uniform_sampler, sample_batch_index

from sklearn.metrics import roc_curve, roc_auc_score


def gain (data_x, gain_parameters, ori_data_x):
  '''Impute missing values in data_x
  
  Args:
    - data_x: data with missing values
    - gain_parameters: GAIN network parameters:
      - batch_size: Batch size
      - hint_rate: Hint rate
      - alpha: Hyperparameter
      - iterations: Iterations
    - ori_data_x: original data without missing values for RMSE calculation
  Returns:
    - imputed_data: imputed data
  '''
  # Define mask matrix - 1: observed, 0: missing
  data_m = 1-np.isnan(data_x)

  # System parameters
  batch_size = gain_parameters['batch_size']
  hint_rate = gain_parameters['hint_rate']
  alpha = gain_parameters['alpha']
  iterations = gain_parameters['iterations']
  missing_rate = np.isnan(data_x).sum()/(data_x.shape[0]*data_x.shape[1])
  missing_rate = round(missing_rate, 3)

  # Other parameters
  no, dim = data_x.shape
  
  # Hidden state dimensions
  h_dim = int(dim)
  
  # Normalization
  norm_data, norm_parameters = normalization(data_x)
  norm_data_x = np.nan_to_num(norm_data, 0)
  
  ## GAIN architecture   
  # Input placeholders
  # Data vector
  X = tf.placeholder(tf.float32, shape = [None, dim])
  # Mask vector 
  M = tf.placeholder(tf.float32, shape = [None, dim])
  # Hint vector
  H = tf.placeholder(tf.float32, shape = [None, dim])
  
  # Discriminator variables
  D_W1 = tf.Variable(xavier_init([dim*2, h_dim])) # Data + Hint as inputs
  D_b1 = tf.Variable(tf.zeros(shape = [h_dim]))
  
  D_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
  D_b2 = tf.Variable(tf.zeros(shape = [h_dim]))
  
  D_W3 = tf.Variable(xavier_init([h_dim, dim]))
  D_b3 = tf.Variable(tf.zeros(shape = [dim]))  # Multi-variate outputs
  
  theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]
  
  #Generator variables
  # Data + Mask as inputs (Random noise is in missing components)
  G_W1 = tf.Variable(xavier_init([dim*2, h_dim]))  
  G_b1 = tf.Variable(tf.zeros(shape = [h_dim]))
  
  G_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
  G_b2 = tf.Variable(tf.zeros(shape = [h_dim]))
  
  G_W3 = tf.Variable(xavier_init([h_dim, dim]))
  G_b3 = tf.Variable(tf.zeros(shape = [dim]))
  
  theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]
  
  ## GAIN functions
  # Generator
  def generator(x,m):
    # Concatenate Mask and Data
    inputs = tf.concat(values = [x, m], axis = 1) 
    G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
    G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)   
    # MinMax normalized output
    G_prob = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3) 
    return G_prob
      
  # Discriminator
  def discriminator(x, h):
    # Concatenate Data and Hint
    inputs = tf.concat(values = [x, h], axis = 1) 
    D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)  
    D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
    D_logit = tf.matmul(D_h2, D_W3) + D_b3
    D_prob = tf.nn.sigmoid(D_logit)
    return D_prob
  
  ## GAIN structure
  # Generator
  G_sample = generator(X, M)
 
  # Combine with observed data
  Hat_X = X * M + G_sample * (1-M)
  
  # Discriminator
  D_prob = discriminator(Hat_X, H)
  
  ## GAIN loss
  D_loss_temp = -tf.reduce_mean(M * tf.log(D_prob + 1e-8) \
                                + (1-M) * tf.log(1. - D_prob + 1e-8)) 
  
  G_loss_temp = -tf.reduce_mean((1-M) * tf.log(D_prob + 1e-8))
  
  MSE_loss = \
  tf.reduce_mean((M * X - M * G_sample)**2) / tf.reduce_mean(M)
  
  D_loss = D_loss_temp
  G_loss = G_loss_temp + alpha * MSE_loss 

  #save loss across iterations for plotting including both parts of 
  D_loss_list = []
  G_loss_list = []
  MSE_loss_alpha_list = []
  MSE_loss_list = []
  rmse_loss_list = []
  G_total_loss_list = []
  rmse_training_data_list = []
  Accuracy_list = []
  Precision_list = []
  Recall_list = []
  Perc_mvs_list = []
  AUC_list = []

  
  ## GAIN solver
  D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
  G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)
  
  ## Iterations
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
   
  # Start Iterations
  for it in tqdm(range(iterations)):    
      
    # Sample batch
    batch_idx = sample_batch_index(no, batch_size)
    X_mb = norm_data_x[batch_idx, :]  
    M_mb = data_m[batch_idx, :]  
    # Sample random vectors  
    Z_mb = uniform_sampler(0, 0.01, batch_size, dim) 
    # Sample hint vectors
    H_mb_temp = binary_sampler(hint_rate, batch_size, dim)
    H_mb = M_mb * H_mb_temp
      
    # Combine random vectors with observed vectors
    X_mb = M_mb * X_mb + (1-M_mb) * Z_mb 
      
    _, D_prob_curr, D_loss_curr = sess.run([D_solver, D_prob ,D_loss_temp], 
                              feed_dict = {M: M_mb, X: X_mb, H: H_mb})
    _, G_loss_curr, MSE_loss_curr = \
    sess.run([G_solver, G_loss_temp, MSE_loss],
             feed_dict = {X: X_mb, M: M_mb, H: H_mb})

    #transform D_prob to a numpy array
    D_prob_curr = np.array(D_prob_curr)

    #calculate accuracy, precision and recall of discriminator
    D_prob_curr = D_prob_curr.flatten()
    D_prob_curr = np.round(D_prob_curr)
    D_prob_curr = D_prob_curr.astype(int)
    M_np = M_mb.flatten()
    M_np = M_np.astype(int)
    accuracy = np.mean(D_prob_curr == M_np)
    precision = np.sum(D_prob_curr * M_np) / np.sum(D_prob_curr)
    recall = np.sum(D_prob_curr * M_np) / np.sum(M_np)

    #calculate percentage of MVs predicted
    perc_mvs = np.sum(1-M_np) / len(M_np)

    #calculate AUC
    auc = roc_auc_score(M_np, D_prob_curr)

    #percentage of true positives and true negatives
    #tn, fp, fn, tp = confusion_matrix(M_np, D_prob_curr).ravel()

    
    #save loss across iterations for plotting
    D_loss_list.append(D_loss_curr)
    G_loss_list.append(G_loss_curr)
    MSE_loss_list.append(MSE_loss_curr)
    MSE_loss_alpha_list.append(alpha * MSE_loss_curr)
    G_total_loss_list.append(G_loss_curr + alpha * MSE_loss_curr)
    Accuracy_list.append(accuracy)
    Precision_list.append(precision)
    Recall_list.append(recall)
    Perc_mvs_list.append(perc_mvs)
    AUC_list.append(auc)

    #get imputed data from this iteration
    Z_mb = uniform_sampler(0, 0.01, no, dim) 
    M_mb = data_m
    X_mb = norm_data_x          
    X_mb = M_mb * X_mb + (1-M_mb) * Z_mb 
    imputed_data = sess.run([G_sample], feed_dict = {X: X_mb, M: M_mb})[0]
    #imputed_data = data_m * norm_data_x + (1-data_m) * imputed_data
    imputed_data = renormalization(imputed_data, norm_parameters)
    imputed_data = rounding(imputed_data, data_x)
    
    #save rmse across iterations for plotting
    rmse_loss_current, rmse_training_data_loss_current= rmse_loss(ori_data_x, imputed_data, data_m, False)
    rmse_loss_list.append(rmse_loss_current)
    rmse_training_data_list.append(rmse_training_data_loss_current)


  #get hour and day in the same string for saving the plot
  time_stamp=get_hour_day( datetime.datetime.now())

  #change directory to save plots to a new folder if it does not exist yet results/time_stamp with f string
  if not os.path.exists(f'loss_plots/{time_stamp}'):
    os.makedirs(f'loss_plots/{time_stamp}')


  #plot loss across iterations
  plt.plot(D_loss_list, label='D_loss')
  plt.plot(G_loss_list, label='G_loss')
  plt.plot(MSE_loss_list, label='MSE_loss')
  plt.plot(MSE_loss_alpha_list, label='alpha*MSE_loss')
  plt.plot(G_total_loss_list, label='G_total_loss')
  plt.title('GAIN Loss Plot')
  plt.xlabel('Iteration')
  plt.ylabel('Loss')
  plt.legend()
  plt.show()


  #save the plot with the hyperparameters in the name
  plt.savefig('loss_plots/'+time_stamp+'/GAIN_loss_'+ str(time_stamp)+ '_' + str(missing_rate) + '_' + str(gain_parameters['batch_size']) + '_' + str(gain_parameters['hint_rate']) + '_' + str(gain_parameters['alpha']) + '_' + str(gain_parameters['iterations']) + '.png')
  plt.clf()

  #plot the losses saved in 4 subplots on top of each other and start them all at 0
  fig, axs = plt.subplots(4, figsize=(14, 10))
  axs[0].set_ylim(0, max(D_loss_list))
  axs[0].plot(D_loss_list, label='D_loss')
  axs[0].set_title('D_loss')
  axs[1].set_ylim(0, max(G_loss_list))
  axs[1].plot(G_loss_list, label='G_loss')
  axs[1].set_title('G_loss')
  axs[2].set_ylim(0, max(MSE_loss_list))
  axs[2].plot(MSE_loss_list, label='MSE_loss')
  axs[2].set_title('MSE_loss')
  axs[3].set_ylim(0, max(rmse_loss_list))
  axs[3].plot(rmse_loss_list, label='RMSE')
  axs[3].set_title('RMSE')
  plt.show()
  plt.savefig('loss_plots/'+time_stamp+'/GAIN_loss_subplots_'+ str(time_stamp) + '_' + str(missing_rate) + '_' + str(gain_parameters['batch_size']) + '_' + str(gain_parameters['hint_rate']) + '_' + str(gain_parameters['alpha']) + '_' + str(gain_parameters['iterations']) + '.png')
  plt.clf()

  #plot accuracy, precision, recall, perc_mvs and AUC across iterations in subplots, precision and recall together
  fig, axs = plt.subplots(4, figsize=(14, 10))
  axs[0].set_ylim(0, 1)
  axs[0].plot(Accuracy_list, label='Accuracy')
  axs[0].set_title('Accuracy')
  axs[1].set_ylim(0, 1)
  axs[1].plot(Precision_list, label='Precision')
  axs[1].plot(Recall_list, label='Recall')
  axs[1].set_title('Precision and Recall')
  axs[1].legend()
  axs[2].set_ylim(0, 1)
  axs[2].plot(AUC_list, label='AUC')
  axs[2].set_title('AUC')
  axs[3].set_ylim(0, 1)
  axs[3].plot(Perc_mvs_list, label='Percentage of MVs')
  axs[3].set_title('Percentage of MVs')
  plt.savefig('loss_plots/'+time_stamp+'/GAIN_accuracy_precision_recall_perc_mvs_'+ str(time_stamp) + '_' + str(missing_rate) + '_' + str(gain_parameters['batch_size']) + '_' + str(gain_parameters['hint_rate']) + '_' + str(gain_parameters['alpha']) + '_' + str(gain_parameters['iterations']) + '.png')
  plt.clf()

  #plot rmse across iterations and save it
  plt.plot(rmse_loss_list, label='RMSE_loss')
  plt.plot(rmse_training_data_list, label='RMSE_training_data_loss')
  plt.title('GAIN RMSE Plot')
  plt.xlabel('Iteration')
  plt.ylabel('RMSE')
  plt.legend()
  plt.show()
  plt.savefig('loss_plots/'+time_stamp+'/GAIN_rmse_'+ str(time_stamp) + '_' + str(missing_rate) + '_'  + str(gain_parameters['batch_size']) + '_' + str(gain_parameters['hint_rate']) + '_' + str(gain_parameters['alpha']) + '_' + str(gain_parameters['iterations']) + '.png')
  plt.clf()


  ## Return imputed data      
  Z_mb = uniform_sampler(0, 0.01, no, dim) 
  M_mb = data_m
  X_mb = norm_data_x          
  X_mb = M_mb * X_mb + (1-M_mb) * Z_mb
  H_mb = M_mb * binary_sampler(hint_rate, no, dim)


  #obtain D_prob for the imputed data
  D_prob_imputed = sess.run([D_prob], feed_dict = {X: X_mb, M: M_mb, H: M_mb})[0]

  #transform D_prob to a numpy array 1D without flattening
  D_prob_imputed = np.array(D_prob_imputed)
  #D_prob_imputed = D_prob_imputed.flatten()
  D_prob_imputed = np.reshape(D_prob_imputed, (D_prob_imputed.shape[0]* D_prob_imputed.shape[1],1))

  
  #get a plot of D_prob distribution for the imputed data
  plt.hist(D_prob_imputed, bins=20)
  plt.xlabel('D Probability for Final Data')
  plt.ylabel('Frequency')
  plt.savefig('loss_plots/'+time_stamp+'/D_prob_curr_distribution_'+ str(time_stamp) + '_' + str(missing_rate) + '_'  + str(gain_parameters['batch_size']) + '_' + str(gain_parameters['hint_rate']) + '_' + str(gain_parameters['alpha']) + '_' + str(gain_parameters['iterations']) + '.png')
  plt.clf()

  #get a ROC curve for the imputed data with the diagonal line
  fpr, tpr, thresholds = roc_curve(data_m.flatten(), D_prob_imputed.flatten())
  plt.plot([0, 1], [0, 1], linestyle='--')
  plt.plot(fpr, tpr)
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.savefig('loss_plots/'+time_stamp+'/ROC_curve_'+ str(time_stamp) + '_' + str(missing_rate) + '_'  + str(gain_parameters['batch_size']) + '_' + str(gain_parameters['hint_rate']) + '_' + str(gain_parameters['alpha']) + '_' + str(gain_parameters['iterations']) + '.png')
  plt.clf()

  imputed_data = sess.run([G_sample], feed_dict = {X: X_mb, M: M_mb})[0]
  
  imputed_data = data_m * norm_data_x + (1-data_m) * imputed_data 

  
  # Renormalization
  imputed_data = renormalization(imputed_data, norm_parameters)  
  
  # Rounding
  imputed_data = rounding(imputed_data, data_x)  
          
  return imputed_data





