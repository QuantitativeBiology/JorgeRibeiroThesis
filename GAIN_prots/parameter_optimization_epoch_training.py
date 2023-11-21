#import all necessary libraries
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt
import optuna
from utils import normalization, renormalization, rounding, rmse_loss, xavier_init, get_hour_day, binary_sampler, uniform_sampler, sample_batch_index, MVS_adder
import plotly
import kaleido
import sklearn
import os

# Define custom suggest function for optuna
def suggest_custom(name, trial, nr_layers):
    '''Suggest function for optuna

    Args:
        - name: name of the parameter
        - trial: optuna trial object
    Returns:
        - list: list of values for the parameter
    '''

    # Define list of values for each parameter
    if name == 'discr_size_layers':
        discr_size_layers = list(range(0, nr_layers))
        discr_size_layers[0] = 1
        for i in range(1, nr_layers):
            discr_size_layers[i] = trial.suggest_float('discr_size_layers_{}'.format(i), 0, 1, step= 0.1)
        return discr_size_layers
    elif name == 'gen_size_layers':
        gen_size_layers = list(range(0, nr_layers))
        gen_size_layers[0] = 1
        for i in range(1, nr_layers):
            gen_size_layers[i] = trial.suggest_float('gen_size_layers_{}'.format(i), 0, 1, step=0.1)
        return gen_size_layers
    else:
        raise ValueError('Unknown parameter: {}'.format(name))



# Regular GAIN function
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

    #To_do: add more hidden layers (2nd)
    #To_do: change size of hidden layers (1st)
    #To_do: export to data base  the results of optuna

    # Hidden state dimensions
    h_dim = int(dim)

    # Normalization
    norm_data, norm_parameters = normalization(data_x)
    norm_data_x = np.nan_to_num(norm_data, 0)

    ## GAIN architecture
    # Input placeholders
    # Data vector
    X = tf.placeholder(tf.float32, shape=[None, dim])
    # Mask vector
    M = tf.placeholder(tf.float32, shape=[None, dim])
    # Hint vector
    H = tf.placeholder(tf.float32, shape=[None, dim])

    # Discriminator variables
    D_W1 = tf.Variable(xavier_init([dim * 2, h_dim]))  # Data + Hint as inputs
    D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

    D_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
    D_b2 = tf.Variable(tf.zeros(shape=[h_dim]))

    D_W3 = tf.Variable(xavier_init([h_dim, dim]))
    D_b3 = tf.Variable(tf.zeros(shape=[dim]))  # Multi-variate outputs

    theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]

    # Generator variables
    # Data + Mask as inputs (Random noise is in missing components)
    G_W1 = tf.Variable(xavier_init([dim * 2, h_dim]))
    G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

    G_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
    G_b2 = tf.Variable(tf.zeros(shape=[h_dim]))
    
    G_W3 = tf.Variable(xavier_init([h_dim, dim]))
    G_b3 = tf.Variable(tf.zeros(shape=[dim]))

    theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]

    ## GAIN functions
    # Generator
    def generator(x, m):
        # Concatenate Mask and Data
        inputs = tf.concat(values=[x, m], axis=1)
        G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
        G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
        # MinMax normalized output
        G_prob = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3)
        return G_prob

    # Discriminator
    def discriminator(x, h):
        # Concatenate Data and Hint
        inputs = tf.concat(values=[x, h], axis=1)
        D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
        D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
        D_logit = tf.matmul(D_h2, D_W3) + D_b3
        D_prob = tf.nn.sigmoid(D_logit)
        return D_prob

    ## GAIN structure
    # Generator
    G_sample = generator(X, M)

    # Combine with observed data
    Hat_X = X * M + G_sample * (1 - M)

    # Discriminator
    D_prob = discriminator(Hat_X, H)

    ## GAIN loss
    D_loss_temp = -tf.reduce_mean(M * tf.log(D_prob + 1e-8) + (1 - M) * tf.log(1. - D_prob + 1e-8))
    G_loss_temp = -tf.reduce_mean((1 - M) * tf.log(D_prob + 1e-8))
    MSE_loss = tf.reduce_mean((M * X - M * G_sample) ** 2) / tf.reduce_mean(M)

    D_loss = D_loss_temp
    G_loss = G_loss_temp + alpha * MSE_loss

    ## GAIN solver
    D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
    G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

    ## Iterations
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Start Iterations
    # Change the model to work by epochs (goes through every batch of data once per epoch)
    epochs = iterations
    original_batch_size = batch_size
    for epoch in range(epochs):
        batch_begginning = 0
        batch_size = original_batch_size
        for it in range(int(no / batch_size)):

            if batch_begginning + batch_size > no:
                batch_size = no - batch_begginning

            X_mb = norm_data_x[batch_begginning:batch_begginning + batch_size, :]
            M_mb = data_m[batch_begginning:batch_begginning + batch_size, :]
            batch_begginning += batch_size

            # Sample random vectors
            Z_mb = uniform_sampler(0, 0.01, batch_size, dim)
            # Sample hint vectors
            H_mb_temp = binary_sampler(hint_rate, batch_size, dim)
            H_mb = M_mb * H_mb_temp
            # Combine random vectors with observed vectors
            X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb

            _, D_loss_curr = sess.run([D_solver, D_loss_temp], feed_dict={M: M_mb, X: X_mb, H: H_mb})
            _, G_loss_curr, MSE_loss_curr = sess.run([G_solver, G_loss_temp, MSE_loss], feed_dict={X: X_mb, M: M_mb, H: H_mb})



    ## Return imputed data
    Z_mb = uniform_sampler(0, 0.01, no, dim)
    M_mb = data_m
    X_mb = norm_data_x
    X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb

    imputed_data = sess.run([G_sample], feed_dict={X: X_mb, M: M_mb})[0]

    # Renormalization
    imputed_data = renormalization(imputed_data, norm_parameters)

    # reasign original values to observed data
    imputed_data = data_m * norm_data_x + (1-data_m) * imputed_data

    # Rounding
    imputed_data = rounding(imputed_data, ori_data_x)

    # RMSE using function from utils
    rmse = rmse_loss(ori_data_x, imputed_data, data_m)

    return rmse
    

# use optuna to find best hyperparameters for GAIN
def objective(trial):
    '''Objective function for optuna

    Args:
        - trial: optuna trial object
    Returns:
        - rmse: rmse of imputed data
    '''

    # Load data and parameters
    ori_data_x_indexed = pd.read_csv('/data/benchmarks/clines/proteomics.csv', index_col=0)
    ori_data_x = ori_data_x_indexed.values

    #Define original mask matrix - 1: observed, 0: missing
    data_m_original = 1-np.isnan(ori_data_x)

    # Introduce missing data (data_m is the mask matrix after adding MVs - 1: observed, 0: missing)
    data_m = MVS_adder(ori_data_x, 0.1, True)
    data_x = ori_data_x.copy()
    data_x[data_m == 0] = np.nan

    # GAIN parameters
    gain_parameters = {
        'batch_size': int(trial.suggest_float('batch_size', 10, 1000, step=1)),
        'hint_rate': trial.suggest_float('hint_rate', 0, 0.9),
        'alpha': trial.suggest_float('alpha', 10, 10000, step= 1),
        'iterations': int(trial.suggest_float('iterations', 10, 1000, step=1)),

    }

    rmse, _ = gain(data_x, gain_parameters, ori_data_x)

    return rmse
    
if __name__ == '__main__':  

    #set working directory for optuna results
    os.chdir('optuna_results')

    #set folder name
    folder_name = '22_Oct_epoch_training'

    # create folder for optuna results
    try:
        os.mkdir(folder_name)
    except OSError:
        print ("Creation of the directory %s failed" % folder_name)
    else:
        print ("Successfully created the directory %s " % folder_name)

    # Optuna study
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)
    print('Number of finished trials:', len(study.trials))
    print('Best trial:', study.best_trial.params)
    print('Best value:', study.best_value)
    print('Best index:', study.best_trial.number)
    print('Best parameters:', study.best_params)
    print('Best trial user attrs:', study.best_trial.user_attrs)


    # Save study results to csv with best parameters values in the name
    study_results = study.trials_dataframe()
    study_results.to_csv(f"{folder_name}/study.csv")

    # Save best parameters to csv
    best_params = study.best_params
    best_params = pd.DataFrame.from_dict(best_params, orient='index')
    #transpose dataframe
    best_params = best_params.transpose()

    #add another column with best value
    best_params['best_value'] = study.best_value
    best_params = best_params.transpose()


    
    #best_params['best_value'] = study.best_value
    best_params.to_csv(f"{folder_name}/best_params.csv")




    # Show graph of study and save to png    
    optuna.visualization.plot_optimization_history(study).write_image(f"{folder_name}/optimization_history.png")
    optuna.visualization.plot_param_importances(study).write_image(f"{folder_name}/param_importances.png")
    optuna.visualization.plot_slice(study).write_image(f"{folder_name}/slice.png")
    optuna.visualization.plot_contour(study).write_image(f"{folder_name}/contour.png")
    optuna.visualization.plot_parallel_coordinate(study).write_image(f"{folder_name}/parallel_coordinate.png")
    optuna.visualization.plot_edf(study).write_image(f"{folder_name}/edf.png")
    #optuna.visualization.plot_intermediate_values(study).write_image(f"{folder_name}/intermediate_values.png")
    #intermediate values ddoesnt make sense for current architecture





    


