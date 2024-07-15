""" Jorge Ribeiro 2023
Created to validate the reconstruction of the dataset from imputation methods

Objective: Reconstructe a protein from the independent dataset from the imputed dataset information in order to check if 
the extra information added is relevant to the model and therefore useful """

# Importing the libraries
from sklearn.linear_model import LinearRegression, Lasso, LassoCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np
import sys
import os
import datetime
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer
import seaborn as sns
import matplotlib.pyplot as plt


sys.path.insert(0, '/home/jorgeribeiro/JorgeRibeiroThesis/GAIN_prots')
from utils import get_hour_day


# Get the file path from the command line
imputed_files_dict = {
    'VAE': '/home/jorgeribeiro/JorgeRibeiroThesis/Proteomics/proteomicsVAE.csv',
    'DefaultGAIN': '/home/jorgeribeiro/JorgeRibeiroThesis/GAIN_prots/results/imputed_data_proteomics_missrate_0.0_batchsize_128_hintrate_0.9_alpha_100.0_iterations_10000.csv',
    'OptunaGAIN': '/home/jorgeribeiro/JorgeRibeiroThesis/GAIN_prots/results/imputed_17h_2Oct_proteomics_rmse_nan_missrate_0.0_batchsize_128_hintrate_0.14_alpha_1000.0_iterations_1000.csv',
    'MissForest': '/home/jorgeribeiro/JorgeRibeiroThesis/results/missForest_30_1000/proteomics_imputed_maxitter30_ntree1000_replaceT_decreasingT.csv',
    'ProteinsMeanValue': '/home/jorgeribeiro/JorgeRibeiroThesis/results/mean_rows.csv',
}

folder_name = sys.argv[1]
if folder_name == None:
    raise ValueError('Please provide the folder name to save the results')

# Importing the datasets
original_dataset = pd.read_csv('/data/benchmarks/clines/proteomics.csv', index_col=0)
independent_dataset = pd.read_csv('/data/benchmarks/clines/proteomics_ccle.csv', index_col=0)

print(f'Independent dataset shape: {independent_dataset.shape}')   

overlap_proteins = list( original_dataset.index.intersection(independent_dataset.index) )
overlap_cell_lines = list( original_dataset.columns.intersection(independent_dataset.columns) )
print(f'Overlap proteins: {len(overlap_proteins)}')
print(f'Overlap cell lines: {len(overlap_cell_lines)}')

#maintain only the overlap cell lines
original_dataset = original_dataset[overlap_cell_lines]
independent_dataset = independent_dataset[overlap_cell_lines]


#get the rows of the independent dataset that are not in the original dataset 
independent_dataset = independent_dataset.loc[~independent_dataset.index.isin(original_dataset.index)]

# Get the rows that dont have any missing values
independent_dataset = independent_dataset.dropna(axis=0, how='any')

#transpose the datasets (cell lines as rows and proteins as columns)
original_dataset = original_dataset.transpose()
independent_dataset = independent_dataset.transpose()

print(f'Original dataset shape: {original_dataset.shape}')
print(f'Independent dataset shape: {independent_dataset.shape}')



# Separate the imputed dataset according to a threshold of missing values per row (proteins)
thresholds = [0.90, 0.95, 0.97, 0.99] # 0.1 means 10% of missing values per row

#create a folder with the current date and time if it doesnt exist
now = get_hour_day(datetime.datetime.now())
if not os.path.exists(f'/home/jorgeribeiro/JorgeRibeiroThesis/results/reconstruction_validation_{folder_name}_{now}'):
    os.mkdir(f'/home/jorgeribeiro/JorgeRibeiroThesis/results/reconstruction_validation_{folder_name}_{now}')

os.chdir(f'/home/jorgeribeiro/JorgeRibeiroThesis/results/reconstruction_validation_{folder_name}_{now}')

#create a dataframe to save the results
results = pd.DataFrame(columns=['dataset','threshold', 'number of samples', 'proteins predicted' , 'protein', 'r2', 'rmse', 'mae', 'r2_full', 'rmse_full', 'mae_full', 'r2_train', 'rmse_train', 'mae_train'])

for file_name in imputed_files_dict:
    #cycle for each of the imputed datasets

    imputed_dataset = pd.read_csv(imputed_files_dict[file_name], index_col=0)
    imputed_dataset = imputed_dataset[overlap_cell_lines]
    imputed_dataset = imputed_dataset.transpose()

    print(f'Imputed dataset shape: {imputed_dataset.shape}')

    # Create a copy of the imputed dataset
    imputed_dataset_copy = imputed_dataset.copy()
    original_dataset_copy = original_dataset.copy()

    for threshold in thresholds:
        
        # Drop the columns with more missing values than the threshold calculated on the original dataset
        original_dataset_copy = original_dataset_copy.dropna(axis=1, thresh=int((threshold)*original_dataset.shape[0]))
        imputed_dataset_copy = imputed_dataset_copy[original_dataset_copy.columns]

        #remove indexes and save them
        #indexes_imputed = imputed_dataset_copy.index
        imputed_dataset_copy = imputed_dataset_copy.reset_index(drop=True)
        # indexes_independent = independent_dataset.index
        independent_dataset = independent_dataset.reset_index(drop=True)
        
        # tranform the dataset to numpy arrays
        numpy_independent = independent_dataset.to_numpy()
        numpy_imputed = imputed_dataset_copy.to_numpy()

        #confirm if numpy_independent and numpy_imputed have the same shape
        if numpy_independent.shape[0] != numpy_imputed.shape[0]:
            raise ValueError('The number of rows of the numpy arrays is not the same')

        for protein in range(numpy_independent.shape[1]):
            #cycle that predicts each protein
            first_column_independent = numpy_independent[:,protein]

            # Create an instance of the Lasso Regression model
            model = Lasso(alpha=0.1, max_iter=10000)
            # model = LassoCV(alphas=[1,0.1, 0.01, 0.001],cv=5, random_state=0, max_iter=10000)

            kfolds=5
            fold_size = int(numpy_independent.shape[0]/kfolds)

            # Create arrays to store the results
            y_pred = np.zeros((kfolds, fold_size))
            r2 = np.zeros(kfolds)
            rmse = np.zeros(kfolds)
            mae = np.zeros(kfolds)

            y_pred_train = np.zeros((kfolds, 4*fold_size))
            r2_train = np.zeros(kfolds)
            rmse_train = np.zeros(kfolds)
            mae_train = np.zeros(kfolds)

            #Create a manual for cycle to do the cross validation
            for i in range(5):
                # Determine the indices for the current fold
                start_idx = i * fold_size
                end_idx = (i + 1) * fold_size

                # Split the data into training and testing sets
                X_train = np.concatenate([numpy_imputed[:start_idx], numpy_imputed[end_idx:]])
                y_train = np.concatenate([first_column_independent[:start_idx], first_column_independent[end_idx:]])
                X_test = numpy_imputed[start_idx:end_idx]
                y_test = first_column_independent[start_idx:end_idx]

                # Fit the model on the training data
                model.fit(X_train, y_train)

                # Use the model to predict on the testing data
                y_pred[i] = model.predict(X_test)

                #get results for the fully reconstructed protein
                r2[i] = r2_score(y_test, y_pred[i])
                rmse[i] = mean_squared_error(y_test, y_pred[i], squared=False)
                mae[i] = mean_absolute_error(y_test, y_pred[i])
                
                #get the training and test error
                y_pred_train[i] = model.predict(X_train)
                r2_train[i] = r2_score(y_train, y_pred_train[i])
                rmse_train[i] = mean_squared_error(y_train, y_pred_train[i], squared=False)
                mae_train[i] = mean_absolute_error(y_train, y_pred_train[i])

            #Obtain a fully reconstructed protein by combining the predictions of the 5 folds
            y_pred_full = np.concatenate(y_pred)

            #get results for the fully reconstructed protein
            r2_full = r2_score(first_column_independent, y_pred_full)
            rmse_full = mean_squared_error(first_column_independent, y_pred_full, squared=False)
            mae_full = mean_absolute_error(first_column_independent, y_pred_full)

            #get the training and test error
            r2_train_final = np.mean(r2_train)
            rmse_train_final = np.mean(rmse_train)
            mae_train_final = np.mean(mae_train)

            r2 = np.mean(r2)
            rmse = np.mean(rmse)
            mae = np.mean(mae)

            #save the results in the dataframe~
            results = results._append({'dataset': file_name, 'threshold': threshold, 'number of samples':numpy_imputed.shape[0], 'proteins predicted':numpy_independent.shape[1], 'protein': independent_dataset.columns[protein], 'r2': r2, 'rmse': rmse, 'mae': mae, 'r2_full': r2_full, 'rmse_full': rmse_full, 'mae_full': mae_full, 'r2_train':r2_train_final, 'rmse_train':rmse_train_final, 'mae_train':mae_train_final}, ignore_index=True)


# Save the results in a csv file
results.to_csv(f'reconstruction_validation_{now}.csv', index=False)

#Get a graph with the comparison between the rmse_full and rmse_train values for each threshold of each dataset
sns.set_theme(style="whitegrid")
ax = sns.boxplot(x="threshold", y="rmse_full", hue="dataset", data=results)
ax.set_title('RMSE values for each threshold')
ax.set_xlabel('Threshold')
ax.set_ylabel('RMSE')
plt.savefig(f'rmse_thresholds_datasets_full_vs_train{now}.png')
plt.clf()

# Get a boxplot graph with the r2 values, one for each threshold and save them with legends
sns.set_theme(style="whitegrid")
ax = sns.boxplot(x="threshold", y="r2", data=results)
ax.set_title('R2 values for each threshold')
ax.set_xlabel('Threshold')
ax.set_ylabel('R2')
plt.savefig(f'r2_thresholds_{now}.png')
plt.clf()

# Get a boxplot graph with the rmse values, one for each threshold and save them with legends
sns.set_theme(style="whitegrid")
ax = sns.boxplot(x="threshold", y="rmse", data=results)
ax.set_title('RMSE values for each threshold')
ax.set_xlabel('Threshold')
ax.set_ylabel('RMSE')
plt.savefig(f'rmse_thresholds_{now}.png')
plt.clf()











