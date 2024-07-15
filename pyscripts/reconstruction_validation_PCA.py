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
from sklearn.decomposition import PCA
from upsetplot import UpSet
import itertools


sys.path.insert(0, '/home/jorgeribeiro/JorgeRibeiroThesis/GAIN_prots')
from utils import get_hour_day


# Get the file path from the command line
imputed_files_dict = {
    'ProteinsMeanValue': '/home/jorgeribeiro/JorgeRibeiroThesis/results/mean_rows.csv',
    'VAE': '/home/jorgeribeiro/JorgeRibeiroThesis/Proteomics/proteomicsVAE.csv',
    'DefaultGAIN': '/home/jorgeribeiro/JorgeRibeiroThesis/GAIN_prots/results/imputed_data_proteomics_missrate_0.0_batchsize_128_hintrate_0.9_alpha_100.0_iterations_10000.csv',
    'OptunaGAIN': '/home/jorgeribeiro/JorgeRibeiroThesis/GAIN_prots/results/imputed_17h_2Oct_proteomics_rmse_nan_missrate_0.0_batchsize_128_hintrate_0.14_alpha_1000.0_iterations_1000.csv',
    'MissForest': '/home/jorgeribeiro/JorgeRibeiroThesis/results/missForest_30_1000/proteomics_imputed_maxitter30_ntree1000_replaceT_decreasingT.csv',
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
#thresholds = [0.90, 0.95, 0.97, 0.99] # 0.1 means 10% of missing values per row

#create a folder with the current date and time if it doesnt exist
now = get_hour_day(datetime.datetime.now())
if not os.path.exists(f'/home/jorgeribeiro/JorgeRibeiroThesis/results/reconstruction_validation_PCA_{folder_name}_{now}'):
    os.mkdir(f'/home/jorgeribeiro/JorgeRibeiroThesis/results/reconstruction_validation_PCA_{folder_name}_{now}')

os.chdir(f'/home/jorgeribeiro/JorgeRibeiroThesis/results/reconstruction_validation_PCA_{folder_name}_{now}')

#create a dataframe to save the results
results = pd.DataFrame(columns=['dataset','MVs %', 'number of proteins (PCA)', 'proteins predicted' , 'protein', 'r2', 'rmse', 'mae', 'r2_full', 'rmse_full', 'mae_full', 'r2_train', 'rmse_train', 'mae_train'])
pca_components_files = list()

for file_name in imputed_files_dict:
    #cycle for each of the imputed datasets

    imputed_dataset = pd.read_csv(imputed_files_dict[file_name], index_col=0)
    imputed_dataset = imputed_dataset[overlap_cell_lines]
    imputed_dataset = imputed_dataset.transpose()
    print(f'Imputed dataset shape: {imputed_dataset.shape}')

    # Create a copy of the imputed dataset
    imputed_dataset_copy = imputed_dataset.copy()
    original_dataset_copy = original_dataset.copy()

    if file_name == 'VAE':
        original_dataset_copy = original_dataset_copy[imputed_dataset_copy.columns]

    #Get the most important features with PCA
    pca = PCA(n_components=0.90)#keep X% of the variance
    pca.fit(imputed_dataset_copy)
    imputed_dataset_copy = pca.transform(imputed_dataset_copy)
    original_dataset_copy = original_dataset_copy.fillna(0)
    original_dataset_copy = pca.transform(original_dataset_copy)

    print(f'Imputed dataset shape after PCA: {imputed_dataset_copy.shape}')

    #Get a graph for the explained variance per component
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    plt.xticks(rotation=90)
    ax = sns.barplot(x=[f'PCA_{i}' for i in range(15)], y=pca.explained_variance_ratio_[:15])
    ax.set_title('Explained variance per component')
    ax.set_xlabel('Component')
    ax.set_ylabel('Explained variance')
    plt.savefig(f'explained_variance_{file_name}_{now}.png')
    plt.clf()


    # Save the PCA components in a dataframe
    pca_components = pd.DataFrame(data=imputed_dataset_copy, columns=[f'PCA_{file_name}_{i}' for i in range(imputed_dataset_copy.shape[1])])
    pca_components_files.append(pca_components)

    # Get MVs percentage from the original np array
    MVs_percentage = np.count_nonzero(np.isnan(original_dataset_copy)) / original_dataset_copy.size
    print(f'MVs percentage: {MVs_percentage}')
    
    # tranform the dataset to numpy arrays
    numpy_independent = independent_dataset.to_numpy()
    numpy_imputed = imputed_dataset_copy

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
        results = results._append({'dataset': file_name, 'MVs %':MVs_percentage , 'number of proteins (PCA)':numpy_imputed.shape[1], 'proteins predicted':numpy_independent.shape[1], 'protein': independent_dataset.columns[protein], 'r2': r2, 'rmse': rmse, 'mae': mae, 'r2_full': r2_full, 'rmse_full': rmse_full, 'mae_full': mae_full, 'r2_train':r2_train_final, 'rmse_train':rmse_train_final, 'mae_train':mae_train_final}, ignore_index=True)


# Save the results in a csv file
results.to_csv(f'reconstruction_validation_PCA_{now}.csv', index=False)

#Get a graph with the comparison between the rmse_full and rmse_train values for each dataset side by side
# results['rmse_full'] = results['rmse_full'].astype(float)
# results['rmse_train'] = results['rmse_train'].astype(float)

# sns.set(style="whitegrid")
# fig, ax = plt.subplots(figsize=(10, 6))
# ax = sns.barplot(x="dataset", y="rmse_full", data=results, color='blue', label='rmse_full')
# ax = sns.barplot(x="dataset", y="rmse_train", data=results, color='red', label='rmse_train')
# ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
# ax.legend()
# plt.title('Comparison between rmse_full and rmse_train')
# plt.tight_layout()
# plt.savefig(f'rmse_full_vs_rmse_train_{now}.png')
# plt.show()
# plt.close()
# plt.clf()


sns.set_theme(style="whitegrid")
ax = sns.boxplot(x="dataset", y="rmse_full", data=results)
ax.set_title('RMSE values for each threshold')
ax.set_xlabel('Dataset')
ax.set_ylabel('RMSE')
plt.savefig(f'rmse_datasets_full_{now}.png')
plt.clf()


# Get a graph of the number of proteins kept after PCA for each dataset
sns.set_theme(style="whitegrid")
ax = sns.barplot(x="dataset", y="number of proteins (PCA)", data=results)
ax.set_title('Number of proteins kept after PCA')
ax.set_xlabel('Dataset')
ax.set_ylabel('Number of proteins')
plt.savefig(f'number_of_proteins_{now}.png')
plt.clf()

# Get a graph to compare the rmses of the different datasets
sns.set_theme(style="whitegrid")
ax = sns.barplot(x="dataset", y="rmse", data=results)
ax.set_title('RMSE values for each threshold')
ax.set_xlabel('Dataset')
ax.set_ylabel('RMSE')
plt.savefig(f'rmse_datasets_{now}.png')
plt.clf()



# Get a graph to compare the rmse_full and rmse_train values for each dataset with bars side by side
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(10, 6))
ax = sns.barplot(x="dataset", y="rmse_full", data=results, color='blue', label='rmse_full')
ax = sns.barplot(x="dataset", y="rmse_train", data=results, color='red', label='rmse_train')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
ax.legend()
plt.title('Comparison between rmse_full and rmse_train')
plt.tight_layout()
plt.savefig(f'rmse_full_vs_rmse_train_{now}.png')
plt.clf()

# Get a graph to compare multiple metrics for each variable with bars side by side
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(10, 6))
metrics = ['rmse_full', 'rmse_train', 'rmse', 'number of proteins (PCA)']  # Add the metrics you want to compare
colors = ['blue', 'red', 'green', 'orange']  # Add colors for each metric

for i, metric in enumerate(metrics):
    ax = sns.barplot(x="dataset", y=metric, data=results, color=colors[i], label=metric)

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
ax.legend()
plt.title('Comparison of Multiple Metrics for Each Variable')
plt.tight_layout()
plt.savefig(f'multiple_metrics_comparison_{now}.png')
plt.clf()

#obtain correlation between the datasets of the pca_components_files list
correlation_array = np.zeros((len(pca_components_files), len(pca_components_files)))
for i in range(len(pca_components_files)):
    for j in range(len(pca_components_files)):
        correlation_array[i,j] = pca_components_files[i].corrwith(pca_components_files[j], axis=0).mean()

# Get a graph to compare the correlation between the datasets
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(10, 6))
ax = sns.heatmap(correlation_array, annot=True, cmap='coolwarm')
ax.set_xticklabels([file_name for file_name in imputed_files_dict])
ax.set_yticklabels([file_name for file_name in imputed_files_dict])
plt.title('Correlation between datasets in PCA Components')
plt.tight_layout()
plt.savefig(f'correlation_datasets_{now}.png')
plt.clf()

#Get a cluster map of the correlation between the datasets
sns.set(style="whitegrid")
ax = sns.clustermap(correlation_array, annot=True, cmap='coolwarm')
plt.title('Clustermap of Correlation between datasets in PCA Components')
plt.tight_layout()
plt.savefig(f'clustermap_correlation_datasets_{now}.png')
plt.clf()


#https://jokergoo.github.io/ComplexHeatmap-reference/book/upset-plot.html
#Obtain an Upset plot to show if there are PCs that are common between datasets with a combination matrix 

combined_df = pd.concat(pca_components_files, axis=1)
binary_matrix = combined_df.notnull().astype(int)
combination_matrix = binary_matrix.T.dot(binary_matrix)
upset = UpSet(combination_matrix)
upset.plot()
plt.title('UpSet Plot of PCA Components Across Files')
plt.savefig(f'upset_plot_{now}.png')
plt.clf()




# tidy_df = pd.concat([df.assign(Source=file_name) for df, file_name in zip(pca_components_files, imputed_files_dict)], ignore_index=True)
# pca_sets = [set(row.dropna()) for _, row in tidy_df.iterrows()]
# pca_df= pd.DataFrame(pca_sets)

# pca_df.columns = [f'PCA_{file_name}_{i}' for file_name in imputed_files_dict for i in range(pca_df.shape[1])]
# pca_df.to_csv(f'pca_components_{now}.csv', index=False)

# upset = UpSet(pca_df)
# upset.plot()
# plt.title('UpSet Plot of PCA Components Across Files')
# plt.savefig(f'upset_plot_{now}.png')
# plt.clf()

#Tried Manually
# intersections = {}
# for i, j in itertools.combinations(range(len(pca_sets)), 2):
#     intersect_size = len(pca_sets[i].intersection(pca_sets[j]))
#     if intersect_size > 0:
#         intersections[(i, j)] = intersect_size

# # Create UpSet plot with intersection sizes
# upset = UpSet(intersections=intersections)





