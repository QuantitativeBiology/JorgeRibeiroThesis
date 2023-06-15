#Jorge Ribeiro
# 2018-05-31
# Imputation of missing values using missForest
# This script performs the imputation of missing values using the missForest

# Load required packages
library(missForest)
library(doParallel)
library(foreach)
library(iterators)
print("Packages loaded")

# Set the number of cores to use
cores <- 24
registerDoParallel(cores = cores)
print("Cores registered")

# Load the dataset
dataset_withGeneSymbol <- read.csv('/data/benchmarks/clines/proteomics.csv')
print("Dataset loaded")

# Remove the first column (sample names)
dataset <- dataset_withGeneSymbol[,-1]
print("First column removed")

# Perform the imputation using missForest
imputed_dataset <- missForest(dataset, verbose = TRUE, maxiter = 10, ntree = 1000, variablewise = TRUE, decreasing = TRUE, parallelize = "variables", replace = TRUE, xtrue = NULL)
print("Imputation done")

# Access the imputed dataset
imputed_data <- imputed_dataset$ximp

# Add the sample names to the imputed dataset
rownames(imputed_data) <- rownames(dataset_withGeneSymbol)
print("Sample names added")

#augment the data with data for future analysis, original missingness, imputation convergence, variable importance, OOB error
imputed_data$original_missingness <- rowSums(is.na(dataset))
imputed_data$imputation_convergence <- rowSums(imputed_dataset$convergence)
imputed_data$variable_importance <- rowSums(imputed_dataset$importance)
imputed_data$OOB_error <- rowSums(imputed_dataset$error)
print("Imputed dataset augmented")

# Save the imputed dataset
write.csv(imputed_data, file = "/results/imputated_matrix.csv")
print("Imputed dataset saved")
```