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


#make a new directory to save the results if it does not exist
if (!file.exists("results/missForest_30_1000"))
{
  dir.create("results/missForest_30_1000")
  print("Directory created")
}

# Perform the imputation using missForest while saving output in a file

sink("results/missForest_30_1000/proteomics_imputation_output_maxitter30_ntree1000_replaceT_decreasingT.txt")
print("Imputation started")
imputed_dataset <- missForest(dataset, maxiter = 30, ntree = 1000, replace = TRUE, decreasing = TRUE, parallelize = "variables", verbose = TRUE,variablewise = TRUE)
print("Imputation done")


# Access the imputed dataset
imputed_data <- imputed_dataset$ximp
print("Imputed dataset accessed")

#obtain the first column from the original dataset
GeneSymbol <- dataset_withGeneSymbol[,1]
print("Sample names accessed")

#make the index of the imputed dataset the gene symbol
rownames(imputed_data) <- GeneSymbol
print("Sample names added")



#save data for future analysis, original missingness, imputation convergence, variable importance, OOB error
imputed_data$original_missingness <- rowSums(is.na(dataset))
imputed_data$imputation_convergence <- imputed_dataset$convergence
imputed_data$variable_importance <- imputed_dataset$importance
imputed_data$OOB_error <- imputed_dataset$error
print("Imputed dataset augmented")


# Save the imputed dataset in a new csv file
write.csv(imputed_data, file = "results/missForest_30_1000/proteomics_imputed_maxitter30_ntree1000_replaceT_decreasingT.csv")
print("Imputed dataset saved")

#save the orignal missingness, imputation convergence, variable importance, OOB error in a file txt
write.table(imputed_data$original_missingness, file = "results/missForest_30_1000/proteomics_original_missingness_maxitter30_ntree1000_replaceT_decreasingT.txt")
write.table(imputed_data$convergence, file = "results/missForest_30_1000/proteomics_imputation_convergence_maxitter30_ntree1000_replaceT_decreasingT.txt")
write.table(imputed_data$variable_importance, file = "results/missForest_30_1000/proteomics_variable_importance_maxitter30_ntree1000_replaceT_decreasingT.txt")
write.table(imputed_data$OOB_error, file = "results/missForest_30_1000/proteomics_OOB_error_maxitter30_ntree1000_replaceT_decreasingT.txt")
print("Imputation convergence, variable importance, OOB error saved")

# Save the imputed dataset in a new RData file
save(imputed_data, file = "results/missForest_30_1000/proteomics_imputed_maxitter30_ntree1000_replaceT_decreasingT.RData")
print("Imputed dataset saved in RData format")


# Stop the parallelization
stopImplicitCluster()
print("Parallelization stopped")

# End of script