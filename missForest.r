#Jorge Ribeiro
# 2018-05-31
# Imputation of missing values using missForest
# This script performs the imputation of missing values using the missForest

# Load required packages
library(missForest)
library(doParallel)
library(foreach)
library(iterators)
library(reticulate)
print("Packages loaded")

# set directory name
directory_name <- "set_seed_20_ntree_100"

# Set the number of cores to use
cores <- 24
registerDoParallel(cores = cores)
print("Cores registered")

# Load the dataset
dataset_withGeneSymbol <- read.csv('/data/benchmarks/clines/proteomics.csv')
print("Dataset loaded")


# Remove the first column (sample names)
dataset_original <- dataset_withGeneSymbol[,-1]
print("First column removed")


#make a new directory with the name saved in directory variable to save the results if it does not exist
if (!dir.exists(paste("results/",directory_name,sep=""))) {
  dir.create(paste("results/",directory_name,sep=""))
  print("Directory created")
}

#import a pyhton functio MVS_adder from a python script in other folder
setwd('/home/jorgeribeiro/JorgeRibeiroThesis/GAIN_prots')
source_python("utils.py")
print("Python script imported")

#transform dataset_original into numpy array and keeping the missing values
dataset_original_values <- as.matrix(dataset_original)
#dataset_original_values[is.na(dataset_original_values)] <- NaN

#add missing values to the dataset
data_m <- MVS_adder(dataset_original_values, 0.1, TRUE)
print("Missing values added")

#print percentage of missing values
x<- sum(is.na(dataset_original))/(dim(dataset_original)[1]*dim(dataset_original)[2])
y<- sum(1-data_m)/(dim(data_m)[1]*dim(data_m)[2])
print("Percentage of missing values in original dataset:")
print(x)
print("Percentage of missing values in dataset with missing values:")
print(y)

setwd('/home/jorgeribeiro/JorgeRibeiroThesis')

#save data_m in a csv file
write.csv(data_m, file = paste("results/",directory_name,"/mask_matrix_ss20_mr0_1",".csv",sep=""))

# create a dataset that is nan where data_m is 0 and the original dataset where data_m is 1
dataset <- data_m * dataset_original_values
dataset[data_m==0] <- NaN

# print percentage of missing values
x <- sum(is.nan(dataset))/(dim(dataset)[1]*dim(dataset)[2])
print("Percentage of missing values:")
print(x)

dataset <- as.data.frame(dataset)

# Perform the imputation using missForest while saving output in a file
sink(paste("results/",directory_name,"/MF_imputation_",directory_name ,".txt",sep=""), append=FALSE)
print("Imputation started")
imputed_dataset <- missForest(dataset, maxiter = 30, ntree = 100, replace = TRUE, decreasing = TRUE, parallelize = "variables", verbose = TRUE,variablewise = TRUE)
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
imputed_data$original_missingness <- rowSums(1-data_m)
imputed_data$imputation_convergence <- imputed_dataset$convergence
imputed_data$variable_importance <- imputed_dataset$importance
imputed_data$OOB_error <- imputed_dataset$error
print("Imputed dataset augmented")

# Save the imputed dataset in a new csv file
write.csv(imputed_data, file = paste("results/",directory_name,"/MF_imputation",directory_name ,".csv",sep=""))
print("Imputed dataset saved")

#save the orignal missingness, imputation convergence, variable importance, OOB error in a file txt
write.table(imputed_data$original_missingness, file = paste("results/",directory_name,"/MF_original_missingness",directory_name ,".txt",sep=""))
write.table(imputed_data$convergence, file = paste("results/",directory_name,"/MF_convergence",directory_name ,".txt",sep=""))
write.table(imputed_data$variable_importance, file = paste("results/",directory_name,"/MF_ivariable_importance",directory_name ,".txt",sep=""))
write.table(imputed_data$OOB_error, file = paste("results/",directory_name,"/MF_OOB_error",directory_name ,".txt",sep=""))
print("Imputation convergence, variable importance, OOB error saved")

# Save the imputed dataset in a new RData file
save(imputed_data, file = paste("results/",directory_name,"/MF_imputation",directory_name ,".RData",sep=""))
print("Imputed dataset saved in RData format")

# Stop the parallelization
stopImplicitCluster()
print("Parallelization stopped")

# End of script