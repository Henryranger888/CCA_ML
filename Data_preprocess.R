############################################################################
# 0) Install & Load Required Packages
############################################################################
install.packages("randomForestSRC")
install.packages("caret")
install.packages("pROC")
install.packages("recipes")
install.packages("xgboost")
install.packages("rpart")
install.packages("survival")
install.packages("readr")
install.packages("dplyr")
install.packages("glmnet")
install.packages("purrr")
install.packages("timeROC")
install.packages("survivalROC")
install.packages("fastshap")

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(survival)
  library(caret)
  library(pROC)
  library(recipes)
  library(randomForestSRC)
  library(xgboost)
  library(rpart)
  library(glmnet)
  library(purrr)
  # Load required packages
  library(tidyverse)
  library(randomForestSRC)
  library(fastshap)
  library(ggplot2)
})

############################################################################
# 0) Load Required Packages
############################################################################
library(tidyverse)
library(recipes)
library(survival)
library(randomForestSRC)
library(pROC)
library(caret)

############################################################################
# 1) Read & Subset Data
############################################################################
# Replace with your actual CSV path/file
df_full <- read_csv("CCA_ML.csv")

# Columns we expect
expected_cols <- c(
  "validation",   # 0 = training, 1 = validation
  "Survival",     # event indicator (1=censor,2=event) or vice versa
  "OS",           # time
  "Age", "Sex", "staging",
  "iCCA/eCCA",
  "treatment", "surgery",
  "HBsAg", "HCV", "PLAT", "INR", "ALB", "crea", "BILI",
  "ALT", "AST", "ALKP", "GGT", "CA199", "albi",
  "CTP score",
  "FIB4",
  "AFP"
)

df <- df_full[, expected_cols]

# Rename some columns for convenience
colnames(df)[colnames(df) == "iCCA/eCCA"] <- "iCCA_eCCA"
colnames(df)[colnames(df) == "CTP score"] <- "CTP_score"

# Convert any character columns to factor
df[] <- lapply(df, function(x) {
  if (is.character(x)) as.factor(x) else x
})

# Split into training vs validation
df_train <- df %>% filter(validation == 0)
df_val   <- df %>% filter(validation == 1)

cat("Initially:\n")
cat("  df_train rows:", nrow(df_train), "\n")
cat("  df_val   rows:", nrow(df_val), "\n\n")

############################################################################
# 2) Build a Recipe & Preprocess
############################################################################
# We'll treat (validation, Survival, OS) as 'outcome' so the recipe won't
# try to transform or impute them, but we won't use them as predictors either.
rec <- recipe(~ ., data = df_train) %>%
  update_role(validation, Survival, OS, new_role = "outcome") %>%

  # Log-transform CA199 & AFP (adding offset=1 avoids log(0))
  step_log(CA199, offset = 1) %>%
  step_log(AFP,   offset = 1) %>%

  # Median-impute numeric
  step_impute_median(all_numeric(), -has_role("outcome")) %>%

  # Mode-impute factor/categorical
  step_impute_mode(all_nominal(), -has_role("outcome")) %>%

  # One-hot encode nominal columns
  step_dummy(all_nominal(), -has_role("outcome")) %>%

  # Center & scale numeric
  step_center(all_numeric(), -has_role("outcome")) %>%
  step_scale(all_numeric(),  -has_role("outcome"))

# Prepare recipe on the training set
rec_prep <- prep(rec, training = df_train, retain = TRUE)

# Bake final training data
train_data <- bake(rec_prep, new_data = df_train, composition = "data.frame")
validation_data <- bake(rec_prep, new_data = df_val, composition = "data.frame")

cat("After recipe, before na.omit:\n")
cat("  train_data rows:", nrow(train_data), "\n")
cat("  validation_data rows:", nrow(validation_data), "\n\n")

# (Optional) Remove rows with any remaining NAs (if any remain)
train_data <- na.omit(train_data)
validation_data <- na.omit(validation_data)

cat("After na.omit:\n")
cat("  train_data rows:", nrow(train_data), "\n")
cat("  validation_data rows:", nrow(validation_data), "\n\n")

############################################################################
# 3) Define & remove the Surv object columns
############################################################################
# Keep the Surv() object in case you want reference, but remove it from X's
train_data$SurvObj      <- with(train_data, Surv(OS, Survival))
validation_data$SurvObj <- with(validation_data, Surv(OS, Survival))

# Remove them from the predictor set (since SurvObj is not a predictor)
train_data_x <- subset(train_data, select = -SurvObj)
validation_data_x <- subset(validation_data, select = -SurvObj)
