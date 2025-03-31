############################################################################
# 0) Load Required Packages
############################################################################
library(tidyverse)
library(recipes)
library(survival)
library(rpart)       # for decision tree (survival tree)
library(pROC)        # for ROC, AUC, CI of AUC
library(caret)       # for confusionMatrix
library(timeROC)     # for time-dependent AUC

############################################################################
# 1) Read & Prepare Data
############################################################################
# Replace with your actual CSV file path
df_full <- read_csv("CCA_ML.csv")

# Define expected columns
expected_cols <- c(
  "validation", "Survival", "OS", "Age", "Sex", "staging", "iCCA/eCCA",
  "treatment", "surgery", "HBsAg", "HCV", "PLAT", "INR", "ALB", "crea", "BILI",
  "ALT", "AST", "ALKP", "GGT", "CA199", "albi", "CTP score", "FIB4", "AFP"
)
df <- df_full[, expected_cols]

# Rename columns
colnames(df)[colnames(df) == "iCCA/eCCA"] <- "iCCA_eCCA"
colnames(df)[colnames(df) == "CTP score"] <- "CTP_score"

# Convert character columns to factors
df[] <- lapply(df, function(x) if (is.character(x)) as.factor(x) else x)

# Check OS units (assuming months based on typical survival data; convert to years)
df$OS <- df$OS / 12  # Convert months to years

# Split into training and validation sets
df_train <- df %>% filter(validation == 0)
df_val   <- df %>% filter(validation == 1)

cat("Training rows:", nrow(df_train), "\n")
cat("Validation rows:", nrow(df_val), "\n\n")

############################################################################
# 2) Recipe for Preprocessing
############################################################################
rec <- recipe(~ ., data = df_train) %>%
  update_role(validation, Survival, OS, new_role = "outcome") %>%
  step_log(CA199, offset = 1) %>%
  step_log(AFP, offset = 1) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_center(all_numeric_predictors()) %>%
  step_scale(all_numeric_predictors())

rec_prep <- prep(rec, training = df_train, retain = TRUE)
train_data <- bake(rec_prep, new_data = df_train)
val_data   <- bake(rec_prep, new_data = df_val)

# Remove rows with missing values and non-positive OS
train_data <- train_data %>% na.omit() %>% filter(OS > 0)
val_data   <- val_data %>% na.omit() %>% filter(OS > 0)

cat("After preprocessing:\n")
cat("  train_data rows:", nrow(train_data), "\n")
cat("  val_data rows:", nrow(val_data), "\n\n")

# Check event distribution
cat("Training set - Censored:", sum(train_data$Survival == 1), "Event:", sum(train_data$Survival == 2), "\n")
cat("Validation set - Censored:", sum(val_data$Survival == 1), "Event:", sum(val_data$Survival == 2), "\n\n")

############################################################################
# 3) Fit Survival Decision Tree
############################################################################
rpart_model <- rpart(
  formula = Surv(OS, ifelse(Survival == 2, 1, 0)) ~ . - OS - Survival - validation,
  data = train_data,
  method = "exp"
)

# Obtain predictions (predicted mean survival time)
train_pred_rpart <- predict(rpart_model, newdata = train_data)
val_pred_rpart   <- predict(rpart_model, newdata = val_data)

# Define risk score as predicted survival time (higher value = lower risk)
train_risk_score_rpart <- train_pred_rpart
val_risk_score_rpart   <- val_pred_rpart

############################################################################
# 4) Winsorize Risk Scores (Optional)
############################################################################
winsorize <- function(x, lower = 0.05, upper = 0.95) {
  q <- quantile(x, probs = c(lower, upper), na.rm = TRUE)
  x[x < q[1]] <- q[1]
  x[x > q[2]] <- q[2]
  return(x)
}

train_risk_score_rpart_adj <- winsorize(train_risk_score_rpart)
val_risk_score_rpart_adj   <- winsorize(val_risk_score_rpart)

cat("--- Decision Tree Risk Score Summary (After Winsorization) ---\n")
print(summary(train_risk_score_rpart_adj))

############################################################################
# 5) Binary Classification with Inverse Risk Score
############################################################################

# Invert the risk score so that a higher risk (i.e. lower survival) becomes more positive.
# Note: Based on your model, pROC reports "controls > cases" when using the inverse,
# which indicates that non-events have higher inverse scores than events.
# To align with the desired outcome (predict event when risk is high), we will classify a subject
# as an event if the adjusted inverse risk score is below the threshold.
train_risk_score_inv <- -train_risk_score_rpart
val_risk_score_inv   <- -val_risk_score_rpart

# Winsorize the inverse risk scores
train_risk_score_inv_adj <- winsorize(train_risk_score_inv)
val_risk_score_inv_adj   <- winsorize(val_risk_score_inv)

# Create binary event indicator: 1 = event, 0 = censored
train_event_binary <- ifelse(train_data$Survival == 2, 1, 0)
val_event_binary   <- ifelse(val_data$Survival == 2, 1, 0)

# ROC analysis using the inverse risk scores
roc_train_inv <- roc(train_event_binary, train_risk_score_inv_adj)
best_thr_inv <- coords(roc_train_inv, x = "best", ret = "threshold", best.method = "youden")[[1]]
cat("\nChosen inverse risk threshold (Youden's J):", best_thr_inv, "\n")

# Use the inverse threshold to classify:
# Since pROC indicated that controls have higher scores than cases for the inverse risk,
# we now classify as event (2) if the adjusted inverse risk score is LESS THAN the threshold.
train_pred_class_inv <- ifelse(train_risk_score_inv_adj < best_thr_inv, 2, 1)
val_pred_class_inv   <- ifelse(val_risk_score_inv_adj < best_thr_inv, 2, 1)

# Compute and print confusion matrices for training data
train_cm_inv <- confusionMatrix(
  factor(train_pred_class_inv, levels = c(1, 2)),
  factor(train_data$Survival, levels = c(1, 2)),
  positive = "2"
)
cat("\n=== Training Classification Metrics with Inverse Risk Score ===\n")
print(train_cm_inv)

# Compute and print confusion matrices for validation data
val_cm_inv <- confusionMatrix(
  factor(val_pred_class_inv, levels = c(1, 2)),
  factor(val_data$Survival, levels = c(1, 2)),
  positive = "2"
)
cat("\n=== Validation Classification Metrics with Inverse Risk Score ===\n")
print(val_cm_inv)

# --- Compute Additional Metrics ---
# Training Metrics
train_accuracy  <- train_cm_inv$overall['Accuracy']
train_precision <- train_cm_inv$byClass['Pos Pred Value']
train_recall    <- train_cm_inv$byClass['Sensitivity']
train_f1        <- 2 * (train_precision * train_recall) / (train_precision + train_recall)
cat(sprintf("\nTraining Accuracy: %.3f", train_accuracy))
cat(sprintf("\nTraining Precision: %.3f", train_precision))
cat(sprintf("\nTraining Recall: %.3f", train_recall))
cat(sprintf("\nTraining F1: %.3f\n", train_f1))

# Validation Metrics
val_accuracy  <- val_cm_inv$overall['Accuracy']
val_precision <- val_cm_inv$byClass['Pos Pred Value']
val_recall    <- val_cm_inv$byClass['Sensitivity']
val_f1        <- 2 * (val_precision * val_recall) / (val_precision + val_recall)
cat(sprintf("\nValidation Accuracy: %.3f", val_accuracy))
cat(sprintf("\nValidation Precision: %.3f", val_precision))
cat(sprintf("\nValidation Recall: %.3f", val_recall))
cat(sprintf("\nValidation F1: %.3f\n", val_f1))

# Recalculate AUC using the inverse risk scores
roc_train_inv_auc <- roc(train_event_binary, train_risk_score_inv_adj)
roc_val_inv_auc   <- roc(val_event_binary, val_risk_score_inv_adj)
cat(sprintf("\nTrain AUC (Inverse): %.3f\n", auc(roc_train_inv_auc)))
cat(sprintf("Val AUC (Inverse): %.3f\n", auc(roc_val_inv_auc)))

# --- Compute 95% Confidence Intervals for AUROC ---
auc_ci_train <- ci.auc(roc_train_inv_auc, conf.level = 0.95)
auc_ci_val   <- ci.auc(roc_val_inv_auc, conf.level = 0.95)
cat(sprintf("\nTrain AUC (95%% CI, Inverse): %.3f (%.3f - %.3f)\n", 
            auc(roc_train_inv_auc), auc_ci_train[1], auc_ci_train[3]))
cat(sprintf("Val AUC (95%% CI, Inverse): %.3f (%.3f - %.3f)\n", 
            auc(roc_val_inv_auc), auc_ci_val[1], auc_ci_val[3]))

############################################################################
# 6) Survival Metrics
############################################################################
train_surv <- Surv(train_data$OS, ifelse(train_data$Survival == 2, 1, 0))
val_surv   <- Surv(val_data$OS, ifelse(val_data$Survival == 2, 1, 0))

# Concordance index
cindex_dt <- concordance(train_surv ~ train_risk_score_rpart_adj)$concordance
cat(sprintf("\nConcordance Index (Train): %.3f\n", cindex_dt))

# Time-dependent AUC
time_points <- 1:5  # Years
timeROC_train_dt <- timeROC(
  T = train_data$OS,
  delta = ifelse(train_data$Survival == 2, 1, 0),
  marker = train_risk_score_rpart_adj,
  cause = 1,
  times = time_points,
  iid = TRUE
)
timeROC_val_dt <- timeROC(
  T = val_data$OS,
  delta = ifelse(val_data$Survival == 2, 1, 0),
  marker = val_risk_score_rpart_adj,
  cause = 1,
  times = time_points,
  iid = TRUE
)

cat("\nTime-dependent AUC (Training):\n")
print(data.frame(Time = time_points, AUC = timeROC_train_dt$AUC))
cat("\nTime-dependent AUC (Validation):\n")
print(data.frame(Time = time_points, AUC = timeROC_val_dt$AUC))

cat("\n=== DONE ===\n")
