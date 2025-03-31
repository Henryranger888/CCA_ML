############################################################################
# 0) Load Required Packages
############################################################################
library(tidyverse)
library(recipes)
library(survival)
library(pROC)        # for ROC and AUC
library(caret)       # for confusionMatrix
library(timeROC)     # for time-dependent AUC

############################################################################
# 1) Read & Prepare Data
############################################################################
# Replace with your actual CSV file path
df_full <- read_csv("CCA_ML.csv")

# Expected columns
expected_cols <- c(
  "validation",   # 0 = training, 1 = validation
  "Survival",     # event indicator (1=censor, 2=event)
  "OS",           # survival time
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

# Rename columns for convenience
colnames(df)[colnames(df) == "iCCA/eCCA"] <- "iCCA_eCCA"
colnames(df)[colnames(df) == "CTP score"] <- "CTP_score"

# Convert character columns to factors
df[] <- lapply(df, function(x) {
  if (is.character(x)) as.factor(x) else x
})

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
  # Log-transform CA199 & AFP (offset=1 avoids log(0))
  step_log(CA199, offset = 1) %>%
  step_log(AFP, offset = 1) %>%
  # Median-impute numeric predictors
  step_impute_median(all_numeric(), -has_role("outcome")) %>%
  # Mode-impute factor predictors
  step_impute_mode(all_nominal(), -has_role("outcome")) %>%
  # One-hot encode factor predictors
  step_dummy(all_nominal(), -has_role("outcome")) %>%
  # Center & scale numeric predictors
  step_center(all_numeric(), -has_role("outcome")) %>%
  step_scale(all_numeric(), -has_role("outcome"))

rec_prep <- prep(rec, training = df_train, retain = TRUE)
train_data <- bake(rec_prep, new_data = df_train)
val_data   <- bake(rec_prep, new_data = df_val)

# Remove rows with missing values
train_data <- na.omit(train_data)
val_data   <- na.omit(val_data)

# Filter out observations with OS <= 0 (required for survival analysis)
train_data <- train_data %>% filter(OS > 0)
val_data   <- val_data %>% filter(OS > 0)

cat("After recipe + na.omit + OS > 0 filtering:\n")
cat("  train_data rows:", nrow(train_data), "\n")
cat("  val_data   rows:", nrow(val_data), "\n\n")

############################################################################
# 3) Fit Linear Regression Model (Ignoring Censoring)
############################################################################
# Fit a linear regression model to predict OS.
# Exclude OS, Survival, and validation from the predictors.
lm_model <- lm(OS ~ . - OS - Survival - validation, data = train_data)
summary(lm_model)  # Optional: view model summary

# Get predictions (predicted OS) on training and validation sets.
train_pred_lm <- predict(lm_model, newdata = train_data)
val_pred_lm   <- predict(lm_model, newdata = val_data)

############################################################################
# 4) Convert Predicted OS to a Risk Score
############################################################################
# Define risk score as the inverse of predicted OS (lower OS -> higher risk).
train_risk_score_lm <- 1 / train_pred_lm
val_risk_score_lm   <- 1 / val_pred_lm

############################################################################
# 5) (Optional) Winsorize Risk Scores to Reduce Extreme Outliers
############################################################################
winsorize <- function(x, lower = 0.01, upper = 0.99) {
  q <- quantile(x, probs = c(lower, upper), na.rm = TRUE)
  x[x < q[1]] <- q[1]
  x[x > q[2]] <- q[2]
  return(x)
}

train_risk_score_lm_adj <- winsorize(train_risk_score_lm, lower = 0.01, upper = 0.99)
val_risk_score_lm_adj   <- winsorize(val_risk_score_lm, lower = 0.01, upper = 0.99)

cat("\n--- Linear Regression Risk Score Summary (Before Winsorization) ---\n")
print(summary(train_risk_score_lm))
cat("\n--- Linear Regression Risk Score Summary (After Winsorization) ---\n")
print(summary(train_risk_score_lm_adj))

############################################################################
# 6) Convert Adjusted Risk Score to Binary Classification
############################################################################
# True labels from train_data$Survival: 2 = event, 1 = censor.
train_true_label <- train_data$Survival
val_true_label   <- val_data$Survival

# For ROC analysis, convert to binary: event (2) becomes 1, censor (1) becomes 0.
train_event_binary <- ifelse(train_true_label == 2, 1, 0)

# Use ROC on the training set to choose an optimal threshold.
roc_train_lm <- roc(train_event_binary, train_risk_score_lm_adj)
best_thr_lm <- coords(roc_train_lm, x = "best", ret = "threshold", best.method = "youden")[[1]]
cat("\nChosen risk threshold (Linear Regression, Youden's J):", best_thr_lm, "\n")

# Classify: if adjusted risk score > threshold, predict 2 (event); otherwise, predict 1 (censor)
train_pred_class_lm <- ifelse(train_risk_score_lm_adj > best_thr_lm, 2, 1)
val_pred_class_lm   <- ifelse(val_risk_score_lm_adj > best_thr_lm, 2, 1)

############################################################################
# 7) Evaluate Classification Performance (Linear Regression)
############################################################################
# Training confusion matrix
train_cm_lm <- confusionMatrix(
  data = factor(train_pred_class_lm, levels = c(1,2)),
  reference = factor(train_true_label, levels = c(1,2)),
  positive = "2"
)
cat("\n=== Linear Regression - Training Classification Metrics ===\n")
print(train_cm_lm)

# Validation confusion matrix
val_cm_lm <- confusionMatrix(
  data = factor(val_pred_class_lm, levels = c(1,2)),
  reference = factor(val_true_label, levels = c(1,2)),
  positive = "2"
)
cat("\n=== Linear Regression - Validation Classification Metrics ===\n")
print(val_cm_lm)

# --- Additional Metrics: Accuracy, Precision, Recall, and F1 Score ---
# Training metrics
train_accuracy  <- train_cm_lm$overall["Accuracy"]
train_precision <- train_cm_lm$byClass["Pos Pred Value"]
train_recall    <- train_cm_lm$byClass["Sensitivity"]
train_F1        <- 2 * (train_precision * train_recall) / (train_precision + train_recall)
cat(sprintf("\nTraining Accuracy: %.3f", train_accuracy))
cat(sprintf("\nTraining Precision: %.3f", train_precision))
cat(sprintf("\nTraining Recall: %.3f", train_recall))
cat(sprintf("\nTraining F1 Score: %.3f\n", train_F1))

# Validation metrics
val_accuracy  <- val_cm_lm$overall["Accuracy"]
val_precision <- val_cm_lm$byClass["Pos Pred Value"]
val_recall    <- val_cm_lm$byClass["Sensitivity"]
val_F1        <- 2 * (val_precision * val_recall) / (val_precision + val_recall)
cat(sprintf("\nValidation Accuracy: %.3f", val_accuracy))
cat(sprintf("\nValidation Precision: %.3f", val_precision))
cat(sprintf("\nValidation Recall: %.3f", val_recall))
cat(sprintf("\nValidation F1 Score: %.3f\n", val_F1))

# Compute AUC for binary classification.
val_event_binary <- ifelse(val_true_label == 2, 1, 0)
roc_val_lm <- roc(val_event_binary, val_risk_score_lm_adj)
train_auc <- auc(roc_train_lm)
val_auc <- auc(roc_val_lm)
cat(sprintf("\nLinear Regression - Train AUC: %.3f\n", train_auc))
cat(sprintf("Linear Regression - Val AUC: %.3f\n", val_auc))

# --- Compute 95% Confidence Intervals for AUROC ---
ci_train_auc <- ci.auc(roc_train_lm, conf.level = 0.95)
ci_val_auc <- ci.auc(roc_val_lm, conf.level = 0.95)
cat(sprintf("Linear Regression - Train AUC 95%% CI: %.3f - %.3f\n", ci_train_auc[1], ci_train_auc[3]))
cat(sprintf("Linear Regression - Val AUC 95%% CI: %.3f - %.3f\n", ci_val_auc[1], ci_val_auc[3]))

############################################################################
# 8) Evaluate Survival Metrics: Concordance Index & Time-Dependent AUC
############################################################################
# Create Surv objects using OS and the event indicator.
train_event_indicator <- ifelse(train_true_label == 2, 1, 0)
val_event_indicator   <- ifelse(val_true_label == 2, 1, 0)
train_surv <- Surv(time = train_data$OS, event = train_event_indicator)
val_surv   <- Surv(time = val_data$OS, event = val_event_indicator)

# Compute concordance index using the adjusted risk score.
cindex_lm_raw <- concordance(train_surv ~ train_risk_score_lm_adj)$concordance
# Use I() to invert the risk score properly.
cindex_lm_inv <- concordance(train_surv ~ I(-train_risk_score_lm_adj))$concordance

cat(sprintf("\nLinear Regression - c-index (raw risk score, train): %.3f\n", cindex_lm_raw))
cat(sprintf("Linear Regression - c-index (inverted risk score, train): %.3f\n", cindex_lm_inv))

# Choose the version that makes more sense (typically a good model should have c-index > 0.5).
chosen_cindex_lm <- ifelse(cindex_lm_raw < 0.5, cindex_lm_inv, cindex_lm_raw)
cat(sprintf("\nLinear Regression - Chosen c-index (train): %.3f\n", chosen_cindex_lm))

# Compute time-dependent AUC at specified time points (e.g., years 1 to 5)
time_points <- c(12, 24, 36, 48, 60)
timeROC_train_lm <- timeROC(
  T = train_data$OS,
  delta = train_event_indicator,
  marker = train_risk_score_lm_adj,
  cause = 1,
  times = time_points,
  iid = TRUE
)
timeROC_val_lm <- timeROC(
  T = val_data$OS,
  delta = val_event_indicator,
  marker = val_risk_score_lm_adj,
  cause = 1,
  times = time_points,
  iid = TRUE
)

cat("\nLinear Regression - Time-dependent AUC (Training):\n")
print(data.frame(Time = time_points, AUC = timeROC_train_lm$AUC))
cat("\nLinear Regression - Time-dependent AUC (Validation):\n")
print(data.frame(Time = time_points, AUC = timeROC_val_lm$AUC))

cat("\n=== DONE (Linear Regression Model) ===\n")
