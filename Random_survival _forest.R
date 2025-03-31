############################################################################
# 4) Fit RSF Model
############################################################################
rsf_model <- rfsrc(
  Surv(OS, Survival) ~ .,
  data       = train_data_x,
  ntree      = 1000,
  nodesize   = 15,
  importance = TRUE
)

############################################################################
# 5) Predict on Training & Validation
############################################################################
# For survival, randomForestSRC::predict() often returns a matrix for 'predicted'.
# We can coerce it to a numeric vector if it is a 1-column matrix.
train_pred <- predict(rsf_model, newdata = train_data_x)
validation_pred <- predict(rsf_model, newdata = validation_data_x)

# Ensure they're numeric vectors
train_risk <- train_pred$predicted[, "event.2", drop = TRUE]
validation_risk <- validation_pred$predicted[, "event.2", drop = TRUE]


############################################################################
# 6) Convert true labels (Survival=1,2) into 0/1 for classification
############################################################################
# Assumption: Survival==2 is "event/death" => label=0
#             Survival==1 is "censor"       => label=1
true_train_labels      <- ifelse(train_data$Survival == 2, 1, 0)
true_validation_labels <- ifelse(validation_data$Survival == 2, 1, 0)

cat("Length of true_train_labels:", length(true_train_labels), "\n")
cat("Length of train_risk:", length(train_risk), "\n")
print(dim(train_pred$predicted))
print(head(train_pred$predicted))

############################################################################
# 7) Threshold at median risk for classification
############################################################################
roc_obj <- roc(true_train_labels, train_risk)
optimal_threshold <- coords(roc_obj, "best", ret = "threshold")[[1]]
train_labels <- ifelse(train_risk > optimal_threshold, 1, 0)
validation_labels <- ifelse(validation_risk > optimal_threshold, 1, 0)

print(optimal_threshold)

cat("Length of train_labels:", length(train_labels), "\n")
cat("Length of true_train_labels:", length(true_train_labels), "\n")


############################################################################
# 8) Calculate AUROC & 95% CI
############################################################################
train_auc <- roc(true_train_labels, train_risk)
validation_auc <- roc(true_validation_labels, validation_risk)

cat("Training AUROC:", auc(train_auc),
    "95% CI:", ci.auc(train_auc), "\n")
cat("Validation AUROC:", auc(validation_auc),
    "95% CI:", ci.auc(validation_auc), "\n\n")
############################################################################
# 9) Confusion Matrix & Other Metrics
############################################################################
train_cm <- confusionMatrix(
  data      = as.factor(train_labels),
  reference = as.factor(true_train_labels),
  positive  = "1"  # define "1" as the "positive" class
)

train_metrics <- c(
  Accuracy  = train_cm$overall["Accuracy"],
  Precision = train_cm$byClass["Precision"],
  Recall    = train_cm$byClass["Recall"],
  F1        = train_cm$byClass["F1"]
)

validation_cm <- confusionMatrix(
  data      = as.factor(validation_labels),
  reference = as.factor(true_validation_labels),
  positive  = "1"
)

validation_metrics <- c(
  Accuracy  = validation_cm$overall["Accuracy"],
  Precision = validation_cm$byClass["Precision"],
  Recall    = validation_cm$byClass["Recall"],
  F1        = validation_cm$byClass["F1"]
)

cat("Training Metrics:\n")
print(train_metrics)

cat("\nValidation Metrics:\n")
print(validation_metrics)

cat("\n=== DONE ===\n")

############################################################################
# Compute Concordance Index (c-index)
############################################################################
library(survival)

# Create Surv objects with proper event indicators (1=event, 0=censored)
train_surv <- with(train_data, Surv(OS, ifelse(Survival == 2, 1, 0)))
validation_surv <- with(validation_data, Surv(OS, ifelse(Survival == 2, 1, 0)))

# Compute the concordance index for training and validation sets.
# Here, higher risk scores should correspond to higher probability of event.
train_concordance <- survConcordance(train_surv ~ train_risk)
validation_concordance <- survConcordance(validation_surv ~ validation_risk)

# Print the c-index results
cat("Training C-index:", train_concordance$concordance, "\n")
cat("Validation C-index:", validation_concordance$concordance, "\n")

library(timeROC)

############################################################################
# Compute time-dependent AUC for 1-5 years
############################################################################
# Load the required package
library(timeROC)

# Recode the event indicator for time-dependent ROC analysis:
# Here, event (death) is coded as 1 and censoring as 0.
train_event <- ifelse(train_data$Survival == 2, 1, 0)
validation_event <- ifelse(validation_data$Survival == 2, 1, 0)

# Define time points (in years) at which to compute the ROC/AUC.
time_points <- 1:5  # Years 1, 2, 3, 4, and 5

# Compute the time-dependent ROC for the training data.
# T is the observed time (OS), delta is the event indicator,
# marker is the continuous risk score (using event.2, where higher risk means higher probability of death),
# cause = 1 indicates we are interested in the event coded as 1.
timeROC_train <- timeROC(T = train_data$OS,
                         delta = train_event,
                         marker = train_risk,
                         cause = 1,
                         times = time_points,
                         iid = TRUE)

# Compute the time-dependent ROC for the validation data.
timeROC_validation <- timeROC(T = validation_data$OS,
                              delta = validation_event,
                              marker = validation_risk,
                              cause = 1,
                              times = time_points,
                              iid = TRUE)

# Display the AUC values at each time point for both training and validation sets.
cat("Time-dependent AUC (Training):\n")
print(data.frame(Time = time_points, AUC = timeROC_train$AUC))

cat("\nTime-dependent AUC (Validation):\n")
print(data.frame(Time = time_points, AUC = timeROC_validation$AUC))


library(randomForestSRC)
library(survival)

# Assuming train_data_x is already defined with variables OS and Survival

set.seed(123)
rsf_model <- rfsrc(
  Surv(OS, Survival) ~ .,
  data = train_data_x,
  ntree = 1000,
  nodesize = 15,
  importance = TRUE
)

# Compute permutation-based VIMP without auto-plotting
vimp_permute <- vimp(rsf_model, method = "permute", plot = FALSE)

# Exclude the "validation" variable and keep only event2
vimp_permute$importance <- as.matrix(
  vimp_permute$importance[rownames(vimp_permute$importance) != "validation", "event.2", drop = FALSE]
)

# Open a PNG device for a high-resolution (300 dpi) plot (10x8 inches)
png("final_vimp_event2.png", width = 10, height = 8, units = "in", res = 300)

# Determine y-axis limits to clearly show both negative and positive values
lower_lim <- min(vimp_permute$importance, na.rm = TRUE) - 0.005
upper_lim <- max(vimp_permute$importance, na.rm = TRUE) + 0.005

# Plot the VIMP for event2 only
plot(vimp_permute, 
     ylim = c(lower_lim, upper_lim),
     cex.names = 1.2,
     cex.axis = 1.2,
     main = "Permutation-based VIMP for Event2",
     ylab = "VIMP Score",
     xlab = "Variables",
     lwd = 2)

dev.off()

# Load required packages for SHAP computations and visualization
library(kernelshap)
library(shapviz)

# Define a prediction function that extracts the risk for the event ("event.2")
pred_fun <- function(model, data) {
  preds <- predict(model, newdata = data)$predicted
  if ("event.2" %in% colnames(preds)) {
    return(preds[, "event.2"])
  } else {
    # Fallback: return the first column if naming is different
    return(preds[, 1])
  }
}

# Select predictor variables.
# Exclude outcome variables ("OS", "Survival") if present.
predictor_vars <- setdiff(colnames(train_data_x), c("OS", "Survival"))
X_explain <- train_data_x[, predictor_vars]

# (Optional) If the training set is large, sample up to 1000 rows for faster computation
if(nrow(X_explain) > 100) {
  set.seed(123)  # for reproducibility
  X_explain <- X_explain[sample(nrow(X_explain), 100), ]
}

# Compute SHAP values using kernelshap and convert to a shapviz object
sv <- kernelshap(rsf_model, X = X_explain, pred_fun = pred_fun) %>% 
  shapviz()

# Plot a bee swarm importance plot of the SHAP values
sv %>% sv_importance(kind = "bee")

# Optionally, create a dependence plot for the top predictor
importance_df <- sv_importance(sv, plot = FALSE)
top_var <- importance_df$variable[which.max(importance_df$mean_abs)]
sv %>% sv_dependence(xvar = top_var)
