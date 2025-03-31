############################################################################
# 10) Plot ROC Curves of All 4 Models & Perform DeLong Tests
############################################################################
library(pROC)  # Make sure pROC is loaded

# 1) Rename your ROC objects so it’s clearer which is which
roc_train_rsf <- train_auc           # from RSF
roc_val_rsf   <- validation_auc      # from RSF

roc_train_xgb <- roc_train           # from XGBoost
roc_val_xgb   <- roc_val            # from XGBoost

roc_train_dt  <- roc_train_dt       # from rpart (Decision Tree)
roc_val_dt    <- roc_val_dt         # from rpart (Decision Tree)

roc_train_lm  <- roc_train_lm       # from Linear Regression
roc_val_lm    <- roc_val_lm         # from Linear Regression


##############################
# A) PLOT TRAINING SET
##############################
# Open a new plot for the training ROC curves
plot(
  roc_train_rsf,
  main = "Training Set ROC Comparison (4 Models)",
  lwd  = 2
)
# Add the other three models’ ROC curves
lines(roc_train_xgb, col = "blue",  lwd = 2)
lines(roc_train_dt,  col = "green", lwd = 2)
lines(roc_train_lm,  col = "purple",lwd = 2)

# Add a legend so we know which line is which
legend(
  "bottomright",
  legend = c("RSF","XGBoost","Decision Tree","Linear Regression"),
  col    = c("black","blue","green","purple"),
  lwd    = 2
)

##############################
# B) PLOT VALIDATION SET
##############################
# Open a new plot window or reuse the same one, depending on preference.
# If you want a separate figure, you could do: dev.new()
plot(
  roc_val_rsf,
  main = "Validation Set ROC Comparison (4 Models)",
  lwd  = 2
)
# Add the other three models’ ROC curves
lines(roc_val_xgb, col = "blue",  lwd = 2)
lines(roc_val_dt,  col = "green", lwd = 2)
lines(roc_val_lm,  col = "purple",lwd = 2)

legend(
  "bottomright",
  legend = c("RSF","XGBoost","Decision Tree","Linear Regression"),
  col    = c("black","blue","green","purple"),
  lwd    = 2
)

##############################
# C) DELONG TESTS (Pairwise)
##############################
cat("\n===== DeLong Tests on Training Set ROC =====\n")
# RSF vs XGB
print( roc.test(roc_train_rsf, roc_train_xgb, method = "delong") )
# RSF vs DT
print( roc.test(roc_train_rsf, roc_train_dt,  method = "delong") )
# RSF vs LM
print( roc.test(roc_train_rsf, roc_train_lm,  method = "delong") )
# XGB vs DT
print( roc.test(roc_train_xgb, roc_train_dt,  method = "delong") )
# XGB vs LM
print( roc.test(roc_train_xgb, roc_train_lm,  method = "delong") )
# DT vs LM
print( roc.test(roc_train_dt,  roc_train_lm,  method = "delong") )


cat("\n===== DeLong Tests on Validation Set ROC =====\n")
# RSF vs XGB
print( roc.test(roc_val_rsf, roc_val_xgb, method = "delong") )
# RSF vs DT
print( roc.test(roc_val_rsf, roc_val_dt,  method = "delong") )
# RSF vs LM
print( roc.test(roc_val_rsf, roc_val_lm,  method = "delong") )
# XGB vs DT
print( roc.test(roc_val_xgb, roc_val_dt,  method = "delong") )
# XGB vs LM
print( roc.test(roc_val_xgb, roc_val_lm,  method = "delong") )
# DT vs LM
print( roc.test(roc_val_dt,  roc_val_lm,  method = "delong") )

cat("\n=== Finished ROC Plots & DeLong Tests ===\n")
