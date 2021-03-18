library(tidyverse)
library(SGL)
library(gglasso)
library(glmnet)
library(seagull)
library(ClusterR)
library(reshape2)


# load script with functions
source(paste0(getwd(),'/functions.R'))

# ******************************************************************************
# ** Introductory Examples  ****************************************************
# ******************************************************************************

## Factor data

data <- factor_sampler(levelsnr = 6, nrgroups = 20, samplesize = 100,
                       grsparsity = 0.8, withinsparsity = 0)
plot(data$coefs_vec, col = "blue", type = "h", ylab = "")

# run Lasso model
lasso <- cv.glmnet(x = data$x, y = data$y, family = "gaussian")
# Mean squared error minimizing model through Cross validation
plot(lasso)
# Coefficient profiles
plot(lasso$glmnet.fit, xvar="lambda")
# selected coefficients
plot(as.vector(coef(lasso, s="lambda.min")), col = "red", type = "h")


# run group lasso model
gglasso <- cv.gglasso(x = data$x, y = data$y, group = data$group_structure)
# Mean squared error minimizing model through Cross validation
plot(gglasso)
# Coefficient profiles
plot(gglasso$gglasso.fit)
# selected coefficients
plot(as.vector(coef(gglasso, s="lambda.min")), col = "red", type = "h")




## Correlated data

data <- correlated_sampler(levelsnr = 50,
                           nrgroups = 10,
                           samplesize = 100,
                           grsparsity = 0.8,
                           withinsparsity = 0,
                           corr = 0.2)
data$corrplot + xlab("") + ylab("")
plot(data$coefs_vec, col = "blue", type = "h", ylab = "")

# run Lasso model
lasso <- cv.glmnet(x = data$x, y = data$y, family = "gaussian")
# Mean squared error minimizing model through Cross validation
plot(lasso)
# Coefficient profiles
plot(lasso$glmnet.fit, xvar="lambda")
# selected coefficients
plot(as.vector(coef(lasso, s="lambda.min")), col = "red", type = "h")


# run group lasso model
gglasso <- cv.gglasso(x = data$x, y = data$y, group = data$group_structure)
# Mean squared error minimizing model through Cross validation
plot(gglasso)
# Coefficient profiles
plot(gglasso$gglasso.fit)
# selected coefficients
plot(as.vector(coef(gglasso, s="lambda.min")), col = "red", type = "h")



# ******************************************************************************
# ** Evaluation I   ************************************************************
# ******************************************************************************

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# what to evaluate?
# factor variable dataset along dimensions:
# nr. groups, group size, group sparsity, within group sparsity

# models to use:
# Lasso (glmnet), Group Lasso (gglasso), Sparse Group Lasso (seagull)

# evaluation through:
# L0 norm, RMSE (evaluate for lambda min)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


## define ranges of parameters
levelrange <- c(4, 6, 10, 15, 30,50)
grsparsityrange <- c(0.25, 0.5, 0.75)
withinsparsityrange <- c(0, 0.5)
grouprange <- c(seq(4, 100, by=1))

## create dataframe where outputs will be stored
results_l0 <- data.frame(group_nr = grouprange,
                         nonzero.coefs = NA,
                         lasso_L0= NA,
                         lasso_L0_1se = NA,
                         gglasso_L0 = NA,
                         gglasso_L0_1se = NA)
results_rmse <- data.frame(group_nr = grouprange,
                           lasso_RMSE = NA,
                           lasso_RMSE_1se = NA,
                           gglasso_RMSE = NA,
                           gglasso_RMSE_1se = NA)

## start simulation
for (levelsnr in levelrange){
print(paste("level:", levelsnr))
for (grsparsity in grsparsityrange){
print(paste("grspars:", grsparsity))
for (withinsparsity in withinsparsityrange){
print(paste("within:", withinsparsity))
for (g in grouprange){
  print(paste("running models for groupsize:", g))

  ## Data
  data <- factor_sampler(levelsnr, g, samplesize*2, grsparsity, withinsparsity)
  # partition in training and out of sample dataset
  train_data_x <- data$x[1:samplesize,]
  train_data_y <- data$y[1:samplesize]
  oos_data_x <- data$x[(samplesize+1):(samplesize*2),]
  oos_data_y <- data$y[(1+samplesize):(samplesize*2)]

  # Nonzero coefs
  results_l0$nonzero.coefs[results_l0$group_nr == g] <- data$nonzerocoeffs

  ## Lasso
  # model
  lasso <- cv.glmnet(train_data_x, train_data_y)
  # l0 norm
  # MSE minimizing lambda
  lasso_coefs <- as.vector(coef(lasso, s="lambda.min"))
  lasso_l0 <- length(lasso_coefs[lasso_coefs != 0])
  results_l0$lasso_L0[results_l0$group_nr == g] <- lasso_l0
  # 1se lambda
  lasso_coefs_1se <- as.vector(coef(lasso, s="lambda.1se"))
  lasso_l0_1se <- length(lasso_coefs_1se[lasso_coefs_1se != 0])
  results_l0$lasso_L0_1se[results_l0$group_nr == g] <- lasso_l0_1se
  # RMSE
  # MSE minimizing lambda
  lasso_pred <- drop(predict(lasso, newx=oos_data_x, s="lambda.min"))
  lasso_rmse <- sqrt(sum((oos_data_y - lasso_pred)^2)/length(lasso_pred))
  results_rmse$lasso_RMSE[results_rmse$group_nr == g] <- lasso_rmse
  # 1se lambda
  lasso_pred_1se <- drop(predict(lasso, newx=oos_data_x, s="lambda.1se"))
  lasso_rmse_1se <- sqrt(sum((oos_data_y - lasso_pred_1se)^2)/length(lasso_pred_1se))
  results_rmse$lasso_RMSE_1se[results_rmse$group_nr == g] <- lasso_rmse_1se

  ## Group Lasso
  # model
  gglasso <- cv.gglasso(train_data_x, train_data_y, loss="ls", group = data$group_structure)
  # l0 norm
  # MSE minimizing lambda
  gglasso_coefs <- as.vector(coef(gglasso, s="lambda.min"))
  gglasso_l0 <- length(gglasso_coefs[gglasso_coefs != 0])
  results_l0$gglasso_L0[results_l0$group_nr == g] <- gglasso_l0
  # 1se lambda
  gglasso_coefs_1se <- as.vector(coef(gglasso, s="lambda.1se"))
  gglasso_l0_1se <- length(gglasso_coefs_1se[gglasso_coefs_1se != 0])
  results_l0$gglasso_L0_1se[results_l0$group_nr == g] <- gglasso_l0_1se
  # RMSE
  # MSE optimizing lambda
  gglasso_pred <- drop(predict(gglasso, newx=oos_data_x, s="lambda.min"))
  gglasso_rmse <- sqrt(sum((oos_data_y - gglasso_pred)^2)/length(gglasso_pred))
  results_rmse$gglasso_RMSE[results_rmse$group_nr == g] <- gglasso_rmse
  # 1se lambda
  gglasso_pred_1se <- drop(predict(gglasso, newx=oos_data_x, s="lambda.1se"))
  gglasso_rmse_1se <- sqrt(sum((oos_data_y - gglasso_pred_1se)^2)/length(gglasso_pred_1se))
  results_rmse$gglasso_RMSE_1se[results_rmse$group_nr == g] <- gglasso_rmse_1se

}

## visualize results
line <- c("lasso_L0", "gglasso_L0", "lasso_RMSE", "gglasso_RMSE")
truth <- c("nonzero.coefs")
color <- c("lasso_L0", "lasso_L0_1se", "lasso_RMSE", "lasso_RMSE_1se")

plot_l0 <- results_l0 %>%
  pivot_longer(!group_nr, values_to = "L0_norm", names_to = "Model") %>%
  mutate(Lambda = ifelse(Model %in% line, "min", "1se"),
         Model = ifelse(Model %in% color, "Lasso",
                        ifelse(Model == "nonzero.coefs", "Simulated", "GroupLasso")))

ggplot(NULL) +
  geom_line(data = plot_l0 %>% filter(Model == "Lasso"),
            aes(x = group_nr, y = L0_norm, linetype = Lambda,
            color = "a"), size = 1) +
  geom_line(data = plot_l0 %>% filter(Model == "GroupLasso"),
            aes(x = group_nr, y = L0_norm, linetype = Lambda,
            color = "b"), size = 1) +
  geom_line(data = plot_l0 %>% filter(Model == "Simulated"),
            aes(x = group_nr, y = L0_norm, color = "c"),
            size = 1) +
  scale_linetype_manual(values = c("dotted", "solid")) +
  scale_color_brewer(palette = "Set2", name = "Model",
                     labels = c("Lasso", "Group Lasso", "Simulated")) +
  labs(caption = paste0("Data structure: Factors; \n",
                       "Parameters: Group size: ", levelsnr,
                       ", Sample size: ", samplesize,
                       ", Group sparsity: ", grsparsity*100,
                       "% , Within sparsity: ", withinsparsity*100, "%")) +
  xlab("Number of groups") +
  theme_minimal() +
  ggsave(paste0(getwd(), "/Factor_results/", "L0_", levelsnr, "-", grsparsity, "-",
                         withinsparsity, ".png"))

plot_RMSE <- results_rmse %>%
  pivot_longer(!group_nr, values_to = "RMSE", names_to = "Model") %>%
  mutate(Lambda = ifelse(Model %in% line, "min", "1se"),
         Model = ifelse(Model %in% color, "Lasso","GroupLasso"))

ggplot(NULL) +
  geom_line(data = plot_RMSE %>% filter(Model == "Lasso"),
            aes(x = group_nr, y = RMSE, linetype = Lambda,
            color = "a"), size = 1) +
  geom_line(data = plot_RMSE %>% filter(Model == "GroupLasso"),
            aes(x = group_nr, y = RMSE, linetype = Lambda,
            color = "b"), size = 1) +
  scale_linetype_manual(values = c("dotted", "solid")) +
  scale_color_brewer(palette = "Set2", name = "Model",
                     labels = c("Lasso", "Group Lasso")) +
  labs(caption = paste0("Data structure: Factors; \n",
                       "Parameters: Group size: ", levelsnr,
                       ", Sample size: ", samplesize,
                       ", Group sparsity: ", grsparsity*100,
                       "% , Within sparsity: ", withinsparsity*100, "%")) +
  xlab("Number of groups") +
  theme_minimal() +
  ggsave(paste0(getwd(), "/Factor_results/", "RMSE_", levelsnr, "-", grsparsity, "-",
                         withinsparsity, ".png"))

}
}
}







# ******************************************************************************
# ** Evaluation II  ************************************************************
# ******************************************************************************

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# what to evaluate?
# correlated variable dataset along dimensions:
# nr. groups, group size, group sparsity, within group sparsity

# models to use:
# Lasso (glmnet), Group Lasso (gglasso), Sparse Group Lasso (seagull)

# evaluation through:
# L0 norm, RMSE (evaluate for lambda min)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


## define ranges of parameters
levelrange <- c(6, 10, 15, 30,50)
grsparsityrange <- c(0.25, 0.5, 0.75)
withinsparsityrange <- c(0, 0.5)
corr <- 0.2
grouprange <- c(seq(4, 100, by=2))
alpharange <- c(0.3,0.9)

samplesize <- 100

## create dataframe where outputs will be stored
results_l0 <- data.frame(group_nr = grouprange,
                         nonzero.coefs = NA,
                         lasso_L0= NA,
                         gglasso_L0 = NA,
                         sglasso_L0 = NA)
results_rmse <- data.frame(group_nr = grouprange,
                           lasso_RMSE = NA,
                           gglasso_RMSE = NA,
                           sglasso_RMSE = NA)
for (levelsnr in levelrange){
print(paste("level:", levelsnr))
for (grsparsity in grsparsityrange){
print(paste("grspars:", grsparsity))
for (withinsparsity in withinsparsityrange){
print(paste("within:", withinsparsity))
for (corr in correlationrange){
print(paste("corr:", corr))
for (alpha in alpharange){
print(paste("alpha:", alpha))
for (g in grouprange){
  print(paste("running models for groupsize:", g))
  ## Data
  data <- correlated_sampler(levelsnr, g, samplesize*2, grsparsity, withinsparsity, corr)
  # partition in training and out of sample
  train_data_x <- data$x[1:samplesize,]
  train_data_y <- data$y[1:samplesize]
  oos_data_x <- data$x[(samplesize+1):(samplesize*2),]
  oos_data_y <- data$y[(1+samplesize):(samplesize*2)]

  # Nonzero coefs
  results_l0$nonzero.coefs[results_l0$group_nr == g] <- data$nonzerocoeffs

  ## Lasso
  # model
  lasso <- cv.glmnet(train_data_x, train_data_y)
  # l0 norm
  # MSE minimizing lambda
  lasso_coefs <- as.vector(coef(lasso, s="lambda.min"))
  lasso_l0 <- length(lasso_coefs[lasso_coefs != 0])
  results_l0$lasso_L0[results_l0$group_nr == g] <- lasso_l0
  # RMSE
  # MSE minimizing lambda
  lasso_pred <- drop(predict(lasso, newx=oos_data_x, s="lambda.min"))
  lasso_rmse <- sqrt(sum((oos_data_y - lasso_pred)^2)/length(lasso_pred))
  results_rmse$lasso_RMSE[results_rmse$group_nr == g] <- lasso_rmse

  ## Group Lasso
  # model
  gglasso <- cv.gglasso(train_data_x, train_data_y, loss="ls", group = data$group_structure)
  # l0 norm
  # MSE minimizing lambda
  gglasso_coefs <- as.vector(coef(gglasso, s="lambda.min"))
  gglasso_l0 <- length(gglasso_coefs[gglasso_coefs != 0])
  results_l0$gglasso_L0[results_l0$group_nr == g] <- gglasso_l0
  # RMSE
  # MSE minimizing lambda
  gglasso_pred <- drop(predict(gglasso, newx=oos_data_x, s="lambda.min"))
  gglasso_rmse <- sqrt(sum((oos_data_y - gglasso_pred)^2)/length(gglasso_pred))
  results_rmse$gglasso_RMSE[results_rmse$group_nr == g] <- gglasso_rmse

  ## Sparse Group Lasso
  # model
  sglasso <- seagull(y = train_data_y, Z = train_data_x, alpha = alpha,
                     groups = data$group_structure)
  # get coefficients at MSE minimizing lambda
  beta.seagull <- sglasso$random_effects
  yhat.seagull <- mse.seagull <- c()
  for(i in seq_len(nrow(beta.seagull))){
    y.seagull      <- oos_data_x %*% beta.seagull[i, ]
    mse.seagull[i] <- mean((y.seagull - oos_data_y) ^ 2)
    yhat.seagull   <- cbind(yhat.seagull, y.seagull)
  }
  ## Extract the estimates of marker effects with smallest mean squared error.
  sglasso_coefs <- beta.seagull[which.min(mse.seagull), ]
  # l0 norm
  sglasso_l0 <- length(sglasso_coefs[sglasso_coefs != 0])
  results_l0$sglasso_L0[results_l0$group_nr == g] <- sglasso_l0
  # RMSE
  sglasso_rmse <- sqrt(min(mse.seagull))
  results_rmse$sglasso_RMSE[results_rmse$group_nr == g] <- sglasso_rmse
}

truth <- c("nonzero.coefs")
las <- c("lasso_L0", "lasso_RMSE")
gglas <- c("gglasso_L0", "gglasso_RMSE")
sglas <- c("sglasso_L0", "sglasso_RMSE")

plot_l0 <- results_l0 %>%
  pivot_longer(!group_nr, values_to = "L0_norm", names_to = "Model") %>%
  mutate(Model = ifelse(Model %in% las, "Lasso",
                    ifelse(Model %in% gglas, "GroupLasso",
                       ifelse(Model %in% sglas, "SparseGroupLasso", "Simulated"))))

ggplot(NULL) +
  geom_line(data = plot_l0 %>% filter(Model == "Lasso"),
            aes(x = group_nr, y = L0_norm, color = "a"), size = 1) +
  geom_line(data = plot_l0 %>% filter(Model == "GroupLasso"),
            aes(x = group_nr, y = L0_norm, color = "b"), size = 1) +
  geom_line(data = plot_l0 %>% filter(Model == "Simulated"),
            aes(x = group_nr, y = L0_norm, color = "c"), size = 1) +
  geom_line(data = plot_l0 %>% filter(Model == "SparseGroupLasso"),
            aes(x = group_nr, y = L0_norm, color = "d"), size = 1) +
  scale_color_manual(values = c("#66C2A5","#FC8D62","#8DA0CB","#E78AC3"), name = "Model",
                     labels = c("Lasso", "Group Lasso", "Simulated", "Sparse Group Lasso")) +
  labs(caption = paste0("Data structure: Correlated Variables; \n",
                       "Parameters: Group size: ", levelsnr,
                       ", Sample size: ", samplesize,
                       ", Group sparsity: ", grsparsity*100,
                       "% , Within sparsity: ", withinsparsity*100,
                        "%, Independent error: ", corr,
                       ", alpha: ", alpha)) +
  xlab("Number of groups") +
  theme_minimal() +
  ggsave(paste0(getwd(), "/Correlated_results/", "L0_", levelsnr, "-", grsparsity, "-",
                         withinsparsity, "-", corr, "-alpha", alpha, ".png"))

plot_RMSE <- results_rmse %>%
  pivot_longer(!group_nr, values_to = "RMSE", names_to = "Model") %>%
  mutate(Model = ifelse(Model %in% las, "Lasso",
                    ifelse(Model %in% gglas, "GroupLasso",
                       ifelse(Model %in% sglas, "SparseGroupLasso", "Simulated"))))

ggplot(NULL) +
  geom_line(data = plot_RMSE %>% filter(Model == "Lasso"),
            aes(x = group_nr, y = RMSE, color = "a"), size = 1) +
  geom_line(data = plot_RMSE %>% filter(Model == "GroupLasso"),
            aes(x = group_nr, y = RMSE,  color = "b"), size = 1) +
  geom_line(data = plot_RMSE %>% filter(Model == "SparseGroupLasso"),
            aes(x = group_nr, y = RMSE,  color = "c"), size = 1) +
  scale_color_manual(values = c("#66C2A5", "#FC8D62", "#E78AC3"), name = "Model",
                     labels = c("Lasso", "Group Lasso", "SparseGroupLasso")) +
  labs(caption = paste0("Data structure: Correlated Variables; \n",
                       "Parameters: Group size: ", levelsnr,
                       ", Sample size: ", samplesize,
                       ", Group sparsity: ", grsparsity*100,
                       "% , Within sparsity: ", withinsparsity*100,
                        "%, Independent error: ", corr,
                       ", alpha: ", alpha)) +
  xlab("Number of groups") +
  theme_minimal() +
  ggsave(paste0(getwd(), "/Correlated_results/", "RMSE_", levelsnr, "-", grsparsity, "-",
                         withinsparsity, "-", corr, "-alpha", alpha, ".png"))

}
}
}
}
}



# ******************************************************************************
# ** Evaluation III  ***********************************************************
# ******************************************************************************

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# what to evaluate?
# factor variable dataset along dimensions:
# nr. groups, group size, group sparsity, within group sparsity

# models to use:
# Lasso (glmnet), Group Lasso (gglasso), Sparse Group Lasso (seagull)

# evaluation through:
# L0 norm, RMSE (evaluate for lambda min)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


## define ranges of parameters
levelrange <- c(4, 6, 10, 15, 30,50)
grsparsityrange <- c(0.25, 0.5, 0.75)
withinsparsityrange <- c(0, 0.5)
grouprange <- c(seq(4, 100, by=1))
alpharange <- c(0.3,0.6,0.9)

## create dataframe where outputs will be stored
results_l0 <- data.frame(group_nr = grouprange,
                         nonzero.coefs = NA,
                         lasso_L0= NA,
                         gglasso_L0 = NA,
                         sglasso_L0 = NA)
results_rmse <- data.frame(group_nr = grouprange,
                           lasso_RMSE = NA,
                           gglasso_RMSE = NA,
                           sglasso_RMSE = NA)
for (levelsnr in levelrange){
print(paste("level:", levelsnr))
for (grsparsity in grsparsityrange){
print(paste("grspars:", grsparsity))
for (withinsparsity in withinsparsityrange){
print(paste("within:", withinsparsity))
for (corr in correlationrange){
print(paste("corr:", corr))
for (alpha in alpharange){
print(paste("alpha:", alpha))
for (g in grouprange){
  print(paste("running models for groupsize:", g))
  ## Data
  data <- factor_sampler(levelsnr, g, samplesize*2, grsparsity, withinsparsity)
  # partition in training and out of sample
  train_data_x <- data$x[1:samplesize,]
  train_data_y <- data$y[1:samplesize]
  oos_data_x <- data$x[(samplesize+1):(samplesize*2),]
  oos_data_y <- data$y[(1+samplesize):(samplesize*2)]

  # Nonzero coefs
  results_l0$nonzero.coefs[results_l0$group_nr == g] <- data$nonzerocoeffs

  ## Lasso
  # model
  lasso <- cv.glmnet(train_data_x, train_data_y)
  # l0 norm
  # MSE minimizing lambda
  lasso_coefs <- as.vector(coef(lasso, s="lambda.min"))
  lasso_l0 <- length(lasso_coefs[lasso_coefs != 0])
  results_l0$lasso_L0[results_l0$group_nr == g] <- lasso_l0
  # RMSE
  # MSE minimizing lambda
  lasso_pred <- drop(predict(lasso, newx=oos_data_x, s="lambda.min"))
  lasso_rmse <- sqrt(sum((oos_data_y - lasso_pred)^2)/length(lasso_pred))
  results_rmse$lasso_RMSE[results_rmse$group_nr == g] <- lasso_rmse

  ## Group Lasso
  # model
  gglasso <- cv.gglasso(train_data_x, train_data_y, loss="ls", group = data$group_structure)
  # l0 norm
  # MSE minimizing lambda
  gglasso_coefs <- as.vector(coef(gglasso, s="lambda.min"))
  gglasso_l0 <- length(gglasso_coefs[gglasso_coefs != 0])
  results_l0$gglasso_L0[results_l0$group_nr == g] <- gglasso_l0
  # RMSE
  # MSE minimizing lambda
  gglasso_pred <- drop(predict(gglasso, newx=oos_data_x, s="lambda.min"))
  gglasso_rmse <- sqrt(sum((oos_data_y - gglasso_pred)^2)/length(gglasso_pred))
  results_rmse$gglasso_RMSE[results_rmse$group_nr == g] <- gglasso_rmse

  ## Sparse Group Lasso
  # model
  sglasso <- seagull(y = train_data_y, Z = train_data_x, alpha = alpha,
                     groups = data$group_structure)
  # get coefficients at MSE minimizing lambda
  beta.seagull <- sglasso$random_effects
  yhat.seagull <- mse.seagull <- c()
  for(i in seq_len(nrow(beta.seagull))){
    y.seagull      <- oos_data_x %*% beta.seagull[i, ]
    mse.seagull[i] <- mean((y.seagull - oos_data_y) ^ 2)
    yhat.seagull   <- cbind(yhat.seagull, y.seagull)
  }
  ## Extract the estimates of marker effects with smallest mean squared error.
  sglasso_coefs <- beta.seagull[which.min(mse.seagull), ]
  # l0 norm
  sglasso_l0 <- length(sglasso_coefs[sglasso_coefs != 0])
  results_l0$sglasso_L0[results_l0$group_nr == g] <- sglasso_l0
  # RMSE
  sglasso_rmse <- sqrt(min(mse.seagull))
  results_rmse$sglasso_RMSE[results_rmse$group_nr == g] <- sglasso_rmse
}

truth <- c("nonzero.coefs")
las <- c("lasso_L0", "lasso_RMSE")
gglas <- c("gglasso_L0", "gglasso_RMSE")
sglas <- c("sglasso_L0", "sglasso_RMSE")

plot_l0 <- results_l0 %>%
  pivot_longer(!group_nr, values_to = "L0_norm", names_to = "Model") %>%
  mutate(Model = ifelse(Model %in% las, "Lasso",
                    ifelse(Model %in% gglas, "GroupLasso",
                       ifelse(Model %in% sglas, "SparseGroupLasso", "Simulated"))))

ggplot(NULL) +
  geom_line(data = plot_l0 %>% filter(Model == "Lasso"),
            aes(x = group_nr, y = L0_norm, color = "a"), size = 1) +
  geom_line(data = plot_l0 %>% filter(Model == "GroupLasso"),
            aes(x = group_nr, y = L0_norm, color = "b"), size = 1) +
  geom_line(data = plot_l0 %>% filter(Model == "Simulated"),
            aes(x = group_nr, y = L0_norm, color = "c"), size = 1) +
  geom_line(data = plot_l0 %>% filter(Model == "SparseGroupLasso"),
            aes(x = group_nr, y = L0_norm, color = "d"), size = 1) +
  scale_color_manual(values = c("#66C2A5","#FC8D62","#8DA0CB","#E78AC3"), name = "Model",
                     labels = c("Lasso", "Group Lasso", "Simulated", "Sparse Group Lasso")) +
  labs(caption = paste0("Data structure: Correlated Variables; \n",
                       "Parameters: Group size: ", levelsnr,
                       ", Sample size: ", samplesize,
                       ", Group sparsity: ", grsparsity*100,
                       "% , Within sparsity: ", withinsparsity*100,
                        "%, Independent error: ", corr,
                       ", alpha: ", alpha)) +
  xlab("Number of groups") +
  theme_minimal() +
  ggsave(paste0(getwd(), "/Factor_results/", "L0_", levelsnr, "-", grsparsity, "-",
                         withinsparsity, "-", corr, "-alpha", alpha, ".png"))

plot_RMSE <- results_rmse %>%
  pivot_longer(!group_nr, values_to = "RMSE", names_to = "Model") %>%
  mutate(Model = ifelse(Model %in% las, "Lasso",
                    ifelse(Model %in% gglas, "GroupLasso",
                       ifelse(Model %in% sglas, "SparseGroupLasso", "Simulated"))))

ggplot(NULL) +
  geom_line(data = plot_RMSE %>% filter(Model == "Lasso"),
            aes(x = group_nr, y = RMSE, color = "a"), size = 1) +
  geom_line(data = plot_RMSE %>% filter(Model == "GroupLasso"),
            aes(x = group_nr, y = RMSE,  color = "b"), size = 1) +
  geom_line(data = plot_RMSE %>% filter(Model == "SparseGroupLasso"),
            aes(x = group_nr, y = RMSE,  color = "c"), size = 1) +
  scale_color_manual(values = c("#66C2A5", "#FC8D62", "#E78AC3"), name = "Model",
                     labels = c("Lasso", "Group Lasso", "SparseGroupLasso")) +
  labs(caption = paste0("Data structure: Correlated Variables; \n",
                       "Parameters: Group size: ", levelsnr,
                       ", Sample size: ", samplesize,
                       ", Group sparsity: ", grsparsity*100,
                       "% , Within sparsity: ", withinsparsity*100,
                        "%, Independent error: ", corr,
                       ", alpha: ", alpha)) +
  xlab("Number of groups") +
  theme_minimal() +
  ggsave(paste0(getwd(), "/Factor_results/", "RMSE_", levelsnr, "-", grsparsity, "-",
                         withinsparsity, "-", corr, "-alpha", alpha, ".png"))


}
}
}
}
}