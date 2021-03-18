# Title     : TODO
# Objective : TODO
# Created by: Luca Poll
# Created on: 3/15/2021




################################################################################
#   FACTOR VARIABLE DATASET
################################################################################

## what happens in the factor sampler function?
## 1) J factors with p_j factor levels are created for a sample size n
## 2) coefficients are randomly drawn from a uniform distribution (-5,5)
## 3) depending on the level of group sparsity and within group sparsity
##    coefficients are set to zero
## 4) dummy variables created from factors are standardized, coefficients too
## 5) Y is simulated given the standardized covariates and coefficients from
##    Y = X \theta + 0.1 *N(0,1)
## 6) Y is demeaned



# sampler function
factor_sampler <- function (levelsnr, nrgroups, samplesize,
                            grsparsity, withinsparsity){

  # create dummy matrix
  base <- expand.grid(letters, letters)
  base <- as.vector(paste0(base$Var1, base$Var2)) # list with all possible factor levels
  sample <- data.frame(
    replicate(n=nrgroups,  # repeat n times depending on number of groups
         sample(base[1:levelsnr], size = samplesize, replace = T),  # generate factor variable for given number of fator levels and sample size
         simplify = T))
  data <- fastDummies::dummy_cols(sample) # TODO

  # Set of relevant groups (min 2, above given level of sparsity)
  # choose if nonzero groups should be random or the first n groups
  relevant_groups_random <- append(c(1,1), rbinom(nrgroups-2,1,1-grsparsity))
  relevant_groups <- c(rep(1, floor(0.5 + (1-grsparsity)*nrgroups)), rep(0, floor(0.5 + grsparsity*nrgroups)))
  # simulate coefficients
  # note: if many levels not all levels might be observed in data. Coeffs will
  # be assigned NA and will be dropped later
  coefs <- matrix(nrow = levelsnr, ncol = nrgroups) #TODO omitted dummy
  observed_factorlevels <- c()
  for (i in 1:nrgroups){
    observed_factorlevels[i] <- length(table(data[,i]))  # TODO
    values <- runif(length(table(data[,i])), -5, 5) # TODO
    coefs[,i] <- append(values, rep(NA, levelsnr - length(values))) # TODO
  }

  # account for sparsity (note that not all levels of factors might be observed)
  # group sparsity
  for (i in 1:nrgroups){
    if (relevant_groups[i] == 0){
      for (j in 1:(levelsnr)){ # TODO
        if (!is.na(coefs[j,i])){
          coefs[j,i] <- 0
        }
      }
    }
  }
  # within group sparsity (note that relevant level randomly assinged per factor)
  relevant_levels <- list()
  for (i in 1:nrgroups){
    levl_rel <- rbinom(levelsnr, 1, 1-withinsparsity) # TODO
    relevant_levels[[i]] <- levl_rel
    for (j in 1:(levelsnr)) # TODO
      if (levl_rel[j] == 0 & !is.na(coefs[j,i])){
      coefs[j,i] <- 0
      }
    }


  # define vector of coefficients
  coefs_vec <- as.vector(coefs)
  coefs_vec <- coefs_vec[!is.na(coefs_vec)]

  # get covariate matrix
  dummymatrix <- as.matrix(data[-c(1:nrgroups)])

  # center
  dummymatrix <- center_scale(dummymatrix)
  scalar1 <- function(x) {x / sqrt(sum(x^2))}
  coefs_vec <- scalar1(coefs_vec)

  # simulate dependent variable
  Y <- dummymatrix %*% coefs_vec + 0.1 * rnorm(samplesize, 0, 1)
  # Y <- Y - mean(Y)

  # create vector with group structure
  grp_struct <- vector()
  for (g in 1:nrgroups){
    grp_struct <- append(grp_struct, rep(g, observed_factorlevels[g]))
  }

  # create output list
  data <- list(x = dummymatrix, y = as.vector(Y), coefs_vec = coefs_vec,
               coefs = coefs, nonzerogroups = sum(relevant_groups),
               nonzerocoeffs = sum(length(coefs_vec[coefs_vec != 0])),
               groups = relevant_groups, levels = relevant_levels,
               group_structure = grp_struct, observed_factors = observed_factorlevels)
  return(data)
}







################################################################################
#   CORRELATED VARIABLES DATASET
################################################################################

## what happens in the correlated sampler function?
## 1) covariates are sampled where for every group one variable is defined from
##    an iid N(0,1). The correlated variables within the same group are defined
##    as p_j = p_1 + \sum_{i=2}^j p_i + z_j where p_j and z_j ~ N(0,1)
## 2) coefficients for all p are drawn from a uniform distr (-5,5)
## 3) depening on level of sparsities are coefficients set to 0
## 4) covariates and coefs normalized
## 5) Y simulated from Y = X \theta + 0.1 \varepsilon


correlated_sampler <- function (levelsnr, nrgroups, samplesize,
                            grsparsity, withinsparsity, corr){

  # covariate matrix for J groups with p_j levels
  covariates <- matrix(nrow = samplesize, ncol = nrgroups*levelsnr)
  # assign one variable per group that is iid and generate p_j correlated variables
  for (J in seq(1, nrgroups*levelsnr, by=levelsnr)){
    covariates[,J] <- rnorm(samplesize)
    for (p in 1:(levelsnr-1)){
      covariates[,(J+p)] <- covariates[,J+p-1] + corr * rnorm(samplesize)
    }
  }

  # coefficients
  coefs_mat <- matrix(runif(levelsnr*nrgroups, -5, 5), nrow = levelsnr, ncol = nrgroups)
  # account for sparsity
  # group sparsity
  relevant_groups <- c(rep(1, floor(0.5 + (1-grsparsity)*nrgroups)), rep(0, floor(0.5 + grsparsity*nrgroups)))
  # within group sparsity
  relevant_levels <- replicate(n=nrgroups, rbinom(levelsnr, 1, 1-withinsparsity))
  # relevant_levels <- replicate(n=nrgroups, rep(1, floor(0.5 + (1-withinsparsity)*levelsnr)), rep(0, floor(0.5 + withinsparsity*levelsnr)))
  # combine
  for (J in 1:nrgroups){
    if (relevant_groups[J] == 0){
      relevant_levels[,J] <- 0
    }
  }
  coefs_vec <- as.vector(coefs_mat) * as.vector(relevant_levels)

  # standardize
  covariates <- center_scale(covariates)
  scalar1 <- function(x) {x / sqrt(sum(x^2))}
  coefs_vec <- scalar1(coefs_vec)

  # simulate Y
  Y <- covariates %*% coefs_vec + 0.1*rnorm(samplesize)
  # demean Y
  # Y <- Y - mean(Y)

  # create vector with group structure
  grp_struct <- rep(1:nrgroups, each = levelsnr)

  # plot for correlation structure
  cormat <- round(cor(covariates),2)
  melted_cormat <- melt(cormat)
  corrplot <- ggplot(data = melted_cormat, aes(x=Var1, y=Var2, fill=value)) +
    geom_tile() + theme_minimal()

  # create output list
  data <- list(x = covariates, y = as.vector(Y), coefs_vec = coefs_vec,
               coefs = coefs_mat, nonzerogroups = sum(relevant_groups),
               nonzerocoeffs = sum(length(coefs_vec[coefs_vec != 0])),
               groups = relevant_groups, levels = relevant_levels,
               group_structure = grp_struct, observed_factors = NA,
               corrplot = corrplot)
  return(data)
}




################################################################################
#   COMBINED DATASET
################################################################################

## what happens in the combined sampler function?

combined_sampler <- function(levelsnr, nrgroups, samplesize, grsparsity,
                            withinsparsity){

  # generate dataset with factor variables
  factors <- factor_sampler(levelsnr, nrgroups = nrgroups/2,
                           samplesize, grsparsity, withinsparsity)

  # generate dataset with correlated variables
  correlated <- correlated_sampler(levelsnr, nrgroups = nrgroups/2,
                                   samplesize, grsparsity, withinsparsity)

  # combine datasets
  covariates <- cbind(factors$x, correlated$x)
  coefs_vec <- append(factors$coefs_vec, correlated$coefs_vec)

  # simulate Y
  Y <- covariates %*% coefs_vec + 0.1*rnorm(samplesize)

  # demean Y
  # Y <- Y - mean(Y)

  # relevant groups
  relevant_groups <- append(factors$groups, correlated$groups)

  # group structure
  grp_struct <- append(factors$group_structure, rep((nrgroups/2+1):nrgroups, each = levelsnr))

  # create output list
  data <- list(x = covariates, y = as.vector(Y), coefs_vec = coefs_vec,
               nonzerogroups = sum(relevant_groups),
               nonzerocoeffs = sum(length(coefs_vec[coefs_vec != 0])),
               groups = relevant_groups, group_structure = grp_struct,
               observed_factors = NA, corrplot = NA)
  return(data)
}

