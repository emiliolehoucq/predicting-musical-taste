# Results
# Emilio Lehoucq

# Setting parameters ------------------------------------------------------

options(java.parameters = "-Xmx54g")

# Loading packages --------------------------------------------------------

library(tidyverse)
library(glmnet) # masks 'expand', 'pack', and 'unpack' from tidyr
library(glmnetUtils)
library(randomForest) # masks 'combine' from 'dplyr' and 'margin' from ggplot2
library(caret) # masks 'lift' from purrr
library(gbm)
library(bartMachine)
library(broom)
library(vip)
library(pdp)
library(ALEPlot)

# Reading data ------------------------------------------------------------

data_train <- readRDS("processed_data/core_2_train.rds")
data_test <- readRDS("processed_data/core_2_test.rds")

# Setting parameters ------------------------------------------------------

set.seed(607) 

fit_control <- trainControl(method = "none", classProbs = TRUE)

# Classical ---------------------------------------------------------------

classical_y_test <- data_test$like_listen_classical

classical_x_test <- data_test

classical_x_test$like_listen_classical <- NULL

classical_bart <- bartMachine(data.frame(classical_x_test), classical_y_test, k = 3, num_trees = 50)

classical_bart_vip <- investigate_var_importance(classical_bart, num_replicates_for_avg = 20)

classical_bart_vip_tibble <- tibble(avg = classical_bart_vip$avg_var_props,
                           var = labels(classical_bart_vip$avg_var_props)) %>% 
  mutate(var = fct_reorder(var, avg)) %>% 
  slice(1:25)

ggplot(classical_bart_vip_tibble) +
  aes(x = avg, y =var) +
  geom_col() +
  theme_bw() +
  theme(panel.border = element_blank(),
        plot.title = element_text(hjust = 1)) +
  ggtitle("Variable importance for the predictors of classical (BART)") +
  xlab("Inclusion proportion") +
  ylab("Predictor") + 
  labs(caption="First 15 predictors displayed.") # 678 x 410

classical_lasso <- glmnet(like_listen_classical ~ ., data = data_test, alpha = 1, lambda = 0.003813685, family = "binomial")

classical_lasso_tidy <- tidy(classical_lasso, return_zeros = FALSE)

write.csv(classical_lasso_tidy, "classical_lasso_tidy.csv")

classical_bt <- train(like_listen_classical ~ ., data = data_test %>% mutate(like_listen_classical = factor(like_listen_classical, 
                                                                                                             labels = make.names(levels(like_listen_classical)))), 
                      method = "gbm", 
                      trControl = fit_control, 
                      verbose = FALSE, 
                      tuneGrid = data.frame(interaction.depth = 1,
                                            n.trees = 800,
                                            shrinkage = 0.089,
                                            n.minobsinnode = 10),
                      metric = "Accuracy")

vip(classical_bt) +
  theme_bw() +
  theme(panel.border = element_blank(),
        plot.title = element_text(hjust = 1)) +
  ggtitle("Variable importance for the predictors of classical (boosted trees)") +
  xlab("Predictor") +
  ylab("Variable importance") # 678 x 410

# Rock and oldies ---------------------------------------------------------

rock_rf <- randomForest(like_listen_classic_rock_oldies ~ ., data = data_test, nodesize = 10, mtry = 28, importance = TRUE)

vip(rock_rf) +
  theme_bw() +
  theme(panel.border = element_blank(),
        plot.title = element_text(hjust = 1)) +
  ggtitle("Variable importance for the predictors of rock and oldies") +
  xlab("Predictor") +
  ylab("Variable importance") # 678 x 410

# Country -----------------------------------------------------------------

country_rf <- randomForest(like_listen_country ~ ., data = data_test, nodesize = 10, mtry = 30 , importance = TRUE)

vip(country_rf) +
  theme_bw() +
  theme(panel.border = element_blank(),
        plot.title = element_text(hjust = 1)) +
  ggtitle("Variable importance for the predictors of country") +
  xlab("Predictor") +
  ylab("Variable importance") # 678 x 410
