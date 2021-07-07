# Logistic regression 2
# Emilio Lehoucq

# Loading packages --------------------------------------------------------

library(tidyverse)
library(broom)

# Reading data ------------------------------------------------------------

train <- readRDS("data_exercises/arrests_train.rds")
test <- readRDS("data_exercises/arrests_test_nr.rds")

# What to upload to Kaggle ------------------------------------------------

logit <- glm(released ~ 1, data = train, family = "binomial")
y_hat_test_prob <- predict(logit, newdata = test, type = "response")

mean(y_hat_test_prob)
table(train$released)/nrow(train)

y_hat_test_class <- ifelse(y_hat_test_prob > 0.5, "Yes", "No")
id <- rownames(test)

submission <- data.frame(id = id, released = as.factor(y_hat_test_class)) # turn response to factor
write.csv(submission, "sample_submission.csv", row.names = FALSE) # don't include row names

# Pipeline ----------------------------------------------------------------

# Data setup --------------------------------------------------------------

data <- tibble(train = list(train), test = list(train))

data # what our tibble looks like

data %>% unnest(train) # training data

data %>% unnest(train) %>% count(released) %>% mutate(prop = n/sum(n)) # response proportions in training data

# Modeling ----------------------------------------------------------------
# Code inspired by the Data Science Manual at Northwestern's Statistics Department (Arend Kuyper)

logits <- data %>% 
  mutate(mod1 = map(train, glm, formula = released ~ 1, family = "binomial"),
         mod2 = map(train, glm, formula = released ~ ., family = "binomial"),
         mod3 = map(train, glm, formula = released ~ .^2, family = "binomial"),
         mod4 = map(train, glm, formula = released ~ colour, family = "binomial"),
         mod5 = map(train, glm, formula = released ~ colour + checks, family = "binomial"),
         mod5 = map(train, glm, formula = released ~ colour + checks + checks, family = "binomial"),
         mod5 = map(train, glm, formula = released ~ colour + checks + checks + sex, family = "binomial"),
         mod6 = map(train, glm, formula = released ~ colour + checks + checks + sex + citizen, family = "binomial"),
         mod7 = map(train, glm, formula = released ~ colour + checks + checks + sex + citizen + employed, family = "binomial")) %>%  # fitting models
  pivot_longer(cols = contains("mod"), names_to = "model_name", values_to =  "model_fit") # tidying our tibble--model name and models are the columns

logits %>% pluck("model_fit", 3) %>% tidy() %>% print(n = Inf)

logits %>% pluck("model_fit", 2) %>% predict(type = "response")

# Helper functions --------------------------------------------------------

# Function to calculate error rate
error_rate_glm <- function(data, model){
  data %>% 
    mutate(pred_prob = predict(model, newdata = data, type = "response"),
           pred_class = ifelse(pred_prob > 0.5, "Yes", "No"),
           error = pred_class != released) %>% 
    pull(error) %>% 
    mean()
}

# Function to form confusion matrix
confusion_mat_glm <- function(data, model){
  data %>% 
    mutate(pred_prob = predict(model, newdata = data, type = "response"),
           pred_class = ifelse(pred_prob > 0.5, "Yes", "No")) %>% 
    count(released, pred_class) %>% 
    mutate(prop = n / sum(n))
}

# Calculating errors ------------------------------------------------------

logits <- logits %>% 
  mutate(train_error = map2_dbl(train, model_fit, error_rate_glm),
         train_confusion = map2(train, model_fit, confusion_mat_glm)) # creating columns with train error and confusion matrix

logits %>% 
  select(model_name, train_error) %>% 
  arrange(train_error) # displaying which model has lowest training error

logits %>% 
  filter(model_name == "mod2") %>% 
  unnest(train_confusion) # displaying confusion matrix of model with lowest training error
