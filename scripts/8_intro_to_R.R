# Intro to R (for predictive models)
# Emilio Lehoucq

# Loading packages --------------------------------------------------------

library(tidyverse)

# Difference  between R and RStudio ---------------------------------------

# Comments and code -------------------------------------------------------

# Comment
x <- NA # code

# Assignment operator -----------------------------------------------------

x <- 5
x = 4 # possible, but not ideal

# How to read errors ------------------------------------------------------

x 4 # Use them in your favor and Google, particularly for StackExchange

# Operators ---------------------------------------------------------------

# Logical operators
x == 4
x < 4
x > 4
x <= 4
x >= 4
x != 4
y <- 5

# Arithmetic operators
z <- x + y
z
z <- x - y
z
z <- x * y
z
z <- x / y
z
z <- x ^ y
z

# Data types --------------------------------------------------------------

# Variables and variable types

class(x)
y <- "five"
class(y)
z <- TRUE
class(z)

gender <- c(rep("male", 2), rep("female", 2))
?rep # Pulling the documentation/"asking for help"
gender <- factor(gender)
gender
class(gender)

# Vectors

v <- c(x, y, z)
v
v[1]
v[c(1, 3)]
length(v)

# Data frames

v2 <- c(1, 2, 3)
v3 <- c("a", "b", "c")
df <- data.frame(v, v2, v3)
df
View(df)
names(df) <- c("vector 1", "vector 2", "vector 3")
df
class(df)
df[1]
df[ , 1]
df[3]
df[ , 3]
df[c(1, 3)]
df[c("vector 1", "vector 3")]
df[1, ]
df[3, ]
df[1, 1]
df$`vector 1`

# Tibbles

t <- tibble(v, v2, v3)
df
t # prints differently
class(t)
t[["v"]] # same with lists
t[[1]] # same with lists
class(as.data.frame(t)) # to interact with older packages

ls()

# For loops ---------------------------------------------------------------

for (i in 1:5) {
  
  print(i)
  
}

vector_to_iterate <- c()
vector_to_iterate

for (i in 1:5) {
  
  vector_to_iterate[i] <- i
  
}

vector_to_iterate

# Functions ---------------------------------------------------------------

say_text <- function(text){
  
  print(text)
  
}

say_text("hey")
say_text(hey)

say_text <- function(text){ # takes a string as input
  
  print(text) # returns string
  
}

#' @name say_text
#' @param text string
#' @returns string
say_text <- function(text){
  
  print(text)
  
}

# Tidyverse ---------------------------------------------------------------

# dplyr
# https://dplyr.tidyverse.org/

mtcars

mtcars %>% # pipe
  filter(mpg == 21.0)

mtcars %>% 
  select(mpg)

mtcars %>% 
  mutate(mpg_by_cyl = mpg / cyl)

# ggplot2
# https://github.com/rstudio/cheatsheets/blob/master/data-visualization-2.1.pdf

ggplot(mtcars)

ggplot(mtcars) +
  aes(hp, mpg)

ggplot(mtcars) +
  aes(hp, mpg) +
  geom_point()

ggplot(mtcars) +
  aes(hp, mpg) +
  geom_point() +
  theme_bw() 

ggplot(mtcars) +
  aes(hp, mpg) +
  geom_point() +
  theme_bw() +
  theme(panel.border = element_blank()) 

ggplot(mtcars) +
  aes(hp, mpg) +
  geom_point() +
  theme_bw() +
  theme(panel.border = element_blank()) +
  ggtitle("Mpg by hp")

ggplot(mtcars) +
  aes(hp, mpg) +
  geom_point() +
  theme_bw() +
  theme(panel.border = element_blank(),
        plot.title = element_text(hjust = 0.5)) +
  ggtitle("Mpg by hp")

# Reading and saving data -------------------------------------------------
# https://r4ds.had.co.nz/data-import.html

saveRDS(mtcars, file = "mtcars.rds")

readRDS(file = "mtcars.rds")

data <- readRDS(file = "mtcars.rds")

# Exercises ---------------------------------------------------------------
# From R4DS

# 4.4, exercise 2
# 10.5, exercise 2
# 19.2.1, exercises 3 and 4
# 21.2.1, exercises 1.1, 1.2, and 1.3
# 21.3.5, exercise 3
