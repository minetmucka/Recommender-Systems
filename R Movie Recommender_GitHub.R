
#############################################################################################################################################################
## Objective:  Build a Movie Recommender System that recommends movies to a particular user based on other movies the user rated                            #
## Data source: Dwell_Time_VersionA.csv and Dwell_Time_VersionB.csv                                                                                         #
## Please install "recommenderlab" package: install.packages("recommenderlab")                                                                              #
## Please install "reshape2" package: install.packages("reshape2")                                                                                          #
#############################################################################################################################################################


# Load the libraries into R
library(recommenderlab)
library(reshape2)

# Load the dataset
data("MovieLense")
MovieLense

# visualize a small part of the dataset 
# to get a better understanding of the data

ml10 <- MovieLense[c(1:10),]
ml10 <- ml10[,c(1:10)]
as(ml10, "matrix")

# Visualize the MovieLense data matrix of the 
# first 100 rows and 100 columns in form of a heatmap
image(MovieLense[1:100,1:100])

# Build the model
train <- MovieLense
our_model <- Recommender(train, method = "UBCF")
our_model #storing our model in our_model variable

# Predict user 115
User = 115
pre <- predict(our_model, MovieLense[User], n = 10)
pre

# List the movies the user has already rated and 
# display the score he/she gave
user_ratings <- train[User]
as(user_ratings, "list")

# Recommendation for user 115
as(pre,"list")
