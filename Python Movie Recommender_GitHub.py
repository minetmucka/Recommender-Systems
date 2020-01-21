#################################################################################
#Objective: Build a Movie Recommender System that recommends movies             #
# to a particular user based on other movies the user rated                     #
#Data Source: fetch_movielens                                                   #
#Python Version: 3.4+                                                           #
#Using Packages: numpy,  lightfm                                                #
#################################################################################

# Import libraries 
import numpy as np
from lightfm import LightFM
from lightfm.datasets import fetch_movielens

# Fetch data using fetch_movielens method
movielens_data = fetch_movielens(min_rating = 3.0)

# Let's see what the movielens_data variable contains
for key, value in movielens_data.items():
    print(key, type(value), value.shape)
	
train = movielens_data['train']
labels = movielens_data['item_labels']

# Build the model 
model = LightFM(learning_schedule ='adagrad', learning_rate = 0.1, loss = 'warp')

# Train the model
model.fit(train, epochs = 30, num_threads = 2)

# Number of users and movies in training data
n_users, n_items = movielens_data['train'].shape
print("n_users: " + str(n_users))
print("n_items: " + str(n_items))

# Randomly selecting a user (i.e. user_id) 
user_id = 118

# List of movies user 118 has already liked
already_liked = labels[train.tocsr()[user_id].indices]

print("\nMovies already liked by User %s:" % user_id)
for x in already_liked:
  print("    %s" % x)
  
# Movies our model predicts user will like
item_ids = np.arange(n_items)

# Print the item_ids
print(item_ids)

# The numpy 'arange' method will return evenly spaced 
# values from 0 to number of items (n_items) depicting item IDs
scores = model.predict(user_id, item_ids)

# Print the scores
print(scores)

# Rank them in order of most liked to least
sorted_ids = np.argsort(-scores) # Using negative sign to sort in order of most liked first

top_items = movielens_data['item_labels'][sorted_ids]

# Print the top_items
print(top_items)

# Recommended movies to the particular user id
print("Top 5 movies to be recommended to user " + str(user_id) + " are:\n")

for movie in range(5):
    print(str(movie+1) + ". " + top_items[movie])
	
