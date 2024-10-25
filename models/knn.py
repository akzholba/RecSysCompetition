from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

from sklearn.neighbors import NearestNeighbors

from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

class KNN:

    def __init__(
        self, create_user_item_matrix, user_features_df, 
        item_features_df, metric='cosine', n_neighbors=10
    ):
        self.user_features_df = user_features_df
        self.item_features_df = item_features_df
        self.create_user_item_matrix = create_user_item_matrix
        self.metric = metric
        self.n_neighbors = n_neighbors

    def predict(self, X):
        num_users, num_items = user_item_matrix.shape
        predictions = []

        for user_id in tqdm(range(num_users),desc = 'user_knn_scores...'):
            user_data = user_item_matrix[user_id].reshape(1, -1)
            distances, indices = knn_model.kneighbors(user_data, n_neighbors=n_neighbors, return_distance=True)

            for item_id in range(num_items):
                # нет оценки пользователя
                if user_item_matrix[user_id, item_id] == 0:  
                    neighbor_scores = []
                    for neighbor in indices.flatten():
                        if user_item_matrix[neighbor, item_id] > 0:
                            neighbor_scores.append(user_item_matrix[neighbor, item_id])

                    if neighbor_scores:
                        predicted_score = np.mean(neighbor_scores)
                        predictions.append(predicted_score)
        return predictions
    
    def fit(self, X, y):
        self.user_item_matrix = self.create_user_item_matrix(
            pd.concat([X, y]), self.user_features_df, self.item_features_df)
        
        knn_model = NearestNeighbors(n_neighbors=self.n_neighbors, metric=self.metric, algorithm='brute')
        knn_model.fit(self.user_item_matrix)
    
        return knn_model
            
    def predict_for_user(self, knn_model, user_id):
        user_data = self.user_item_matrix[user_id].reshape(1, -1)
        distances, indices = knn_model.kneighbors(user_data, return_distance=True)
        
        return indices, distances