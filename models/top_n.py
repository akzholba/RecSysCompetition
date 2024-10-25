from tqdm import tqdm
import numpy as np
import pandas as pd

class TopN:

    def __init__(self, create_user_item_matrix, user_features_df, item_features_df):
        self.user_features_df = user_features_df
        self.item_features_df = item_features_df
        self.create_user_item_matrix = create_user_item_matrix

    def predict(self, X):
        predictions = []
        num_users = X.user_id.nunique()
        num_items = X.item_id.nunique()

        for user_id in tqdm(range(num_users), desc = 'top_user_score_loading...'):
            for item_id in range(num_items):  
                # в качестве оценки оценка другими пользователями
                predicted_score = self.normalized_popularity[item_id] 
                predictions.append(predicted_score)
        return predictions
    
    def fit(self, X, y):
        user_item_matrix = self.create_user_item_matrix(
            pd.concat([X, y]), self.user_features_df, self.item_features_df)
        # смотрим на популярность фильма
        popularity = np.sum(user_item_matrix > 0, axis=0)

        # нормализация
        min_popularity = np.min(popularity)
        max_popularity = np.max(popularity)
        self.normalized_popularity = 5 * (popularity - min_popularity) / (max_popularity - min_popularity)
        return self
        