from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from tqdm import tqdm

def train_classifier(train_df, target_column, classifier):
    # Разделяем данные на признаки (X) и цели (y)
    X = train_df.drop(target_column, axis=1)
    y = train_df[target_column]
        
    # Обучаем классификатор
    model = classifier.fit(X, y)
    
    return model

def make_predictions(features_df, model):
    predictions = model.predict(features_df)
    return predictions

def train_models(train_df, target_column, classifiers):
    trained_models = {}
    for name, classifier in classifiers.items():
        print(f"\nTraining {name}")
        trained_model = train_classifier(train_df, target_column, classifier)
        trained_models[name] = trained_model

    return trained_models

def create_user_item_matrix(df, df_user_feat, df_item_feat):
    n_users = df_user_feat['user_id'].nunique()
    n_items = df_item_feat['item_id'].nunique()
    user_item_matrix = np.zeros((n_users, n_items))
    for line in df.itertuples():
        user_item_matrix[line[1], line[2]] = line[3]
    return user_item_matrix