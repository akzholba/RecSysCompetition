import pandas as pd

def get_user_features_from_train(train_df, item_features):
    """
    Извлекает признаки пользователей из тренировочного датасета.
    Возвращает датафрейм с агрегированными признаками пользователей.
    """
    # Средний рейтинг пользователя
    user_avg_rating = train_df.groupby('user_id')['rating'].mean().reset_index()
    user_avg_rating.columns = ['user_id', 'user_avg_rating']
    
    # Количество оцененных фильмов
    user_num_ratings = train_df.groupby('user_id')['rating'].count().reset_index()
    user_num_ratings.columns = ['user_id', 'user_num_ratings']

    # Присоединяем все признаки
    user_features = pd.merge(user_avg_rating, user_num_ratings, on='user_id')

    # Средний рейтинг по жанрам
    merged_data = pd.merge(train_df, item_features, on='item_id', how='left')
    genre_columns = [col for col in item_features.columns if 'genre' in col]
    user_genre_ratings = merged_data.groupby('user_id')[genre_columns].mean().reset_index()
    user_genre_ratings.columns = ['user_id'] + [f'user_avg_rating_{genre}' for genre in genre_columns]
    user_features = pd.merge(user_features, user_genre_ratings, on='user_id', how='left')

    return user_features

def get_item_features_from_train(train_df, item_features):
    """
    Извлекает признаки фильмов из тренировочного датасета.
    Возвращает датафрейм с агрегированными признаками фильмов.
    """
    # Средний рейтинг фильма
    movie_avg_rating = train_df.groupby('item_id')['rating'].mean().reset_index()
    movie_avg_rating.columns = ['item_id', 'movie_avg_rating']
    
    # Количество оценок у фильма
    movie_num_ratings = train_df.groupby('item_id')['rating'].count().reset_index()
    movie_num_ratings.columns = ['item_id', 'movie_num_ratings']
    
    # Количество жанров у фильма
    item_features['num_genres'] = item_features.iloc[:, 1:].sum(axis=1)
    
    # Присоединяем все признаки
    item_features_full = pd.merge(item_features, movie_avg_rating, on='item_id', how='left')
    item_features_full = pd.merge(item_features_full, movie_num_ratings, on='item_id', how='left')
    
    return item_features_full[['item_id', 'num_genres', 'movie_avg_rating', 'movie_num_ratings']]

def join_features(test_df, user_features, item_features):
    """
    Присоединяет признаки пользователей и фильмов к тестовому набору данных.
    """
    # Присоединение признаков пользователей
    test_df = pd.merge(test_df, user_features, on='user_id', how='left')
    
    # Присоединение признаков фильмов
    test_df = pd.merge(test_df, item_features, on='item_id', how='left')
    
    return test_df

# Пример использования функции 
# from feature_additions import *


# user_features_from_train = get_user_features_from_train(train_df, item_features_df)
# item_features_from_train = get_item_features_from_train(train_df, item_features_df)

# # Присоединение признаков к тестовому набору данных
# test_df_with_features = join_features(test_df, user_features_from_train, item_features_from_train)

# # Итоговый тестовый датасет с признаками
# test_df_with_features.head()