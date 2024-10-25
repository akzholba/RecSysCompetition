import pandas as pd
import numpy as np

def negative_sampling(df, max_timestamp=None):
    """
    Функция для проведения негативного сэмплирования.
    
    :param df: Датафрейм с данными о взаимодействиях пользователей и объектов.
    :param max_timestamp: Максимальное значение timestamp в датасете. Если None, вычисляется автоматически.
    :return: Датафрейм с негативным сэмплингом.
    """
    
    # Если max_timestamp не задано, берем максимальное значение из датафрейма
    if max_timestamp is None:
        max_timestamp = df['timestamp'].max()
    
    # Список уникальных пользователей
    users = df['user_id'].unique()
    
    # Список уникальных фильмов
    items = df['item_id'].unique()
    
    # Количество просмотров каждым пользователем
    viewed_counts = df.groupby('user_id')['item_id'].count().to_dict()
    
    # Создаем пустой датафрейм для хранения результатов
    sampled_df = pd.DataFrame(columns=['user_id', 'item_id', 'rating', 'timestamp'])
    
    # Проходим по каждому пользователю
    for user in users:
        
        # Находим фильмы, которые пользователь просмотрел
        viewed_items = df.query("user_id == @user")['item_id'].values
        
        # Находим фильмы, которые пользователь не смотрел
        unviewed_items = np.setdiff1d(items, viewed_items)
        
        # Отбираем столько же негативных примеров, сколько фильмов просмотрел пользователь
        num_viewed = viewed_counts.get(user, 0)
        negative_samples = np.random.choice(unviewed_items, size=num_viewed)
        
        # Формируем пары (пользователь, объект, метка)
        new_rows = []
        for item in negative_samples:
            new_rows.append({
                'user_id': user,
                'item_id': item,
                'rating': 0,
                'timestamp': max_timestamp
            })
        
        # Добавляем новые строки в датафрейм
        sampled_df = pd.concat([sampled_df, pd.DataFrame(new_rows)], ignore_index=True)
    
    return sampled_df