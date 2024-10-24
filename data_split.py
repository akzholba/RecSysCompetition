import pandas as pd

def split_data_by_user(df, test_size=3, val_size=3):
    """
    Функция разделения данных на обучающую, валидационную и тестовую выборки.
    
    :param df: DataFrame с данными
    :param test_size: размер тестового набора (по умолчанию 1)
    :param val_size: размер валидационного набора (по умолчанию 1)
    :return: три DataFrame: train_df, val_df, test_df
    """
    # Группируем данные по каждому пользователю
    grouped = df.groupby('user_id')
    
    # Список для хранения индексов каждой группы
    train_indices = []
    val_indices = []
    test_indices = []
    
    # Проходимся по каждой группе
    for _, group in grouped:
        # Сортируем группу по timestamp
        sorted_group = group.sort_values(by='timestamp', ascending=False)
        
        # Получаем индексы для каждой выборки
        test_idx = sorted_group.index[:test_size]
        val_idx = sorted_group.index[test_size:test_size + val_size]
        train_idx = sorted_group.index[test_size + val_size:]
        
        # Добавляем индексы в соответствующие списки
        train_indices.extend(train_idx)
        val_indices.extend(val_idx)
        test_indices.extend(test_idx)
    
    # Формируем DataFrames для каждой выборки
    df = df.drop(['timestamp'], axis = 1)
    train_df = df.loc[train_indices].copy()
    val_df = df.loc[val_indices].copy()
    test_df = df.loc[test_indices].copy()
    
    return train_df, val_df, test_df

