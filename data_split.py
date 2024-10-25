import pandas as pd


def split_data_by_user(df, test_size=1):

    """
    Функция разделения данных на обучающую, валидационную и тестовую выборки.
    
    :param df: DataFrame с данными
    :param test_size: размер тестового набора (по умолчанию 1)

    :return: два DataFrame: train_df, test_df

    """
    # Группируем данные по каждому пользователю
    grouped = df.groupby('user_id')
    
    # Список для хранения индексов каждой группы
    train_indices = []
    test_indices = []
    
    # Проходимся по каждой группе
    for _, group in grouped:
        # Сортируем группу по timestamp
        sorted_group = group.sort_values(by='timestamp', ascending=False)
        
        # Получаем индексы для каждой выборки
        test_idx = sorted_group.index[:test_size]
        train_idx = sorted_group.index[test_size:]
        
        # Добавляем индексы в соответствующие списки
        train_indices.extend(train_idx)
        test_indices.extend(test_idx)
    
    # Формируем DataFrames для каждой выборки
    train_df = df.loc[train_indices].copy()
    test_df = df.loc[test_indices].copy()
    
    return train_df, test_df

