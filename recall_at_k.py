import pandas as pd

def recall_at_k_overall(df, actual_col, predicted_col, k=10):
    """
    Вычисляет общий Recall@K для всех пользователей в DataFrame.

    Параметры:
    - df (pd.DataFrame): DataFrame, содержащий фактически релевантные и предсказанные элементы для каждого пользователя.
    - actual_col (str): Название колонки с фактически релевантными элементами (список).
    - predicted_col (str): Название колонки с предсказанными элементами (список).
    - k (int): Количество топ рекомендаций, по умолчанию 10.

    Возвращает:
    - float: Общее значение метрики Recall@K для всех пользователей.
    """
    # Переменные для подсчета общего числа релевантных элементов и тех, что попали в топ-K рекомендаций
    total_relevant_items = 0
    relevant_items_found = 0
    
    # Проходим по каждой строке DataFrame
    for _, row in df.iterrows():
        actual_items = set(row[actual_col])
        predicted_items = set(row[predicted_col][:k])  # Предсказанные предметы ограничиваем топ-K
        
        # Общее количество релевантных предметов
        total_relevant_items += len(actual_items)
        
        # Количество найденных релевантных предметовd
        relevant_items_found += len(actual_items.intersection(predicted_items))
    
    # Если нет релевантных предметов, возвращаем 0, чтобы избежать деления на 0
    if total_relevant_items == 0:
        print('Произошла неприятная ситуевина')
        return 0.0
    
    # Вычисляем общий Recall@K
    recall = relevant_items_found / total_relevant_items
    return recall