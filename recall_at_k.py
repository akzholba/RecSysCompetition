#https://insidelearningmachines.com/precisionk_and_recallk/ код для метрики взяли отсюда
import pandas as pd

def recall_at_k(df: pd.DataFrame, k: int, y_test: str, y_pred: str) -> float:
    """
    Function to compute recall@k for an input boolean dataframe
    
    Inputs:
        df     -> pandas dataframe containing boolean columns y_test & y_pred
        k      -> integer number of items to consider
        y_test -> string name of column containing actual user input
        y-pred -> string name of column containing recommendation output
        
    Output:
        Floating-point number of recall value for k items
    """    
    # extract the k rows
    dfK = df.head(k)
    # compute number of all relevant items
    denominator = df[y_test].sum()
    # compute number of recommended items that are relevant @k
    numerator = dfK[dfK[y_pred] & dfK[y_test]].shape[0]
    # return result
    if denominator > 0:
        return numerator/denominator
    else:
        return None