import numpy as np
def recall_at_k(y_true: List[int], y_pred: List[List[np.ndarray]], k: int):
    """
    Calculates recall at k ranking metric.

    Args:
        y_true: Labels. Not used in the calculation of the metric.
        y_predicted: Predictions.
            Each prediction contains ranking score of all ranking candidates for the particular data sample.
            It is supposed that the ranking score for the true candidate goes first in the prediction.

    Returns:
        Recall at k
    """
    num_examples = float(len(y_pred))
    predictions = np.array(y_pred)
    predictions = np.flip(np.argsort(predictions, -1), -1)[:, :k]
    num_correct = 0
    for el in predictions:
        if 0 in el:
            num_correct += 1
    return float(num_correct) / num_examples