import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

def set_seed(seed):
    """
    Sets the seed for generating random numbers for Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed to set for random number generation.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def eval_preds(labels, preds, metrics=['acc', 'f1'], res_file_path=None):
    """
    Evaluate predictions given the labels and specified metrics.
    
    Parameters:
    - labels (list or np.array): True labels.
    - preds (list or np.array): Predicted labels.
    - metrics (list of str): List of metrics to compute. Default is ['acc', 'f1'].
    - res_file_path (str or None): Path to save the results. Default is None.
    
    Returns:
    - dict: Computed metrics.
    """
    computed_metrics = {}
    
    # Ensure labels and preds are numpy arrays
    labels = np.array(labels)
    preds = np.array(preds)
    
    # compute each metric
    if 'acc' in metrics:
        computed_metrics['acc'] = accuracy_score(labels, preds)
    if 'f1' in metrics:
        computed_metrics['f1'] = f1_score(labels, preds, average='macro')
        
    # save results to file
    if res_file_path is not None:
        with open(res_file_path, 'w') as f:
            f.write(f'num samples: {len(labels)}\n')
            for metric, value in computed_metrics.items():
                f.write(f'{metric}: {value}\n')
    
    return computed_metrics
