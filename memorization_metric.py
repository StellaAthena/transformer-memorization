import numpy as np
import torch
def memorization_metric(ground_truth,predictions):
    '''
    ground_truth: pytorch tensor of shape (batch_size,token_length)
    predictions: pytorch tensor of shape (batch_size,token_length)

    Out: torch array of shape (batch_size)
    '''
    return torch.sum(ground_truth == predictions,axis=-1)

if __name__ == '__main__':
    ground_truth = torch.ones(size=(64,1024))
    predictions = torch.ones(size=(64,1024)) #batch size may be dynamic
    print(memorization_metric(ground_truth,predictions))