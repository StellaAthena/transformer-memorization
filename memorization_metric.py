import numpy as np
import torch
from torch import nn
def memorization_metric(predictions,ground_truth):
    '''
    ground_truth: tuple of pytorch tensors of shape (batch_size,token_probabilites) of length num_tokens, the shape of scores of generate method
    predictions: array of shape (batch_size,num_tokens)

    Out: torch array of shape (batch_size)
    '''

    loss = nn.NLLLoss(reduction='none')
    token_wise_losses = []
    token_size = ground_truth.shape[1]
    batch_size = ground_truth.shape[0]
    for i in range(token_size):
        curr_loss = loss(predictions[i].cpu(),
                        ground_truth[:,i].type(torch.LongTensor))
        token_wise_losses.append(curr_loss.numpy())

    
    total_loss = np.asarray(token_wise_losses).transpose()
    total_loss = np.average(total_loss,axis=-1)
    return total_loss

if __name__ == '__main__':
    predictions = tuple([torch.rand(size=(64,500)) for i in range(10)]) #vocab_size=500,batch_size=64,out_num_tokens=10
    ground_truth = torch.ones(size=(64,10),dtype=torch.int32) #out_batch_size,out_num_tokens
    print(memorization_metric(predictions,ground_truth).shape) #Output: (64,)