from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import tensorflow as tf
import numpy as np
import time
import random
import wandb
from memorization_metric import memorization_metric
import argparse
from threading import Thread
from queue import Queue

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluates the memorization of input tfrecords based on the memorization metric')
    parser.add_argument('--tfrecord-index',help='index file containing a list of tfrecord file paths')
    parser.add_argument('--wandb-project-name',help='wandb project name for the current run')
    return parser.parse_args()


def get_dataset_paths(args):
    '''
    Gets all tfrecord paths from the given index file

    Raises ValueError if it is not provided
    '''
    if(args.tfrecord_index):
        with open(args.tfrecord_index) as file:
            for path in file.read().splitlines():
                yield path
    else:
        raise ValueError("No tfrecord index file provided. Pleas provide a file to continue")


def parse_fn(example_proto):
    '''
    Converts a tfrecord proto to a tensor of shape (2048,)
    '''
    features = {
        "text": tf.io.VarLenFeature(tf.int64)
    }
    parsed_features = tf.io.parse_single_example(example_proto, features)

    return tf.cast(tf.sparse.to_dense(tf.sparse.reorder(parsed_features["text"])), tf.uint32)

def score(model,data,token_size=64):
    '''Calculates the memorization metric for the given input tokens
    '''
    inp_tensor = data
    inp = inp_tensor[:,:token_size//2].cuda()
    ground_truth = inp_tensor[:,token_size//2:token_size].cuda()
    res = model.generate(inp,do_sample=False, temperature=0.9, min_length=token_size,max_length=token_size)[:,token_size//2:token_size]
    return memorization_metric(ground_truth,res).cpu().numpy()
if __name__ == '__main__':
    BATCH_SIZE = 512

    args = parse_args()
    if(args.wandb_project_name):
        wandb.init(project=args.wandb_project_name)
    
    #Loading model on eight cuda devices
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B").half().eval()
    model.parallelize()
    step = 1
    # for path in get_dataset_paths(args):
    #     ds = tf.data.TFRecordDataset(path).map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).batch(DEVICES)
    ds = tf.data.Dataset.from_tensor_slices(tf.random.uniform(shape=(2048,2048),maxval=int(1e4),dtype=tf.dtypes.int32)).batch(BATCH_SIZE)
    start_time = time.time()
    for batch in iter(ds):
        batch = torch.tensor(batch.numpy(),dtype=torch.int32,requires_grad=False)
        res = score(model,batch)
        print(f'Time taken per batch: {time.time() - start_time:06}s') #Achieves about 600s per batch of 64 (excluding model loading time)
        
        if(args.wandb_project_name):
            wandb.log({'memorization_metric':np.average(x)},step)


