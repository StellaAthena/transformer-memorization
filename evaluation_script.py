from transformers import AutoConfig, AutoModelForCausalLM
import torch
import tensorflow as tf
import numpy as np
import time
import random
import wandb
from memorization_metric import memorization_metric
import argparse
from tqdm import tqdm
from multiprocessing import Process,Pipe,set_start_method
from result_records import TFrecordCreator

try:
    from collections.abc import MutableMapping
except ImportError:
    from collections import MutableMapping

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


class ScoreModel(Process):
    """Parallelizes evaluation of the records

    Creates a saperate spawned process for each model. 
    Data is sent and recieved through multiprocessing.Pipe()
    This runs indefinitely once started
    """
    def __init__(self,device,token_size=256):
        self.token_size = token_size
        self.device = device
        self.model = None
        self.reciever, self.sender = Pipe() #Tensors are sent and recieved through Pipe()
        super().__init__()
    
    def get_model(self):
        self.model = AutoModelForCausalLM.from_pretrained('EleutherAI/gpt-j-6B').to(f'cuda:{self.device}')
    def score(self,token_size):
        '''Evaluates the memorization metric of data from the Pipe()

        > Data input consists is a batched tensor of shape (batch_size,token_size)
        > uitlizes first token_size//2 tokens to generate next token_size//2 tokens and evaluates them
        '''
        while(True):
            inp_tensor = self.reciever.recv()
            if(inp_tensor is None):
                break
            inp = inp_tensor[:,:token_size//2].to(f'cuda:{self.device}')
            ground_truth = inp_tensor[:,token_size//2:token_size].to(f'cuda:{self.device}')

            res = self.model.generate(inp,do_sample=False, temperature=0.9,use_cache=False, min_length=token_size,max_length=token_size)[:,token_size//2:token_size] 

            self.reciever.send(memorization_metric(ground_truth,res).cpu().numpy())
    def run(self):
        start_time = time.time()
        if(not self.model):
            self.get_model()
        print(f'Model created in: {time.time() - start_time:06}s')
        self.score(256)


def parse_fn(example_proto):
    '''
    Converts a tfrecord proto to a tensor of shape (2048,)
    '''
    features = {
        "text": tf.io.VarLenFeature(tf.int64)
    }
    parsed_features = tf.io.parse_single_example(example_proto, features)

    return tf.cast(tf.sparse.to_dense(tf.sparse.reorder(parsed_features["text"])), tf.uint32)


if __name__ == '__main__':
    BATCH_SIZE = 8
    DEVICES = torch.cuda.device_count()
    set_start_method('spawn')
    args = parse_args()
    if(args.wandb_project_name):
        wandb.init(project=args.wandb_project_name)
    
    #Loading model on eight cuda devices
    models = [ScoreModel(i) for i in range(DEVICES)]
    [model.start() for model in models]


    records = TFrecordCreator(RESULTS_PATH)
    step = 1
    for path in get_dataset_paths(args):
        ds = tf.data.TFRecordDataset(path).map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).batch(DEVICES)
        for batch in tqdm(iter(ds)):
            batch = batch.numpy()
            
            res = [0]*DEVICES
            [model.sender.send(torch.tensor(batch[i],dtype=torch.int32)) for model in models]
            for i in range(DEVICES):
                res[i] = models[i].sender.recv()
            
            res = np.asarray(res).flatten() 
            for i in res:
                records.write(i)
            
            if(args.wandb_project_name):
                wandb.log({'memorization_metric':np.average(x)},step)
    
    #Deleting models
    [model.sender.send(None)]
    [model.join() for model in models] 
