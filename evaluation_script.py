from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import tensorflow as tf
import numpy as np
import time
import random
import wandb
from memorization_metric import memorization_metric
import argparse
from multiprocessing import Process,Pipe,set_start_method

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
    def __init__(self,device):
        self.device = device
        self.model = None
        self.stop_scoring = False
        self.reciever, self.sender = Pipe()
        super().__init__()
    def score(self,token_size):
        '''Evaluates the memorization metric of data from the Pipe()

        > Data input consists is a batched tensor of shape (batch_size,token_size)
        > uitlizes first token_size//2 tokens to generate next token_size//2 tokens and evaluates them
        '''
        while(True):
            inp_tensor = self.reciever.recv().to(f'cuda:{self.device}')
            inp = inp_tensor[:,:token_size//2]
            ground_truth = inp_tensor[:,token_size//2:token_size]

            #Generation takes a lot longer for no apparant reason
            res = self.model.generate(inp,do_sample=False, temperature=0.9,use_cache=False, min_length=2048,max_length=2048)[:,token_size//2:token_size] 

            self.reciever.send(memorization_metric(ground_truth,res).cpu().numpy())
    def run(self):
        start_time = time.time()
        if(not self.model):
            self.model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B").half().eval().to(f'cuda:{self.device}')
        print(f'Model created in: {time.time() - start_time:06}s')
        self.score(2048)


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
    step = 1
    # for path in get_dataset_paths(args):
    #     ds = tf.data.TFRecordDataset(path).map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).batch(DEVICES)
    ds = tf.data.Dataset.from_tensor_slices(tf.random.uniform(shape=(256,2048),maxval=int(1e4),dtype=tf.dtypes.int32)).batch(BATCH_SIZE).batch(torch.cuda.device_count())
    start_time = time.time()
    for batch in iter(ds):
        batch = batch.numpy()
        res = [0]*DEVICES
        for i in range(DEVICES):
            models[i].sender.send(torch.tensor(batch[i],dtype=torch.int32))
        for i in range(DEVICES):
            res[i] = models[i].sender.recv()
        res = np.asarray(res).flatten() #Takes about 1000s seconds per batch of 64
        print(f'Time taken per batch: {time.time() - start_time:06}s')
        
        if(args.wandb_project_name):
            wandb.log({'memorization_metric':np.average(x)},step)
