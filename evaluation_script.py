from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import tensorflow as tf
import numpy as np
import time
import random
import wandb
from memorization_metric import memorization_metric
import argparse
from torch.multiprocessing import Process,set_start_method,Pipe

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluates the memorization of input tfrecords based on the memorization metric')
    parser.add_argument('--tfrecord-index',help='index file containing a list of tfrecord file paths')
    parser.add_argument('--wandb-project-name',help='wandb project name for the current run')
    return parser.parse_args()


def get_dataset_paths(args):
    if(args.tfrecord_index):
        with open(args.tfrecord_index) as file:
            for path in file.read().splitlines():
                yield path
    else:
        raise ValueError("No tfrecord index file provided. Pleas provide a file to continue")

class ScoreModel(Process):
    def __init__(self,device):
        self.device = device
        self.model = None
        self.stop_scoring = False
        self.internal,self.external = Pipe()
        super().__init__()
    def score(self):
        while(not self.stop_scoring):
            inp_tensor = self.internal.recv().to(f'cuda:{self.device}')
            inp = inp_tensor[:,:1024]
            ground_truth = inp_tensor[:,1024:]
            res = self.model.generate(inp,do_sample=False, temperature=0.9, min_length=2048,max_length=2048)[:,1024:2048]
            self.internal.send(memorization_metric(ground_truth,res))
        self.internal.close()
    def run(self):
        start_time = time.time()
        if(not self.model):
            self.model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B").eval().to(f'cuda:{self.device}')
        print(f'Model created in: {time.time() - start_time:06}s')
        self.score()

def score(inp_tensor,id,results):
    '''
    Calculates the memorization metric for the given input tensor
    '''
    inp_tensor = torch.tensor(inp_tensor,dtype=torch.int32)
    inp = inp_tensor[:,:1024]
    ground_truth = inp_tensor[:,1024:]
    res = models[id].generate(inp.to(f'cuda:{id}'),do_sample=True, temperature=0.9, min_length=2048,max_length=2048)[:,1024:2048].cpu()
    
    results[id] = memorization_metric(ground_truth,res).numpy()

def batched_evaluate(inp_batched_tensor,pipes):
    '''
    Distributes the batched tensor (With a batch size of number of gpus) to each gpu to be scored
    '''
    for i in range(8):
        one,two = pipes[i]
        one.send(inp_batched_tensor[i])
        one.close()

    return np.asarray(results).flatten()


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
    BATCH_SIZE = 2
    DEVICES = torch.cuda.device_count()
    args = parse_args()
    set_start_method('forkserver')
    if(args.wandb_project_name):
        wandb.init(project=args.wandb_project_name)
    
    #Loading model on eight cuda devices
    models = [ScoreModel(i) for i in range(DEVICES)]
    [model.start() for model in models]
    pipes = [model.external for model in models]
    # step = 1
    # for path in get_dataset_paths(args):
    #     ds = tf.data.TFRecordDataset(path).map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE)
    ds = tf.data.Dataset.from_tensor_slices(tf.random.uniform(shape=(128,2048),maxval=int(1e4),dtype=tf.dtypes.int32)).batch(BATCH_SIZE).batch(torch.cuda.device_count())
    start_time = time.time()
    for x in iter(ds):
        x = x.numpy()
        res = [0]*DEVICES
        for i in range(DEVICES):
            pipes[i].send(torch.tensor(x[i],dtype=torch.int32))
        for i in range(DEVICES):
            res[i] = pipes[i].recv().cpu().numpy()
        print(res)
        print(f'Time taken per batch: {time.time() - start_time:06}s')
        
        if(args.wandb_project_name):
            wandb.log({'memorization_metric':x},step)
