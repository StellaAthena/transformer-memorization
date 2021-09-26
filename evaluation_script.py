from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import torch
import tensorflow as tf
import numpy as np
import wandb
from memorization_metric import memorization_metric
import argparse
from result_records import TFrecordCreator
from dataset_loader import build_train_dataset
from threading import Thread
import queue
import torch.distributed as dist
import os
import time


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluates the memorization of input tfrecords based on the memorization metric')
    parser.add_argument('--wandb-project-name',help='wandb project name for the current run')
    return parser.parse_args()

class BatchedDataset(Thread):
    def __init__(self,batch_size,take_every,q):
        super().__init__()
        self.batch_size = batch_size
        self.take_every = take_every
        self.q = q
    def run(self):
        ds = build_train_dataset(
                    data_prefix="/mnt/ssd-1/data/pile/pile_text_document",
                    data_impl="mmap",
                    splits_string="949,50,1",
                    train_valid_test_num_samples=[self.batch_size, 0, 0],
                    seq_length=2048,
                    seed=1234, #Default seed
                    skip_warmup=True
                )
        tokens = []
        indicies = []
        val = 1
        idx = 0
        for i in ds:
            idx += 1
            if(idx%self.take_every != 0):
                continue
            tokens.append(i['text'][:TOKEN_SIZE])
            indicies.append(idx)
            if(val%self.batch_size == 0):
                self.q.put((np.asarray(tokens).reshape((self.batch_size,-1)),indicies))
                indicies = []
                tokens = []
            val += 1
        self.q.put((None,None))
        self.q.task_done()



def score(model,data,token_size=64):
    '''Calculates the memorization metric for the given input tokens
    '''
    inp_tensor = data[:,:token_size]
    inp = inp_tensor[:,:token_size//2].cuda()
    res = model.generate(inp,do_sample=False,use_cache=False, temperature=0.9, min_length=token_size,max_length=token_size,return_dict_in_generate=True,output_scores=True).scores
    return memorization_metric(res,inp_tensor[:,token_size//2:token_size])

if __name__ == '__main__':
    BATCH_SIZE = 100
    RESULTS_PATH = 'memorization_results.tfrecords'
    TOKEN_SIZE = 64
    TAKE_EVERY = 500
    
    args = parse_args()
    if(args.wandb_project_name):
        wandb.init(project=args.wandb_project_name)
        wandb.config.batch_size = BATCH_SIZE
        wandb.config.token_size = TOKEN_SIZE
        wandb.config.take_every = TAKE_EVERY

    records = TFrecordCreator(RESULTS_PATH) #store results

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo",rank=0,world_size=1)
    
    rec_queue = queue.Queue()
    ds = BatchedDataset(BATCH_SIZE,TAKE_EVERY,rec_queue)
    ds.start()

    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B").half().eval()
    model.parallelize({
        0:[0], #using least possible number of layers to accomodate for storing model scores
        1:list(range(1,10)),
        2:list(range(10,19)),
        3:list(range(19,28))
    })

    start_time = time.time()
    batch,indicies = rec_queue.get()
    step = 1
    while(batch is not None):
        batch = torch.tensor(batch,dtype=torch.int32,requires_grad=False)
        res = score(model,batch,TOKEN_SIZE)
        for i,j in zip(res,indicies):
            records.write(i,j)
        
        if(args.wandb_project_name):
            wandb.log({'memorization_metric':np.ma.masked_invalid(res).mean(),"index":indicies[-1]})
        
        print(f'{time.time() - start_time:3}s')
        start_time = time.time()
        batch,indicies = rec_queue.get()
        step += 1
    records.close()
