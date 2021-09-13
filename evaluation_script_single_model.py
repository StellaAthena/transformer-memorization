from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import torch
import tensorflow as tf
import numpy as np
import wandb
from memorization_metric import memorization_metric
import argparse
from result_records import TFrecordCreator

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

def device_map():
    """
    Creates a device map to utilize all 8 gpu devices
    """
    return {
        0:[0,1],
        1:[2,3,4,5],
        2:[6,7,8,9],
        3:[10,11,12,13],
        4:[14,15,16,17],
        5:[18,19,20,21],
        6:[22,23,24],
        7:[25,26,27]
    }
if __name__ == '__main__':
    BATCH_SIZE = 512
    RESULTS_PATH = 'temp.tfrecords' #use gcs path if you want to store them somewhere else
    TOKEN_SIZE = 64
    
    args = parse_args()
    if(args.wandb_project_name):
        wandb.init(project=args.wandb_project_name)
    
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B").half().eval()
    model.parallelize(device_map())

    records = TFrecordCreator(RESULTS_PATH) #store results
    step = 1
    for path in get_dataset_paths(args):
        ds = tf.data.TFRecordDataset(path).map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE)
        for batch in tqdm(iter(ds)):
            batch = torch.tensor(batch.numpy(),dtype=torch.int32,requires_grad=False)
            res = score(model,batch,TOKEN_SIZE)
            
            for i in res:
                records.write(i)
            
            if(args.wandb_project_name):
                wandb.log({'memorization_metric':np.average(x)},step)
    records.close()
