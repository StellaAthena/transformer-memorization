# transformer-memorization


## Trained Model Checkpoints

To support this effort, we have made several trained model checkpoints publicly avaliable.

### Model Details

| Params        | Batch Size | Num Layers | Dim Model | Num Heads | Cores per Replica | Replicas per Batch | GAS | TPU Size |
|:-------------:|:----------:|:----------:|:---------:|:---------:|:-----------------:|:------------------:|:---:|:--------:|
| 162,675,936   | 512        | 12         | 768       | 16        | 8                 | 2                  | 8   | 256      |
| 304,663,776   | 512        | 32         | 768       | 16        | 8                 | 1                  | 16  | 256      |
|               | 512        | 28         |           |           |                   |                    |     | 256      |
|               | 512        | 28         |           |           |                   |                    |     | 256      |
|               | 512        | 28         |           |           |                   |                    |     | 256      |
|               | 512        | 28         |           |           |                   |                    |     | 256      |
|               | 512        | 28         |           |           |                   |                    |     | 256      |
|               | 512        | 28         |           |           |                   |                    |     | 256      |
| 6,053,381,344 | 512        | 28         | 4,096     | 16        | 16                | 1                  | 16  | 256      |

<br><br>

## Evaluation details
* The scripts `evaluation_script.py` and `evaluation_script_single_model.py` evaluate the memorization of input tfrecords based on the memorization metric
* Scripts have the following arguments:
    * `--wandb-project-name` : wandb project name for the current run
