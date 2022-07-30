# SDCNet
This is the implementation code of SDCNet, a GCN-based method, to efficiently predict cell line-specific SDCs facilitating the discovery of rational combination therapies. It can learn and fuse the unique features of drug combinations in a specific cell line and their invariant patterns across different cell lines, and the common features can improve the prediction accuracy for each cell line. 

![](fig1.pdf SDCNet)

## Requirements
Python 3.8 or higher  
pandas 1.3.5
numpy 1.21.2
tensorflow 2.4.1    


## Datasets
From the O’Neil dataset, the data from 31 cell lines are used to compare the performances of SDCNet and existing methods on SDC prediction. The data of the remaining eight cell lines are assembled as Transfer dataset to evaluate the performance of transfer learning on SDC prediction. For the other three datasets, all data are used to evaluate the generalization performance of the methods except some samples are excluded due to lack of the molecular information of drugs. The synergistic effect of the drug combinations are quantified through four types of synergy scores including Loewe additivity (Loewe), Bliss independence (Bliss), zero interaction potency (ZIP), and highest single agent (HSA).

1. O'Neil dataset
The experimental replicates of the drug combinations in the O’Neil dataset are averaged separately for various synergy types. 
2. ALMANAC dataset
3. CLOUD datset
4. FORCINA datset


## Training
python get_sdcnet_loewe.py -learning_rate 0.001 -epochs 10000 -embedding_dim 320 -drop_out 0.2 -weight_decay 0 -val_test_size 0.1

python get_sdcnet_bliss.py -learning_rate 0.001 -epochs 10000 -embedding_dim 320 -drop_out 0.2 -weight_decay 0 -val_test_size 0.1

python get_sdcnet_zip.py -learning_rate 0.001 -epochs 10000 -embedding_dim 320 -drop_out 0.2 -weight_decay 0 -val_test_size 0.1

python get_sdcnet_hsa.py -learning_rate 0.001 -epochs 10000 -embedding_dim 320 -drop_out 0.2 -weight_decay 0 -val_test_size 0.1

The most commonly used datset, O'Neil dataset, is used as an example to training the model. The hyperparameters are what we used in the study.

## Default parameters of the scrips
|Argument|Default|Description|
|---|---|----|
| learning_rate|  0.001|  Initial learning rate. |
| epochs|  1|  The number of training epochs. |
| embedding_dim|  320|  The number of dimension for drug embeddings. |
| dropout|  0.2|  Dropout rate (1 - keep probability) |
| weight_decay|  0|  Weight for L2 loss on embedding matrix. |
| val_test_size|  0.1|  the rate of validation and test samples. |

## predicting with pretrained model
python get_sdcnet_loewe_load_pretrainmodel.py -modelfile ../trained_model/sdcnet_loewe/best_model.ckpt

python get_sdcnet_bliss_load_pretrainmodel.py -modelfile ../trained_model/sdcnet_bliss/best_model.ckpt

python get_sdcnet_zip_load_pretrainmodel.py -modelfile ../trained_model/sdcnet_zip/best_model.ckpt

python get_sdcnet_hsa_load_pretrainmodel.py -modelfile ../trained_model/sdcnet_hsa/best_model.ckpt

We still use O'Neil dataset as example to make the prediction through the pretrained model. The size of pretrained models are too large, so they are accessible with baidu netdisk.


## Reference
Please cite our work if you find our code/paper is useful to your work.

```   
@article{Zhang, 
title={Predicting cell line-specific synergistic drug combinations through relational graph convolutional network with attention mechanism}, 
author={Peng Zhang, Shikui Tu, Wen Zhang, Lei Xu}, 
journal={Briefings in Bioinformatics}, 
volume={}, 
number={}, 
year={2022}, 
month={}, 
pages={} 
}
```
