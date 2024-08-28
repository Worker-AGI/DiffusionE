# DiffusionE
Official code for the paper "DiffusionE: Reasoning on Knowledge Graphs via Diffusion-based Graph Neural Networks"


## Dependencies

- torch == 1.12.1
- torch_scatter == 2.0.9
- numpy == 1.21.6
- scipy == 1.10.1

## Reproduction

### Transductive settings (in `\transductive`)

#### Reproduction with training scripts

##### Family dataset

```
python3 train.py --data_path ./data/family/ --train --topk 100 --layers 8 --fact_ratio 0.90 --gpu 0
```

##### UMLS dataset
```
python3 train.py --data_path ./data/umls/ --train --topk 100 --layers 5 --fact_ratio 0.90 --gpu 0
```

##### WN18RR dataset
```
python3 train.py --data_path ./data/WN18RR/ --train --topk 1000 --layers 8 --fact_ratio 0.96 --gpu 0
```

##### FB15k-237 dataset
```
python3 train.py --data_path ./data/fb15k-237/ --train --topk 2000 --layers 7 --fact_ratio 0.99 --remove_1hop_edges --gpu 0
```

##### NELL995 dataset
```
python3 train.py --data_path ./data/nell/ --train --topk 2000 --layers 6 --fact_ratio 0.95 --gpu 0
```

##### YAGO3-10 dataset
```
python3 train.py --data_path ./data/YAGO/ --train --topk 1000 --layers 8 --fact_ratio 0.995 --gpu 0
```

### Inductive settings (in `\inductive`)

#### Reproduction with training scripts

The full training scripts can be found in [inductive/reproduce.sh](https://github.com/LARS-research/DiffusionE/blob/main/inductive/reproduce.sh).

For example, training on `WN18RR v1` dataset:

```
python3 train.py --data_path ./data/WN18RR_v2 --gpu 1
python3 train.py --data_path ./data/fb237_v1 --gpu 1
python3 train.py --data_path ./data/nell_v1 --gpu 4
```
