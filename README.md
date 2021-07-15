# d3s_repro
This project was developed as part of [ML Reproducibility Challenge 2020 and Spring 2021](https://paperswithcode.com/rc2020) and aims to reproduce results from paper [D3S - A Discriminative Single Shot Segmentation Tracker](https://openreview.net/forum?id=6N0v-QkkLD). The paper describes new neural network architecture - D3S - for both visual object tracking and video object segmentation. Original implementation can be found [here](https://github.com/alanlukezic/d3s).

## Scope of Reproducibility
In our reproducibility study, we focused on training and evaluation of D3S for visual object tracking tasks due to limited time.

## Methodology
Our work is based on code provided by the authors of the original paper. The training code was reorganized and partially re-implemented. As a result, our version consists of only the most necessary code (the original code consists of other experiments not presented in the paper). For model evaluation, we use the pytracking framework following the authors of the original article.

We use ```NVIDIA Tesla V100``` GPU with ```CUDA 9.2``` and ```pytorch 1.7.1```  for model training and validation. For model training you need to download [yotube-vos 2018](https://youtube-vos.org/dataset/) dataset in ```DATA/``` folder and run:
```
python -m experiments.run_training
```
For model evaluation you need to download [vot2016](https://www.votchallenge.net/vot2016/), [vot2108](https://www.votchallenge.net/vot2018/), [GOT10-K](http://got-10k.aitestunion.com/index) and [TrackingNet](https://tracking-net.org/) datasets, set correct paths in ```pytracking/evaluation/local.py``` and run:
- vot2016 dataset:
  ```
  python pytracking/run_tracker.py segm default_params --dataset vot16
  ```
- vot2018 dataset:
  ```
  python pytracking/run_tracker.py segm default_params --dataset vot18
  ```
- GOT10-k dataset:
  ```
  python pytracking/run_tracker.py segm default_params --dataset gotv
  ```
- TrackingNet dataset:
  ```
  python pytracking/run_tracker.py segm default_params --dataset tn
  ```
The tools provided with the datasets were used to calculate the metrics.

## Results
Below is a comparison of the results obtained with those given in the original article:

|  Dataset | Metric | Our result | Original result |
| ---- | ---- | ---- | ---- |
| vot 2016 | EAO | 0.494 | 0.493 |
|          | Acc. | 0.67 | 0.66 |
|          | Rob. | 0.131 | 0.131 |
| vot 2018 | EAO | 0.487 | 0.489 |
|          | Acc. | 0.63 | 0.64 |
|          | Rob. | 0.153 | 0.150 |
| GOT10-k | AO | 0.60 | 59.7 |
|  | SR0.75 | 47.3 | 46.2 |
|  | SR0.5 |68.6 | 67.6 |
| TrackingNet | AUC | 72.8 | 72.8 |
|  | Prec. | 66.5 | 66.4 |
|  | Prec.N | 76.8 | 76.8 |

Training takes 16 hours. Evaluation speed listed in table below:

| Dataset | vot2016 | vot2018 | GOT10-k | TrackingNet |
| ------- | ------- | ------- | ------- | ----------- |
| Our FPS |    22   |   21    |   16    |   23        |

In the original paper 25fps evaluation speed was reported.
