# d3s_repro
This project was developed as part of [ML Reproducibility Challenge 2020 and Spring 2021](https://paperswithcode.com/rc2020) and aims to reproduce results from paper [D3S - A Discriminative Single Shot Segmentation Tracker](https://openreview.net/forum?id=6N0v-QkkLD). The paper describes new neural network architecture - D3S - for both visual object tracking and video object segmentation. Original implementation can be found [here](https://github.com/alanlukezic/d3s).

## Scope of Reproducibility
In our reproducibility study, we focused on training and evaluation of D3S for visual object tracking tasks due to limited time.

## Methodology
Our work is based on code provided by the authors of the original paper. The training code was reorganized and partially re-implemented. As a result, our version consists of only the most necessary code (the original code consists of other experiments not presented in the paper). For model evaluation, we use the pytracking framework following the authors of the original article.

We use NVIDIA Tesla V100 GPU for model training and validation. For model training run:
```
python -m experiments.run_training
```
For model evaluation run:
- vot2016 dataset:
  ```
  python pytracking/run_tracker.py segm default_params --dataset vot16
  ```
- vot2018 dataset:
  ```
  python pytracking/run_tracker.py segm default_params --dataset vot18
  ```
- Got10-k dataset:
  ```
  python pytracking/run_tracker.py segm default_params --dataset gotv
  ```
- TrackingNet dataset:
  ```
  python pytracking/run_tracker.py segm default_params --dataset tn
  ```
