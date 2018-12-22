# Problem Definition
Distinguishing between the writing styles that are used to write text.


## Papers/News Headlines

- A binary classification problem. We have some [dataset](https://github.com/AhmedHani/nlpeus/tree/master/projects/style_recognition/datasets/paper-news), collected from [Style Transfer in Text: Exploration and Evaluation](https://arxiv.org/pdf/1711.06861.pdf), for papers and news headlines. The target is to classify a given headline into either a paper or news
- We mainly used a [Character-level LSTM architecture](https://github.com/AhmedHani/nlpeus/blob/master/models/torch_charnn.py) to do our experiments (detailed experiments and hyperparameters below)
- You can view the full experiments from [here](https://github.com/AhmedHani/nlpeus/tree/master/projects/style_recognition/shared/experiments/papers_news)


### Dataset analysis
```
number of instances: 216041
average number of words: 9
average number of chars: 68
number of unique words: 155030
number of unique chars: 174
classes frequencies: 
		news: 108503
		papers: 107538
```


### Experiments

You can find the experiments summary from [here](https://github.com/AhmedHani/nlpeus/blob/master/projects/style_recognition/shared/experiments/papers_news/experiments.xlsx)

For all experiments
- Pad and truncate the sentences to have max chars of length 50

| Experiment Setup | Average Precision | Average Recall | Average F-score | Total Accuracy
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Number of Classes: 2<br>Input Length: 50<br>Model Name: CharRNN<br>Epochs: 10<br>Batch Size: 128<br>Device: cpu<br>Notes: None | 0.623 | 0.605 | 0.588 | 0.602 |
| Number of Classes: 2<br>Input Length: 50<br>Model Name: CharRNN<br>Epochs: 10<br>Batch Size: 128<br>Device: cuda<br>Notes: None | 0.984 | 0.984 | 0.984 | 0.984 |
| Number of Classes: 2<br>Input Length: 50<br>Model Name: CharRNN<br>Epochs: 8<br>Batch Size: 128<br>Device: cpu<br>Notes: None | 0.598 | 0.594 | 0.590 | 0.596 |
| Number of Classes: 2<br>Input Length: 50<br>Model Name: CharRNN<br>Epochs: 2<br>Batch Size: 128<br>Device: cpu<br>Notes: None | 0.767 | 0.719 | 0.706 | 0.72 |
| Number of Classes: 2<br>Input Length: 50<br>Model Name: CharRNN<br>Epochs: 10<br>Batch Size: 128<br>Device: cuda<br>Notes: _new | 0.986 | 0.986 | 0.986 | 0.986 |
| Number of Classes: 2<br>Input Length: 50<br>Model Name: CharRNN<br>Epochs: 3<br>Batch Size: 128<br>Device: cuda<br>Notes: None | 0.982 | 0.982 | 0.982 | 0.982 |
| Number of Classes: 2<br>Input Length: 50<br>Model Name: CharRNN<br>Epochs: 5<br>Batch Size: 128<br>Device: cuda<br>Notes: None | 0.984 | 0.984 | 0.984 | 0.984 |


## [Spooky Author Identification](https://www.kaggle.com/c/spooky-author-identification/data)

- A Kaggle competition that you can find from [here](https://www.kaggle.com/c/spooky-author-identification) 
- We still do experiments

### Dataset analysis

```
number of instances: 19579
average number of words: 26
average number of chars: 148
number of unique words: 45839
number of unique chars: 84
classes frequencies: 
		EAP: 7900
		MWS: 6044
		HPL: 5635
```


### Experiments

| Experiment Setup | Average Precision | Average Recall | Average F-score | Total Accuracy
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Number of Classes: 3<br>Input Length: 50<br>Model Name: MultiCharRNN<br>Epochs: 50<br>Batch Size: 16<br>Device: cuda<br>Notes: None | 0.521 | 0.519 | 0.519 | 0.525 |
| Number of Classes: 3<br>Input Length: 50<br>Model Name: MultiCharRNN<br>Epochs: 3<br>Batch Size: 32<br>Device: cpu<br>Notes: _ | 0.349 | 0.345 | 0.335 | 0.37 |
