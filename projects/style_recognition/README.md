# Problem Definition
Distinguishing between the writing styles that are used to write text (eg. Papers/News Headlines)


# Experiments


| Experiment Setup | Average Precision | Average Recall | Average F-score | Total Accuracy
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Number of Classes: 2<br>Input Length: 50<br>Model Name: CharRNN<br>Epochs: 10<br>Batch Size: 128<br>Device: cpu<br>Notes: None | 0.623 | 0.605 | 0.588 | 0.602 |
| Number of Classes: 2<br>Input Length: 50<br>Model Name: CharRNN<br>Epochs: 10<br>Batch Size: 128<br>Device: cuda<br>Notes: None | 0.984 | 0.984 | 0.984 | 0.984 |
| Number of Classes: 2<br>Input Length: 50<br>Model Name: CharRNN<br>Epochs: 8<br>Batch Size: 128<br>Device: cpu<br>Notes: None | 0.598 | 0.594 | 0.590 | 0.596 |
| Number of Classes: 2<br>Input Length: 50<br>Model Name: CharRNN<br>Epochs: 2<br>Batch Size: 128<br>Device: cpu<br>Notes: None | 0.767 | 0.719 | 0.706 | 0.72 |
| Number of Classes: 2<br>Input Length: 50<br>Model Name: CharRNN<br>Epochs: 10<br>Batch Size: 128<br>Device: cuda<br>Notes: _new | 0.986 | 0.986 | 0.986 | 0.986 |
| Number of Classes: 2<br>Input Length: 50<br>Model Name: CharRNN<br>Epochs: 3<br>Batch Size: 128<br>Device: cuda<br>Notes: None | 0.982 | 0.982 | 0.982 | 0.982 |
| Number of Classes: 2<br>Input Length: 50<br>Model Name: CharRNN<br>Epochs: 5<br>Batch Size: 128<br>Device: cuda<br>Notes: None | 0.984 | 0.984 | 0.984 | 0.984 |

# Experiments Details

```
########### Experiment #1 ###########

author: A.H. Al-Ghidani
project: style_recognition
date and time: 2018-12-15 21:24

experiment setup:
	 total training samples: 172831
	 total validation samples: 21603
	 total testing samples: 21603
	 model: CharRNN
	 epochs: 10
	 batch size: 128
	 number of classes: 2
	 input length: 50
	 device: cuda


+--------------+------------+
| class index  | class name |
+--------------+------------+
| class news   | 0          |
| class papers | 1          |
+--------------+------------+

confusion matrix
+---------+---------+---------+
| --      | class 0 | class 1 |
+---------+---------+---------+
| class 0 | 10730   | 233     |
| class 1 | 117     | 10552   |
+---------+---------+---------+

accuracy: 0.984

+-----------+--------------------+--------------------+
| --        | class 0            | class 1            |
+-----------+--------------------+--------------------+
| precision | 0.978746693423333  | 0.9890336488893055 |
| recall    | 0.9892136074490643 | 0.9783959202596199 |
| fscore    | 0.9839523154516276 | 0.9836860259159131 |
+-----------+--------------------+--------------------+

average fscore: 0.9838191706837704



########### Experiment #2 ###########

author: A.H. Al-Ghidani
project: style_recognition
date and time: 2018-12-13 15:37

experiment setup:
	 total training samples: 172831
	 total validation samples: 21603
	 total testing samples: 21603
	 model: CharRNN
	 epochs: 2
	 batch size: 128
	 number of classes: 2
	 input length: 50
	 device: cpu


+--------------+------------+
| class index  | class name |
+--------------+------------+
| class news   | 0          |
| class papers | 1          |
+--------------+------------+

confusion matrix
+---------+---------+---------+
| --      | class 0 | class 1 |
+---------+---------+---------+
| class 0 | 1076    | 565     |
| class 1 | 81      | 582     |
+---------+---------+---------+

accuracy: 0.72

+-----------+--------------------+--------------------+
| --        | class 0            | class 1            |
+-----------+--------------------+--------------------+
| precision | 0.65569774527727   | 0.8778280542986425 |
| recall    | 0.9299913569576491 | 0.5074106364428945 |
| fscore    | 0.7691208005718371 | 0.6430939226519338 |
+-----------+--------------------+--------------------+

average fscore: 0.7061073616118854



########### Experiment #3 ###########

author: A.H. Al-Ghidani
project: style_recognition
date and time: 2018-12-13 14:02

experiment setup:
	 total training samples: 172831
	 total validation samples: 21603
	 total testing samples: 21603
	 model: CharRNN
	 epochs: 5
	 batch size: 128
	 number of classes: 2
	 input length: 50
	 device: cuda


+--------------+------------+
| class index  | class name |
+--------------+------------+
| class news   | 0          |
| class papers | 1          |
+--------------+------------+

confusion matrix
+---------+---------+---------+
| --      | class 0 | class 1 |
+---------+---------+---------+
| class 0 | 10723   | 138     |
| class 1 | 210     | 10561   |
+---------+---------+---------+

accuracy: 0.984

+-----------+--------------------+--------------------+
| --        | class 0            | class 1            |
+-----------+--------------------+--------------------+
| precision | 0.9872939876622778 | 0.980503203045214  |
| recall    | 0.9807920973200402 | 0.9871015982802132 |
| fscore    | 0.9840323024685693 | 0.9837913367489521 |
+-----------+--------------------+--------------------+

average fscore: 0.9839118196087607

```