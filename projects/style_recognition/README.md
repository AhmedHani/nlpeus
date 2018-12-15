```
########### Experiment #1 ###########

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



########### Experiment #2 ###########

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