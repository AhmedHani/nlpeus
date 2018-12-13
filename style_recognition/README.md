```
author: A.H. Al-Ghidani
project: style_recognition
date and time: 2018-12-13 14:02

########## Experiment #1 ##########

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