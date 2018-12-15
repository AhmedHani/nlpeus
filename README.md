<a href="url"><img src="https://github.com/AhmedHani/nlpeus/blob/master/nlpeus-logo.png" align='right'></a>


# nlpeus
[nlpeus](https://github.com/AhmedHani/nlpeus) is a training playground for several Natural Language Processing research projects. 

I created this repository for learning, so, I am not looking forward to getting state-of-the art results for each of the implemented tasks. My main target is to educate myself in handling several projects in one repository and writing clean, generic and maintainable code that can be used for most of the tasks.

I am using [PyTorch](https://pytorch.org/) to implement my models. I prefer it as it is not a very black box library. I can easily trace and debug my code to understand what it is going on.

For each of the implemented tasks, I will illustrate the experiments done with their results.

See you at the training arena!


## Requirements
- matplotlib==1.3.1
- numpy==1.13.1
- nltk==3.4
- pandas==0.23.4
- scipy==1.1.0
- scikit_learn==0.20.1
- terminaltables==3.1.0
- torch==1.0.0

```bash
    pip install -r requirements.txt
```


## Projects

- [Style Recognition](https://github.com/AhmedHani/nlpeus/tree/master/projects/style_recognition): Distinguishing between the writing styles that are used to write text (eg. Papers/News Headlines)