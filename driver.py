from style_recognition.data_processing import DataProcessing
from utils.data_utils import Batcher
from utils.experiment_utils import Experiment


dp = DataProcessing(news_file='./style_recognition/datasets/news.txt',
                    papers_file='./style_recognition/datasets/paper.txt',
                    pre_processing=False)

data = dp.news_data + dp.papers_data
class2index, index2class = dp.class2index, dp.index2class 

batcher = Batcher(data=data, batch_size=64, with_shuffle=True, divide_train_valid_test=True)

experiment = Experiment()


