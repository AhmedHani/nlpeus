# Copyright (c) 2018-present, Ahmed H. Al-Ghidani.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

__author__ = "Ahmed H. Al-Ghidani"
__copyright__ = "Copyright 2018, The nlpeus Project, https://github.com/AhmedHani/nlpeus"
__license__ = "GPL"
__maintainer__ = "Ahmed H. Al-Ghidani"
__email__ = "ahmed.hani.ibrahim@gmail.com"

import os
import time
import shutil
import codecs
import datetime
import matplotlib.pyplot as plt


class Experiment(object):

    def __init__(self, total_samples,
                 total_training_samples,
                 total_valid_samples,
                 total_test_samples,
                 model_name,
                 epochs,
                 batch_size,
                 number_classes,
                 input_length,
                 device,
                 author_name=None):
        self.total_samples = total_samples
        self.total_training_samples = total_training_samples
        self.total_valid_samples = total_valid_samples
        self.total_test_samples = total_test_samples
        self.model_name = model_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.number_classes = number_classes
        self.input_length = input_length
        self.device = device
        self.author_name = author_name

        self.saved_model_dir = None
        self.saved_data_dir = None
        self.eval_file_path = None
        self.pickle_file_path = None
        self.info_file_path = None
        self.learning_curve_image = None

    def create(self, research_interface):
        project_dir = os.path.dirname(os.path.dirname(research_interface))

        if not os.path.exists(os.path.join(project_dir, 'shared')):
            os.mkdir(os.path.join(project_dir, 'shared'))

        experiment_resources = os.path.join(project_dir, 'shared')

        if not os.path.exists(os.path.join(experiment_resources, 'experiments')):
            os.mkdir(os.path.join(experiment_resources, 'experiments'))

        experiment_resources = os.path.join(experiment_resources, 'experiments')

        experiment_name = 'nclasses({})ninput({})model({})epochs({})batchsize({})device({})'.format(
            self.number_classes,
            self.input_length,
            self.model_name,
            self.epochs,
            self.batch_size,
            self.device
        )

        self.experiment_dir = os.path.join(experiment_resources, experiment_name)

        if os.path.exists(self.experiment_dir):
            print('this experiment setup is already done before, do you want to repeat it? [yes/no]')
            answer = str(input())

            if answer.strip().rstrip().lower() == 'yes' or answer.strip().rstrip().lower() == 'y':
                print('do you want to overwrite the previous experiment? [yes/no]')
                answer = str(input())

                if answer.strip().rstrip().lower() == 'yes' or answer.strip().rstrip().lower() == 'y':
                    shutil.rmtree(self.experiment_dir)
                else:
                    print('write a suffix for the new experiment name')
                    answer = str(input())
                    self.experiment_dir = os.path.join(experiment_resources, experiment_name + '_{}'.format(answer))
            else:
                print('experiment will be terminated')
                exit()

        os.mkdir(self.experiment_dir)
        print('experiment location: {}\n'.format(self.experiment_dir))

        self.saved_model_dir = os.path.join(self.experiment_dir, 'saved_model')
        os.mkdir(self.saved_model_dir)

        self.saved_data_dir = os.path.join(self.experiment_dir, 'saved_data')
        os.mkdir(self.saved_data_dir)

        self.eval_file_path = os.path.join(self.experiment_dir, 'eval.log')
        self.pickle_file_path = os.path.join(self.experiment_dir, 'eval.pkl')
        self.info_file_path = os.path.join(self.experiment_dir, 'info.txt')
        self.learning_curve_image = os.path.join(self.experiment_dir, 'learning_curve.png')

        with codecs.open(self.info_file_path, 'w', encoding='utf-8') as writer:
            writer.write('author: {}\n'.format(self.author_name))

            project_name = os.path.basename(project_dir)
            writer.write('project: {}\n'.format(project_name))

            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            writer.write('date and time: {}\n\n'.format(current_time))

            writer.write('experiment setup:\n')
            writer.write('\t total training samples: {}\n'.format(self.total_training_samples))
            writer.write('\t total validation samples: {}\n'.format(self.total_valid_samples))
            writer.write('\t total testing samples: {}\n'.format(self.total_test_samples))
            writer.write('\t model: {}\n'.format(self.model_name))
            writer.write('\t epochs: {}\n'.format(self.epochs))
            writer.write('\t batch size: {}\n'.format(self.batch_size))
            writer.write('\t number of classes: {}\n'.format(self.number_classes))
            writer.write('\t input length: {}\n'.format(self.input_length))
            writer.write('\t device: {}\n'.format(self.device))

        return self.experiment_dir

    def run(self, trainer, batcher, encoder, data_axis, transformations=None, class2index=None):

        try:
            epochs_average_losses = []

            for epoch in range(1, self.epochs + 1):
                batches_losses = []
                cnter = 0

                while batcher.hasnext(target='train'):
                    current_batch = batcher.nextbatch(target='train')
                    X = [item[data_axis['X']] for item in current_batch]
                    Y = [item[data_axis['Y']] for item in current_batch]

                    if transformations is not None:
                        for transformation in transformations:
                            X = transformation(X)

                    x_train = encoder.encode(X)

                    if class2index is None:
                        y_train = Y
                    else:
                        y_train = [[class2index[item]] for item in Y]

                    batch_loss = trainer.fit_batch(x_train, y_train)

                    batches_losses.append(batch_loss)

                    print("Epoch: {}/{}\tBatch: {}/{}\tLoss: {}".format(epoch,
                                                                        self.epochs,
                                                                        cnter,
                                                                        batcher.total_batches(target='train'),
                                                                        batch_loss))
                    cnter += 1

                print("\nEpoch: {}/{}\tAverageLoss: {}\n".format(epoch, self.epochs,
                                                                 sum(batches_losses) / float(len(batches_losses))))
                epochs_average_losses.append(sum(batches_losses) / float(len(batches_losses)))

                time.sleep(3)

                batcher.initialize()
                batcher.shuffle_me('train')
        except KeyboardInterrupt:
            print('End training at epoch: {}'.format(epoch))
            print('Begin evaluating the model on the validation data')

        plt.plot(range(len(epochs_average_losses)), epochs_average_losses)
        plt.xlabel('epochs')
        plt.ylabel('loss value')
        plt.title('learning curve during the training phase')
        plt.savefig(self.learning_curve_image)

        try:
            cnter = 0

            while batcher.hasnext(target='valid'):
                current_batch = batcher.nextbatch(target='valid')
                X = [item[data_axis['X']] for item in current_batch]
                Y = [item[data_axis['Y']] for item in current_batch]

                if transformations is not None:
                    for transformation in transformations:
                        X = transformation(X)

                x_valid = encoder.encode(X)

                if class2index is None:
                    y_valid = Y
                else:
                    y_valid = [class2index[item] for item in Y]

                conf_matrix = trainer.eval_batch(x_valid, y_valid)

                print("Batch: {}/{}".format(cnter, batcher.total_batches('valid')))
                cnter += 1
        except KeyboardInterrupt:
            print('End validating at batch: {}'.format(cnter))
            print('Begin writing results and evaluations')

        trainer.show_evaluation(precision_recall_fscore=True,
                                conf_matrix=True,
                                accuracy=True,
                                stdout=self.eval_file_path,
                                pickle_path=self.pickle_file_path)

        model_weights_name = os.path.join(self.saved_model_dir, 'torch_model.pt')
        trainer.save(model_weights_name)

        print('\nexperiment location: {}\n'.format(self.experiment_dir))
