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
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import pickle as pkl
from glob import glob
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

                trainer.eval_batch(x_valid, y_valid)

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


class ExperimentSummarizer(object):

    def __init__(self, experiments_location):
        self.experiments_location = experiments_location
        self.output_location = os.path.dirname(os.path.dirname(os.path.dirname(self.experiments_location)))
        self.project_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(self.experiments_location)))).upper()

    def run(self):
        summary_file = os.path.join(self.output_location, 'summary.txt')
        sheet_file = os.path.join(self.output_location, 'experiments.xlsx')

        experiments_data = []

        with codecs.open(summary_file, 'w', encoding='utf-8') as writer:
            for i, experiment_dir in enumerate(glob(self.experiments_location + '/*/')):
                if os.path.basename(os.path.dirname(experiment_dir)) == '__pycache__':
                    continue

                experiment_name = os.path.basename(os.path.dirname(experiment_dir))
                experiment_info_file = os.path.join(experiment_dir, 'info.txt')
                experiment_eval_file = os.path.join(experiment_dir, 'eval.log')
                experiment_results_file = os.path.join(experiment_dir, 'eval.pkl')
                experiment_learning_curve_image = os.path.join(experiment_dir, 'learning_curve.png')

                with codecs.open(experiment_info_file, 'r', encoding='utf-8') as reader:
                    writer.write('########### Experiment #{} ###########\n\n'.format(i + 1))
                    writer.write(reader.read())
                    writer.write('\n\n')
                
                with codecs.open(experiment_eval_file, 'r', encoding='utf-8') as reader:
                    writer.write(reader.read())
                    writer.write('\n\n')
                
                experiments_data.append([experiment_name, experiment_info_file, experiment_eval_file, 
                                        experiment_results_file, experiment_learning_curve_image])
        
        research_sheet = [('Project Name', self.project_name),
                          ('Export Date', datetime.datetime.now().strftime("%Y-%m-%d %H:%M")),
                          ('Number of Experiments', len(experiments_data)),
                          ('--', '--'),
                          ('Experiment Setup', 'Average Precision', 'Average Recall', 'Average F-score', 'Total Accuracy')]
        
        writer = pd.ExcelWriter(sheet_file, engine='xlsxwriter')

        for experiment_item in experiments_data:
            name_tokens = experiment_item[0].replace('(', ' ').replace(')', ' ').strip().rstrip().split()
            nclasses = name_tokens[1]
            ninput = name_tokens[3]
            model = name_tokens[5]
            epochs = name_tokens[7]
            batch_size = name_tokens[9]
            device = name_tokens[11]

            try:
                suffix = name_tokens[12]
            except IndexError:
                suffix = None

            research_experiment_setup = "Number of Classes: {}\nInput Length: {}\nModel Name: {}\nEpochs: {}\nBatch Size: {" \
                                "}\nDevice: {}\nNotes: {}".format(nclasses, ninput, model, epochs, batch_size, device, suffix)

            results_pickle_file = experiment_item[3]

            with open(results_pickle_file, 'rb') as reader:
                results = pkl.load(reader)

            research_sheet.append((research_experiment_setup,
                                    results['average_precision'],
                                    results['average_recall'],
                                    results['average_fscore'],
                                    results['accuracy']))

        df = pd.DataFrame(research_sheet)
        df.to_excel(writer, 'All Experiments', index=0, index_label=0, header=False)

        workbook = writer.book
        worksheet = writer.sheets['All Experiments']

        format_ = workbook.add_format()
        format_.set_align('center')
        format_.set_align('vcenter')
        format_.set_text_wrap()
        worksheet.set_column('A:Z', 30, format_)

        for i, experiment_item in enumerate(experiments_data):
            current_sheet = "Experiment {}".format(i + 1)

            name_tokens = experiment_item[0].replace('(', ' ').replace(')', ' ').strip().rstrip().split()
            nclasses = name_tokens[1]
            ninput = name_tokens[3]
            model = name_tokens[5]
            epochs = name_tokens[7]
            batch_size = name_tokens[9]
            device = name_tokens[11]
            suffix = name_tokens[13] if len(name_tokens) == 14 else None

            experiment_info_file = experiment_item[1]
            experiment_results_file = experiment_item[3]
            experiment_learning_curve_image = experiment_item[4]

            with codecs.open(experiment_info_file, 'r', encoding='utf-8') as reader:
                experiment_info = list(map(lambda v: v.strip().rstrip(), reader.readlines()))

                author = experiment_info[0]
                project = experiment_info[1]
                date_and_time = experiment_info[2]
                total_training_samples = experiment_info[5].split()[-1]
                total_valid_samples = experiment_info[6].split()[-1]
                total_test_samples = experiment_info[7].split()[-1]

                sheet = [('Project Name', project), ('Author', author), ('Date and Time', date_and_time),
                        ('Number of Training Samples', total_training_samples),
                        ('Number of Validation Samples', total_valid_samples), ('Number of Testing Samples', total_test_samples),
                        ('--', '--'), ('Number of Classes', nclasses), ('Input Length', ninput), ('Model', model),
                        ('Epochs', epochs), ('Batch Size', batch_size), ('Device', device), ('Notes', suffix), ('--', '--')]

                with open(experiment_results_file, 'rb') as reader:
                    results = pkl.load(reader)

                sheet.append(('Average Precision', results['average_precision']))
                sheet.append(('Average Recall', results['average_recall']))
                sheet.append(('Average F-score', results['average_fscore']))
                sheet.append(('--', '--'))

                df = pd.DataFrame(sheet)
                df.to_excel(writer, current_sheet, index=0, index_label=0, header=False)

                workbook = writer.book
                worksheet = writer.sheets[current_sheet]

                worksheet.insert_image('C3', experiment_learning_curve_image)

                format_ = workbook.add_format()
                format_.set_align('center')
                format_.set_align('vcenter')
                format_.set_text_wrap()

                worksheet.set_column('A:Z', 30, format_)

        writer.save()        
        