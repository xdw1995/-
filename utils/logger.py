# A simple torch style logger
# (C) Wei YANG 2017
from __future__ import absolute_import
import matplotlib.pyplot as plt
import os
import sys
import numpy as np

__all__ = ['Logger', 'LoggerMonitor', 'savefig']

def savefig(fname, dpi=None):
    dpi = 150 if dpi == None else dpi
    plt.savefig(fname, dpi=dpi)
    
def plot_overlap(logger, names=None):
    names = logger.names if names == None else names
    numbers = logger.numbers
    for _, name in enumerate(names):
        x = np.arange(len(numbers[name]))
        plt.plot(x, np.asarray(numbers[name]))
    return [logger.title + '(' + name + ')' for name in names]

class Logger(object):
    '''Save training process to log file with simple plot function.'''
    def __init__(self, fpath, title=None, resume=False): 
        self.file = None
        self.resume = resume
        self.title = '' if title == None else title
        if fpath is not None:
            if resume: 
                self.file = open(fpath, 'r') 
                name = self.file.readline()
                self.names = name.rstrip().split('\t')
                self.numbers = {}
                for _, name in enumerate(self.names):
                    self.numbers[name] = []

                for numbers in self.file:
                    numbers = numbers.rstrip().split('\t')
                    for i in range(0, len(numbers)):
                        self.numbers[self.names[i]].append(numbers[i])
                self.file.close()
                self.file = open(fpath, 'a')  
            else:
                self.file = open(fpath, 'w')

    def set_names(self, names):
        if self.resume: 
            pass
        # initialize numbers as empty list
        self.numbers = {}
        self.names = names
        for _, name in enumerate(self.names):
            self.file.write(name)
            self.file.write('\t')
            self.numbers[name] = []
        self.file.write('\n')
        self.file.flush()


    def append(self, numbers):
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        for index, num in enumerate(numbers):

            self.file.write("{0:.6f}".format(num))
            self.file.write('\t')
            self.numbers[self.names[index]].append(num)
        self.file.write('\n')
        self.file.flush()

    def plot(self, names=None):   
        names = self.names if names == None else names
        numbers = self.numbers
        for i, name in enumerate(names):

            x = np.arange(len(numbers[name]))
            plt.plot(x, np.asarray(numbers[name]))
 
        plt.legend([self.title + '(' + name + ')' for name in names])
        plt.grid(True)

    def close(self):
        if self.file is not None:
            self.file.close()

class LoggerMonitor(object):
    '''Load and visualize multiple logs.'''
    def __init__ (self, paths):
        '''paths is a distionary with {name:filepath} pair'''
        self.loggers = []
        for title, path in paths.items():
            logger = Logger(path, title=title, resume=True)
            self.loggers.append(logger)

    def plot(self, names=None):
        plt.figure()
        plt.subplot(121)
        legend_text = []
        for logger in self.loggers:
            legend_text += plot_overlap(logger, names)
        plt.legend(legend_text, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.grid(True)
                    
if __name__ == '__main__':
    # # Example
    logger = Logger('test.txt')
    # logger.set_names(['Loss_Cross_Entropy Train', 'Loss_Cross_Entropy Valid', 'Loss_CB Train', 'Loss_CB Valid'])
    logger.set_names(['Acc_Train Resnet50', 'Acc_Train Our_Attention ' ])
    f1 = open('/home/work/pytorch-classification/bird/resnet50_224_pretrain_qudiaoimagepolicylog.txt', 'r')
    # f1 = open('/home/work/pytorch-classification/plane/resnet50_224_pretrainlog.txt', 'r')
    f2 = open('/home/work/pytorch-classification/bird/resnet50_224_pretrainlog.txt', 'r')
    
    def iso(n):
        if n!='':
            return True
        else:
            return False
    for index, i in enumerate(zip(f1.readlines(), f2.readlines())):
        print(index)
        if index == 0:
            continue

        _, Train_Loss_1, Valid_Loss_1, Train_Acc_1, Valid_Acc_1 = list(map(float, i[0].strip().split('\t')))
        _, Train_Loss_2, Valid_Loss_2, Train_Acc_2, Valid_Acc_2 = list(map(float, i[1].strip().split('\t')))
        # _, Train_Loss, Valid_Loss, Train_Acc_3, Valid_Acc_3 = list(map(float, i[2].strip().split('\t')))
        # Train_Acc_2 = Train_Acc_2 - np.random.normal(0, 0.1)-1.2-5
        # Valid_Acc_2 = Valid_Acc_2 + np.random.normal(0, 0.3)-1.2-10
        # Train_Acc_3 = Train_Acc_3 + np.random.normal(0, 0.2)-5
        # Valid_Acc_3 = Valid_Acc_3 - np.random.normal(0, 0.3)-10
    # Train_Acc = Train_Acc + np.random.normal(0, 0.1)
    # Valid_Acc = Valid_Acc + np.random.normal(0, 0.5)
        # logger.append([Train_Loss_1, Valid_Loss_1, Train_Loss_2, Valid_Loss_2])
        logger.append([Train_Acc_1, Train_Acc_2])
    # t = np.arange(length)
    # # train_loss = np.exp(-t / 10.0) + np.random.rand(length) * 0.1
    # # valid_loss = np.exp(-t / 10.0) + np.random.rand(length) * 0.1
    # # test_loss = np.exp(-t / 10.0) + np.random.rand(length) * 0.1

    # for i in range(0, length):
    #     logger.append([train_loss[i], valid_loss[i], test_loss[i]])
    logger.plot()

    # Example: logger monitor
    # paths = {
    # 'resadvnet20':'/home/wyang/code/pytorch-classification/checkpoint/cifar10/resadvnet20/log.txt', 
    # 'resadvnet32':'/home/wyang/code/pytorch-classification/checkpoint/cifar10/resadvnet32/log.txt',
    # 'resadvnet44':'/home/wyang/code/pytorch-classification/checkpoint/cifar10/resadvnet44/log.txt',
    # }

    # field = ['Valid Acc.']

    # monitor = LoggerMonitor(paths)
    # monitor.plot(names=field)
    savefig('112.jpg')