import matplotlib.pyplot as plt
import numpy as np


class AvgrageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


class PicBoard:
    def __init__(self, data, title=None, x_label=None, y_label=None):
        self.title = title
        self.x_label = x_label
        self.y_label = y_label
        self.data = data

    def draw_plot(self):
        plt.plot(self.data)
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.title(self.title)
        plt.show()


class LossPicBoard:
    def __init__(self, train_loss, valid_loss, title=None, x_label=None, y_label=None):
        self.title = title
        self.x_label = x_label
        self.y_label = y_label
        self.train_loss = train_loss
        self.valid_loss = valid_loss

    def draw_plot(self):
        fig, ax = plt.subplots()
        ax.plot(self.train_loss, label='train')
        ax.plot(self.valid_loss, label='valid')
        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.y_label)
        ax.set_title(self.title)
        ax.legend()
        plt.show()
        

class FunctionBoard:
        def __init__(self, x, y, true_y):
            self.x = x
            self.y = y
            self.true_y = true_y
        
        def draw_plot(self):
            fig, ax = plt.subplots()
            ax.scatter(self.x, self.y, label='y', alpha=0.2)
            ax.scatter(self.x, self.true_y, label='true_y', alpha=0.2)
            ax.legend()
            plt.show()
            
            
def get_accuracy(output, y):
    prediction = output
    prediction[prediction >= 0.5] = 1.
    prediction[prediction < 0.5] = 0.
    accuracy = np.mean((prediction[0,:] == y[0,:]))
    return accuracy