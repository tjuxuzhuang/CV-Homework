from network import *
from utils import *
from dataloader import *
import argparse

parser = argparse.ArgumentParser("MNIST")
parser.add_argument('--data', type=str, default=r"./mnist/")
parser.add_argument('--batch_size', type=int, default=1000, help='batch size')
parser.add_argument('--epochs', type=int, default=50, help='epochs')
parser.add_argument('--learning_rate', type=float, default=0.05, help='init learning rate')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--shuffle', type=bool, default=False, help='shuffle')
parser.add_argument('--input_size', type=int, default=784, help='输入向量维度')
parser.add_argument('--output_size', type=int, default=10, help='输出向量维度')
parser.add_argument('--loss', type=str, default="logprobs", help='使用的损失函数')
parser.add_argument('--classes', type=int, default=10, help='分类任务中的种类')
parser.add_argument('--lr_decay', default=[25, 0.1])
args = parser.parse_args()


valid_accuracy = []

def main():
    model = Network(args.input_size, args.batch_size, args.loss, args.learning_rate)
    model.add_layer(100, 'Sigmoid')
    model.add_layer(100, 'Sigmoid')
    model.add_layer(10, 'Sigmoid')
    train_data = MNISTLoader(args.data, 'train', args.classes, args.batch_size)
    valid_data = MNISTLoader(args.data, 't10k', args.classes, args.batch_size)
    train_loss = []
    valid_loss = []
    for epoch in range(args.epochs):
        epoch_train_loss = train(epoch, model, train_data)
        train_loss.append(epoch_train_loss)
        epoch_valid_loss = valid(epoch, model, valid_data)
        valid_loss.append(epoch_valid_loss)
    loss_pic = LossPicBoard(train_loss, valid_loss, "loss", "epoch", "loss")
    loss_pic.draw_plot()
    accuracy_pic = PicBoard(valid_accuracy, "accuracy", "epoch", "accuracy")
    accuracy_pic.draw_plot()
    

def train(epoch, model, train_data):
    batch_num = train_data.batch_num
    batch_loss = AvgrageMeter()
    for batch in range(batch_num):
        batch_input = train_data.data_queue[batch].reshape((-1, model.input_size))
        batch_label = train_data.label_queue[batch].reshape((-1, args.output_size))
        loss = model.backpropagation(batch_input, batch_label)[0]
        model.update_parameters(args.learning_rate)
        batch_loss.update(loss)
    return batch_loss.avg


def valid(epoch, model, valid_data):
    batch_num = valid_data.batch_num
    batch_loss = AvgrageMeter()
    batch_accuracy = AvgrageMeter()
    for batch in range(batch_num):
        batch_input = valid_data.data_queue[batch].reshape((-1, model.input_size))
        batch_label = valid_data.label_queue[batch].reshape((-1, args.output_size))
        batch_output = model.forward(batch_input)
        loss = model.get_loss(batch_output, batch_label)[0]
        accuracy = get_accuracy(batch_output, batch_label)
        batch_accuracy.update(accuracy)
        batch_loss.update(loss)
    valid_accuracy.append(batch_accuracy.avg)
    return batch_loss.avg
    

if __name__ == '__main__':
    main()
