import torch.nn as nn
import torch.optim as optim
import torch
from net import ConvNet
import numpy as np
from bokeh.plotting import figure
from bokeh.io import show
from bokeh.models import LinearAxis, Range1d
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from mnist_dataset import MNISTDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b',
                        help='Batch size',
                        default=512,
                        type=int)
    parser.add_argument('-e',
                        help='Epochs number',
                        default=1,
                        type=int)
    parser.add_argument('-m',
                        help='Saved model',
                        default=r'C:\Users\ACER\Desktop\project\mnist_classification\Pytorch_project\conv_net_model.pth')
    parser.add_argument('-trd',
                        help='Path train',
                        default=r'C:\Users\ACER\Desktop\project\mnist_classification\Pytorch_project\train')
    parser.add_argument('-ted',
                        help='Path test',
                        default=r'C:\Users\ACER\Desktop\project\mnist_classification\Pytorch_project\test')
    parser.add_argument('-o',
                        help='Path to save model',
                        default=r'C:\Users\ACER\Desktop\project\mnist_classification\Pytorch_project\conv_net_model.pth')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    trans = transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    # Используем собстевенный датасет
    train_dataset = MNISTDataset(root=args.trd, train=True, transform=trans)
    test_dataset = MNISTDataset(root=args.ted, train=True, transform=trans)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.b, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.b, shuffle=False)
    best_acc = 0

    model = ConvNet()
    if args.m is not '':
        model.load_state_dict(torch.load(args.m))
    # Добавить загрузку предобученной модели
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_list = []
    acc_list = []
    t_acc_list = []

    for epoch in range(args.e):
        train_pbar = tqdm(train_loader)
        for i, (images, labels) in enumerate(train_pbar):
            # Прямой запуск
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())

            # Обратное распространение и оптимизатор
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Отслеживание точности
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            del outputs, predicted
            train_pbar.set_description('Loss: {0:.2f} Acc: {1:.2f}'.format(
                loss.item(),
                (correct / total) * 100))
       # Необходимо добавить визуализацию Train Loss and Acc(сохранять в массив)
        acc_list.append(correct / total)

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for (images, labels) in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                del outputs, predicted
            print((' Test Accuracy: {0:.2f}'.format((correct / total) * 100)))
            t_acc_list.append(correct / total)

            if (correct / total) > best_acc:
                # Сохраняем модель и строим график
                torch.save(model.state_dict(), args.o)
    # Визуализировать результаты в графиках
    p = figure(y_axis_label='Loss', width=850, y_range=(0, 1), title='PyTorch ConvNet results')
    p.extra_y_ranges = {'Accuracy': Range1d(start=0, end=100)}
    p.add_layout(LinearAxis(y_range_name='Accuracy', axis_label='Accuracy (%)'), 'right')
    p.line(np.arange(len(loss_list)), loss_list)
    p.line(np.arange(len(loss_list)), np.array(acc_list), y_range_name='Accuracy', color='red')
    show(p)
if __name__ == '__main__':
    main()


