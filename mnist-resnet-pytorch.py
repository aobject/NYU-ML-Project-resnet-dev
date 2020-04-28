import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_digits # The MNIST data set is in scikit learn data set
from sklearn.preprocessing import StandardScaler  # It is important in neural networks to scale the date
from sklearn.model_selection import train_test_split  # The standard - train/test to prevent overfitting and choose hyperparameters
from bokeh.plotting import figure, output_file, show


class MNISTDataset(Dataset):
    def __init__(self, data, label):
        self.data = data.reshape((-1,8,8,1))
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = self.data[item].transpose((2, 0, 1))
        image = torch.from_numpy(image)
        target = self.label[item]
        target = torch.from_numpy(target)
        return (image, target)


class MNIST(nn.Module):

    # Our batch shape for input x is (1, 8, 8)

    def __init__(self):
        super(MNIST, self).__init__()

        # Input channels = 1, output channels = 18
        self.conv1 = torch.nn.Conv2d(1, 18, kernel_size=3, stride=1, padding=1)
        # self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = nn.Linear(18 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):

        # Activation of the first convolution
        # Size changes from (1, 8, 8) to (18, 8, 8)
        x = F.relu(self.conv1(x))

        # x = self.pool(x)

        # Reshape data to input to the input layer of the neural net
        # Size changes from (18, 16, 16) to (1, 1152)
        x = x.view(-1, 18 * 8 * 8)

        # Computes the activation of the first fully connected layer
        # Size changes from (1, 1152) to (1, 64)
        x = torch.sigmoid(self.fc1(x))

        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)

        return x


def scorecard(output, labels):
    score = 0
    score = np.where(np.argmax(output, axis=1) == np.argmax(labels, axis=1), 1, 0)
    # for i in range(len(output)):
    #     if np.argmax(output[i,:], axis=1) == np.argmax(labels[i,:], axis=1):
    #         score += 1
    score = np.sum(score)
    return score


def trainModel(model, batch_size, num_epochs, learning_rate):
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", batch_size)
    print("epochs=", num_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 30)

    digits = load_digits()
    X = digits.data
    y = digits.target

    X_scale = StandardScaler()
    X = X_scale.fit_transform(digits.data)

    # Split the data into training and test set.  60% training and %40 test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    def convert_y_to_vect(y):
        y_vect = np.zeros((len(y), 10))
        for i in range(len(y)):
            y_vect[i, y[i]] = 1
        return y_vect

    # convert digits to vectors
    y_v_train = convert_y_to_vect(y_train)
    y_v_test = convert_y_to_vect(y_test)

    train_dataset = MNISTDataset(X_train, y_v_train)
    test_dataset = MNISTDataset(X_test, y_v_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)
    mean, std = (0.5,), (0.5,)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    model = model.double()

    loss_train_seq = []
    loss_test_seq = []
    acc_train_seq = []
    acc_test_seq = []

    for i in range(num_epochs):
        loss_train = 0.0
        loss_test = 0.0
        acc_train = 0.0
        acc_test = 0.0
        train_examples = 0
        test_examples = 0

        for j, (images, labels) in enumerate(train_loader, 0):
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            acc_train += scorecard(output.detach().numpy(), labels.detach().numpy())
            train_examples += len(labels)
            loss_train += loss.detach().numpy()
            loss.backward()
            optimizer.step()


        for images, labels in test_loader:

            with torch.no_grad():
                output = model(images)

                loss = criterion(output, labels)
                loss_test += loss.detach().numpy()
                acc_test += scorecard(output.detach().numpy(), labels.detach().numpy())
                test_examples += len(labels)
        loss_train_seq.append(loss_train / train_examples)
        loss_test_seq.append(loss_test / test_examples)
        acc_train_seq.append(acc_train / train_examples)
        acc_test_seq.append(acc_test / test_examples)
        print('Epoch {} with test loss {} and test accuracy {}\n\n'.format(i, (loss_test / test_examples),(acc_test / test_examples) ))

    p = figure(title='Loss', plot_width=400, plot_height=400)
    p.line(range(1,len(loss_train_seq)+1), loss_train_seq, color='firebrick', line_width=2, legend_label="Train")
    p.line(range(1, len(loss_test_seq) + 1), loss_test_seq, color='navy', line_width=2, legend_label="Test")
    p.legend.location = "top_right"
    show(p)

    p = figure(title='Accuracy', plot_width=400, plot_height=400)
    p.line(range(1,len(acc_train_seq)+1), acc_train_seq, color='firebrick', line_width=2, legend_label="Train")
    p.line(range(1, len(acc_test_seq) + 1), acc_test_seq, color='navy', line_width=2, legend_label="Test")
    p.legend.location = "bottom_right"
    show(p)



if __name__ == "__main__":

    batch_size = 1
    learning_rate = 0.01
    num_epochs = 10

    model = MNIST()

    trainModel(model, batch_size, num_epochs, learning_rate)

