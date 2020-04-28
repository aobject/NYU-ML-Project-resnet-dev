# import numpy as np
from utils.cnn import CNN
from utils.utils import forward, train, shuffle, get_data, plot_results
import time


if __name__ == '__main__':
    kernel_layers = 64
    pool_size = 2
    epochs = 3000
    out_nodes = 10
    alpha = 0.4
    seed = 33

    X_train, X_test, y_train, y_test = get_data()

    model = CNN(kernel_layers, pool_size, out_nodes, seed)

    loss_train_seq = []
    loss_test_seq = []
    acc_train_seq = []
    acc_test_seq = []
    train_examples = len(y_train)
    test_examples = len(y_test)

    timestr = time.strftime("%Y%m%d-%H%M-results.txt")
    file = open(timestr, 'w')

    file.write('=' * 30)
    file.write('==== Hyperparameters ====')
    file.write('='*30)
    file.write('Learning Rate: {}'.format(alpha))
    file.write('Kernel Layers: {}'.format(kernel_layers))
    # file.write('Pool Size: {}'.format(pool_size))
    file.write('Epochs: {}'.format(epochs))
    file.write('Seed: {}'.format(seed))
    file.write('=' * 30 + '\n')

    print('=' * 30)
    print('==== Hyperparameters ====')
    print('='*30)
    print('Learning Rate: {}'.format(alpha))
    print('Kernel Layers: {}'.format(kernel_layers))
    # print('Pool Size: {}'.format(pool_size))
    print('Epochs: {}'.format(epochs))
    print('Seed: {}'.format(seed))
    print('=' * 30 + '\n')
    for epoch in range(epochs):
        print('=' * 30)
        print('Starting epoch {} of {}'.format(epoch, epochs))
        X_train, y_train = shuffle(X_train, y_train)

        loss_train = 0.0
        loss_test = 0.0
        acc_train = 0.0
        acc_test = 0.0

        for i, (image, label) in enumerate(zip(X_train, y_train)):
            loss, acc = train(image, label, model, alpha)
            loss_train += loss
            acc_train += acc


        for i, (image, label) in enumerate(zip(X_test, y_test)):
            _, loss, acc = forward(image, label, model)
            loss_test += loss
            acc_test += acc


        loss_train_seq.append(loss_train / train_examples)
        loss_test_seq.append(loss_test / test_examples)
        acc_train_seq.append(acc_train / train_examples)
        acc_test_seq.append(acc_test / test_examples)

    plot_results(loss_train_seq, loss_test_seq, acc_train_seq, acc_test_seq)

    print('Test Loss: {}\nTest Accuracy: {} %'.format(loss_test_seq[-1], acc_test_seq[-1] * 100))
    print('Train Loss: {}\nTrain Accuracy: {} %\n\n'.format(loss_train_seq[-1], acc_train_seq[-1] * 100))

    file.write('Test Loss: {}\nTest Accuracy: {} %'.format(loss_test_seq[-1], acc_test_seq[-1] * 100))
    file.write('Train Loss: {}\nTrain Accuracy: {} %\n\n'.format(loss_train_seq[-1], acc_train_seq[-1] * 100))

    file.close()
    print('Done!')
