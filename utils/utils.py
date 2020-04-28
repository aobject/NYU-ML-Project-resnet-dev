import numpy as np
from sklearn.datasets import load_digits # The MNIST data set is in scikit learn data set
from sklearn.preprocessing import StandardScaler  # It is important in neural networks to scale the date
from sklearn.model_selection import train_test_split  # The standard - train/test to prevent overfitting and choose hyperparameters
from utils.cnn import CNN
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource, HoverTool, NumeralTickFormatter
from bokeh.io import output_notebook
# output_notebook()

def get_data():
    digits = load_digits()
    X = digits.data
    y = digits.target

    X_scale = StandardScaler()
    X = X_scale.fit_transform(digits.data)

    # Split the data into training and test set.  60% training and %40 test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    # Reshape X_train and X_test to be (-1, 8, 8)
    X_train = X_train.reshape((-1, 8, 8))
    X_test = X_test.reshape((-1, 8, 8))
    return (X_train, X_test, y_train, y_test)

def shuffle(X, y):
    new_order = np.random.permutation(len(y))
    X = X[new_order]
    y = y[new_order]
    return X, y

def forward(x, y, model):
    result = model.kernels.forward(x)
    # result = model.pool.forward(result)
    yhat = model.sigmoid.forward(result)
    loss, accuracy = model.sigmoid.loss(y, yhat)
    return yhat, loss, accuracy


def train(x, y, model, alpha=0.1):
    # forward
    _, loss, accuracy = forward(x, y, model)

    # back
    delta = model.sigmoid.back(alpha)
    # delta = model.pool.back(delta)
    model.kernels.back(delta, alpha)

    return loss, accuracy


def plot_results(loss_train_seq, loss_test_seq, acc_train_seq, acc_test_seq):
    print('The test set prediction accuracy is {}%'.format(acc_test_seq[-1] * 100))
    source = ColumnDataSource(data={
        'epoch'            : range(1, len(loss_test_seq) + 1),
        'train_loss'    : loss_train_seq,
        'test_loss'        :  loss_test_seq,
    })

    p = figure(title='Loss', plot_width=400, plot_height=400)

    p.line(x='epoch', y='train_loss', color='#329fe3', line_alpha=0.8 , line_width=2, legend_label="Train", source=source)
    p.line(x='epoch', y='test_loss', color='#e33270', line_alpha=0.8, line_width=2, legend_label="Test", source=source)
    p.legend.location = "top_right"
    p.xaxis.axis_label = 'Epochs'
    p.yaxis.axis_label = 'Loss'

    p.add_tools(HoverTool(
        tooltips=[
                  ('Epochs', '@epoch{int}'),
                  ('Training Loss', '@train_loss{0.000 a}'),
                  ('Test Loss', '@test_loss{0.000 a}'),
        ],

        mode='mouse'
    ))

    show(p)

    source = ColumnDataSource(data={
        'epoch'            : range(1, len(acc_train_seq) + 1),
        'train_acc'    : np.array(acc_train_seq),
        'test_acc'        :  np.array(acc_test_seq),
    })

    p = figure(title='Accuracy', plot_width=400, plot_height=400)

    p.line(x='epoch', y='train_acc', color='#329fe3', line_alpha=0.8 , line_width=2, legend_label="Train", source=source)
    p.line(x='epoch', y='test_acc', color='#e33270', line_alpha=0.8, line_width=2, legend_label="Test", source=source)
    p.legend.location = "bottom_right"
    p.xaxis.axis_label = 'Epochs'
    p.yaxis.axis_label = 'Accuracy'
    p.yaxis.formatter = NumeralTickFormatter(format='0 %')

    p.add_tools(HoverTool(
        tooltips=[
                  ('Epochs', '@epoch{int}'),
                  ('Training Accuracy', '@{train_acc}{%0.2f}'),
                  ('Test Accuracy', '@{test_acc}{%0.2f}'),
        ],

        mode='mouse'
    ))

    show(p)




