import os
from keras.layers import Input, Dense, Conv1D, Flatten, MaxPooling1D, Dropout
from keras.models import Model, Sequential
from keras.regularizers import l1_l2
import h5py
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.utils.multiclass import unique_labels
import uuid
import random
import json

data_path = '/home/igamboa/thesis/data'
experiments_path = '/home/igamboa/experiments_1.json'
output_path = '/home/igamboa/thesis/results'

def network(input_window_size=60, filter_number=32, conv_window=(3,), pooling_window=(2,), dropout_rate=[],
            activation='relu', dense_activation='softmax', optimizer='adam', loss='categorical_crossentropy', layers=1,
            l1_value=0.0001, l2_value=0.0001):
    model = Sequential()

    # Input Layer
    model.add(Conv1D(filter_number, conv_window, activation=activation, padding='same', input_shape=(input_window_size, 1),
                     activity_regularizer=l1_l2(l1=l1_value, l2=l2_value)))
    model.add(MaxPooling1D(pooling_window, padding='same'))
    model.add(Dropout(dropout_rate[0]))


    # Hidden Layers
    current_filter = 1
    filter_number_temp = filter_number
    for i in range(layers):
        filter_number_temp = filter_number_temp *  2
        model.add(Conv1D(filter_number_temp, conv_window, activation=activation, padding='same',
                         activity_regularizer=l1_l2(l1=l1_value, l2=l2_value)))
        model.add(MaxPooling1D(pooling_window, padding='same'))
        model.add(Dropout(dropout_rate[current_filter]))
        current_filter = current_filter + 1

    # Output Layer
    model.add(Flatten())
    model.add(Dense(2, activation=dense_activation))

    model.compile(optimizer=optimizer[0], loss=loss, metrics=['accuracy', 'categorical_accuracy'])

    return model

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues,
                         name='1'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

def training(path, input_window_size=60, scaler=MinMaxScaler(), normalizer=False, filter_number=32, conv_window=(3,), pooling_window=(2,), dropout_rate=[0.2, 0.5],
            activation='relu', dense_activation='softmax', optimizer=('adam', 0), loss='categorical_crossentropy', layers=1,
            l1_value=0.0001, l2_value=0.0001, name='', epochs=50, batch_size=100):

    saved_args = locals()
    print(saved_args)

    h5f = h5py.File(os.path.join(path, "window_output/{}_{}_{}.h5".format(input_window_size, scaler, normalizer)), 'r')
    x_dataset = h5f['x_dataset'][:]
    y_dataset = h5f['y_dataset'][:]

    model = network(input_window_size, filter_number, conv_window, pooling_window, dropout_rate,
                  activation, dense_activation, optimizer, loss, layers, l1_value, l2_value)

    x_train, x_test, y_train, y_test = train_test_split(x_dataset, y_dataset, test_size=0.1)

    history = model.fit(x_train, y_train, batch_size, epochs, validation_data=(x_test, y_test))

    plt.plot(history.history['acc'], label='train_acc')
    plt.plot(history.history['val_acc'], label='test_acc')
    plt.show()
    plt.savefig(os.path.join(output_path,"{}.png".format(name)))
    model.save(os.path.join(output_path,"{}.h5".format(name)))

    classes = [1 if i[1] > 0.5 else 0 for i in model.predict(x_test)]

    true_classes = [np.argmax(i) for i in y_test]

    print("accuracy_score: ", accuracy_score(true_classes, classes))
    print("precision_score: ", precision_score(true_classes, classes))
    print("recall_score: ", recall_score(true_classes, classes))
    print("f1_score: ", f1_score(true_classes, classes))

    plot_confusion_matrix(true_classes, classes, classes=['not_p', 'p'],
                        title='Confusion matrix, without normalization')

def hinted_tuple_hook(obj):
    if '__tuple__' in obj:
        return tuple(obj['items'])
    else:
        return obj

with open(experiments_path) as experiments_file:
    experiments = json.loads(experiments_file.read(), object_hook=hinted_tuple_hook)

for experiment in experiments:
    experiment_params = {
        'input_window_size': experiment[0],
        'scaler': experiment[1],
        'normalizer': experiment[2],
        'filter_number': experiment[3],
        'conv_window': experiment[4],
        'pooling_window': experiment[5],
        'dropout_rate': experiment[6],
        'activation': experiment[7],
        'dense_activation': experiment[8],
        'optimizer': experiment[9],
        'loss': experiment[10],
        'layers': experiment[11],
        'l1_value': experiment[12],
        'l2_value': experiment[13],
        'epochs': experiment[14],
        'batch_size': experiment[15]
    }
    training(data_path, name=str(uuid.uuid1()), **experiment_params)
