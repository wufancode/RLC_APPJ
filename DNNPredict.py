import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import tensorflow as tf

from tensorflow import keras
from keras.layers import Dense, Flatten
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from keras import Model
from keras import layers
import keras_tuner as kt
from scipy.stats import pearsonr, spearmanr
from minepy import MINE

from sklearn.ensemble import RandomForestRegressor
# %%
sel = 2  # 1-NO, 2-O
if sel == 1:
    data_path = './Data_NO.csv'
    mid_path = 'DNN_model/Train_NO/'
    model_path = 'DNN_model/DNN_Model_NO_2'
    label_name = 'NO'
    para = [420, 200, 90]
    lr_rate = 0.001
elif sel == 2:
    data_path = 'Data_O.csv'
    mid_path = 'DNN_model/Train_O/'
    model_path = 'DNN_model/DNN_Model_O_2'
    label_name = 'O'
    para = [256, 100, 80]
    lr_rate = 0.001
print(tf.__version__)
column_names = ['Voltage', 'Flow', 'Distan', 'O2', 'H2o', 'Frequency']
raw_dataset = pd.read_csv(data_path,
                          na_values="?", comment='\t',
                          sep=",", skipinitialspace=True)
dataset = raw_dataset.copy()
# plt.hist(dataset['NO'], bins=25)
# plt.xlabel("NO")
# _ = plt.ylabel("Count")
# plt.show()


def norm(x, x_min, x_max):
    return (x - x_min) / (x_max-x_min)


# dataset = norm(dataset)
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)
train_stats = dataset.describe()
# train_stats.pop('NO')
train_stats = train_stats.transpose()
print(train_stats)
train_labels = train_dataset.pop(label_name)
test_labels = test_dataset.pop(label_name)
y_max = max(np.max(train_labels), np.max(test_labels))
y_min = min(np.min(train_labels), np.min(test_labels))
m = MINE()


def de_norm(x):
    return y_min+x*(y_max-y_min)


train_dataset.to_csv(mid_path + "train_data.csv", sep=',', header=False, index=False)
test_dataset.to_csv(mid_path + "test_data.csv", sep=',', header=False, index=False)
train_labels.to_csv(mid_path + "train_label.csv", sep=',', header=False, index=False)
test_labels.to_csv(mid_path + "test_label.csv", sep=',', header=False, index=False)

# normed_train_data = train_dataset
# normed_test_data = test_dataset
x_max = np.max(train_dataset)
x_min = np.min(train_dataset)
normed_train_data = norm(train_dataset, x_min, x_max)
normed_test_data = norm(test_dataset, x_min, x_max)
# for i in column_names:
#     print("PCC:", i, pearsonr(normed_train_data[i], train_labels))
# for i in column_names:
#     print("SPEARMAN:", i, spearmanr(normed_train_data[i], train_labels))
#
#
# forest = RandomForestRegressor(n_estimators=10000, n_jobs=-1, random_state=0)
# forest.fit(normed_train_data, train_labels)
# importances = forest.feature_importances_
# indices = np.argsort(importances)[::-1]
# for f in range(normed_train_data.shape[1]):
#     #给予10000颗决策树平均不纯度衰减的计算来评估特征重要性
#     print("%2d) %-*s %f" % (f+1, 30, column_names[f], importances[indices[f]]))
#
# train_stats = normed_train_data.describe()
# train_stats = train_stats.transpose()
# print(train_stats)


class SyslModel(tf.keras.Model):
    def __init__(self):
        super(SyslModel, self).__init__()
        self.d1 = Dense(para[0], activation='relu')
        self.d2 = Dense(para[1], activation='relu')
        self.d3 = Dense(para[2], activation='relu')
        self.d4 = Dense(1)

    def call(self, inputs):
        x = self.d1(inputs)
        x = self.d2(x)
        x = self.d3(x)
        return self.d4(x)

    # def compute_metrics(self, x, y, y_pred, sample_weight):
    #     # This super call updates `self.compiled_metrics` and returns results
    #     # for all metrics listed in `self.metrics`.
    #     metric_results = super(SyslModel, self).compute_metrics(
    #         x, y, y_pred, sample_weight)
    #
    #     # Note that `self.custom_metric` is not listed in `self.metrics`.
    #     self.custom_metric.update_state(x, y, y_pred, sample_weight)
    #     metric_results['R2'] = r2_score(y, y_pred)
    #     return metric_results
    #
    # @property
    # def metrics(self):
    #     return [self.loss_tracker]


def build_model():
    model_stance = SyslModel()

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_rate)
    model_stance.compile(loss='mse',
                         optimizer=optimizer,
                         metrics=['mae', 'mse'])
    return model_stance


# 通过为每个完成的时期打印一个点来显示训练进度
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print('')
        print('.', end='')


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist['mae'] = hist['mae']*(y_max-y_min)
    hist['val_mae'] = hist['val_mae'] * (y_max - y_min)
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [NO]')
    plt.plot(hist['epoch'], hist['mae'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mae'],
             label='Val Error')
    plt.legend()
    hist['mse'] = hist['mse'] * (y_max - y_min) ** 2
    hist['val_mse'] = hist['val_mse'] * (y_max - y_min) ** 2
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$NO^2$]')
    plt.plot(hist['epoch'], hist['mse'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'],
             label='Val Error')
    plt.legend()
    hist.to_csv(mid_path+"hist.csv", sep=',')
    # plt.show()


# %%
MyModel = build_model()
EPOCHS = 5000
# patience 值用来检查改进 epochs 的数量
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
history = MyModel.fit(normed_train_data, norm(train_labels, y_min, y_max), epochs=EPOCHS,
                      validation_split=0.1, verbose=0, callbacks=[early_stop, PrintDot()])

plot_history(history)

MyModel.save(model_path)
loss, mae, mse = MyModel.evaluate(normed_test_data, norm(test_labels, y_min, y_max), verbose=2)
test_predictions = de_norm(MyModel.predict(normed_test_data).flatten())

print("Testing set mse:", mean_squared_error(test_labels, test_predictions)*(y_max-y_min) ** 2,
      "Testing set mae:", mean_absolute_error(test_labels*(y_max-y_min), test_predictions*(y_max-y_min)),
      "R2: ", r2_score(test_labels, test_predictions))
predict_labels = pd.DataFrame(np.array([test_predictions, test_labels]).transpose())
predict_labels.to_csv(mid_path+"predict_label.csv", sep=',', header=False, index=False)
plt.figure()
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.axis('equal')
plt.axis('square')
plt.xlim([0, plt.xlim()[1]])
plt.ylim([0, plt.ylim()[1]])
_ = plt.plot([-800, 800], [-800, 800])
# plt.show()
# plt.figure()
# error = test_predictions - test_labels
# plt.hist(error, bins=25)
# plt.xlabel("Prediction Error")
# _ = plt.ylabel("Count")
plt.show()
