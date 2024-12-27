"""
=====================
Monte Carlo Dropout
=====================
Monte Carlo (MC) dropout is an alternative to Variational Inference to build and
train bayesian neural networks. Normally the dropout is used in the NN during
training which helps avoid overfitting and increases generalization. This dropout
is turned off during test/inference/prediction. The idea of MC dropout is simple.
We turn on the dropout during test, and the NN becomes probabilistic/bayesian because now
every time we call ``predict``, the output is different. By calling predict
for sufficient number of times n (say n=100), we can get the mean and standard
deviation of the distribution. This distribution is also the distribution of weights of NN.
This was first introduced by
`Gal et al., 2016 <https://proceedings.mlr.press/v48/gal16.html?trk=public_post_comment-text>`_

"""

import os
import random

import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Dense, Dropout

from ai4water.utils import TrainTestSplit
from ai4water.utils import LossCurve
from ai4water.utils.utils import find_best_weight
from ai4water.utils.utils import dateandtime_now
from ai4water.postprocessing import ProcessPredictions

from easy_mpl import plot

from utils import SAVE
from utils import make_data
from utils import set_rcParams
from utils import residual_plot
from utils import print_metrics
from utils import regression_plot
from utils import get_version_info
from utils import maybe_save_prediction

# %%

seed = 313
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

# %%

for lib, ver in get_version_info().items():
    print(lib, ver)

# %%

set_rcParams()

# %%

data, _, encoders= make_data(encoding='le')
X_train, X_test, y_train, y_test = TrainTestSplit(seed=142).\
    random_split_by_groups(x=data.iloc[:,0:-1], y=data.iloc[:, -1],
    groups=data['Adsorbent'])

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# %%

input_features = X_train.columns.tolist()

# %%

inp = Input(shape=(X_train.shape[1],))
d1 = Dense(32)(inp)
drop1 = Dropout(0.3)(d1)
d2 = Dense(32)(drop1)
drop2 = Dropout(0.3)(d2)
d3 = Dense(32, activation="relu")(drop2)
drop3 = Dropout(0.3)(d3)
out = Dense(1)(drop3)

model = Model(inp, out)
model.compile(loss="mse", optimizer=Adam(lr=0.0001))

# %%

model.summary()

# %%

callbacks = []
fname = "{val_loss:.5f}.hdf5"
w_path = os.path.join(os.getcwd(), "results", dateandtime_now())
if not os.path.exists(w_path):
    os.makedirs(w_path)
callbacks.append(ModelCheckpoint(
    filepath=w_path + f"{os.sep}weights_" + "{epoch:03d}_" + fname,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True))

callbacks.append(EarlyStopping(
    monitor='val_loss', min_delta=0.1,
    patience=100, verbose=0, mode='auto'
))

# %%

# h = model.fit(
#     X_train.values, y_train.values,
#     validation_data=(X_test, y_test),
#     epochs=500,
#     callbacks=callbacks
# )
#
# best_weights = find_best_weight(w_path)
# if best_weights is not None:
#     weight_file_path = os.path.join(w_path, best_weights)
#     model.load_weights(weight_file_path)
#     print(f'updated weights using {best_weights}')

# l = LossCurve(save = False)
#
# l.plot(h.history)

model.load_weights('../models/MCDropout/weights.hdf5')

# %%

for i in range(5):
    print(model.predict(X_train.values[0].reshape(1, -1),
                        verbose=False))

# %%

for i in range(5):
    print(model(X_train.values[0].reshape(1, -1),
                training=True))

# %%

n = 100
train_pred = np.full(shape=(n, X_train.shape[0]), fill_value=np.nan)
for i in range(n):
    train_pred[i] = model(X_train.values, training=True).numpy().reshape(-1,)

print(train_pred.shape)

# %%

tr_std = np.std(train_pred, axis=0)
tr_mean = np.mean(train_pred, axis=0)

# %%

plot(tr_mean, '.', label="Prediction Mean", show=False)
plot(y_train.values, '.', label="True")

# %%

print_metrics(y_train, tr_mean, 'Train')

# %%

test_pred = np.full(shape=(n, X_test.shape[0]), fill_value=np.nan)
for i in range(n):
    test_pred[i] = model(X_test.values, training=True).numpy().reshape(-1,)

print(test_pred.shape)

# %%

test_std = np.std(test_pred, axis=0)
test_mean = np.mean(test_pred, axis=0)

# %%

print_metrics(y_test, test_mean, 'Test')

# %%

maybe_save_prediction(y_train.values, tr_mean, 'mc_train')

# %%

maybe_save_prediction(y_test.values, test_mean, 'mc_test')

# %%

ax = residual_plot(y_train, tr_mean, y_test, test_mean, label="qe")
ax[0].set_ylim(-180, ax[0].get_ylim()[1])
if SAVE:
    plt.savefig("../manuscript/figures/residue_mc",
                dpi=600, bbox_inches="tight")
plt.show()

# %%

ax = regression_plot(y_train, tr_mean, y_test, test_mean)
ax.set_xlim(-20, ax.get_xlim()[1])
ax.set_ylim(-20, ax.get_ylim()[1])
if SAVE:
    plt.savefig("../manuscript/figures/reg_mc", dpi=600, bbox_inches="tight")
plt.show()

# %%

pp = ProcessPredictions('regression', 1, show=False)
output = pp.edf_plot(y_train, tr_mean,
                     label=("Absolute Error (Training)", "Prediction (Training)"))
output[1].legend(loc=(0.5, 0.18), frameon=False)
output = pp.edf_plot(y_test, test_mean, marker='*', ax=output[0], pred_axes=output[1],
                     label=("Absolute Error (Test)", "Prediction (Test)"))
output[1].legend(loc=(0.57, 0.18), frameon=False)
output[0].set_xlabel('Absolute Error', fontsize=12)
output[1].set_xlabel('Prediction', fontsize=12)
output[0].set_ylabel('Commulative Probability', fontsize=12)
if SAVE:
    plt.savefig("../manuscript/figures/mc_edf", dpi=600, bbox_inches="tight")
plt.show()
