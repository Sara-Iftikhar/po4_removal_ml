"""
===============================
5. Probabalistic Neural Network
===============================
The output of Neural Network is a probability distribution instead of
a single value. This algorithm captures uncertainty inherent in the data
by predicting whole distribution for the target. This type of uncertainty
is also called aleotoric uncertainty.
"""
import os
import random

import tensorflow as tf
import matplotlib.pyplot as plt

from ai4water.utils.utils import get_version_info
from ai4water.utils.utils import TrainTestSplit, reset_seed
from ai4water.postprocessing import ProcessPredictions

from utils import SAVE
from utils import set_rcParams
from utils import negative_loglikelihood
from utils import print_metrics
from utils import maybe_save_prediction
from utils import residual_plot, regression_plot
from utils import  make_data, BayesianNN, BayesModel

# %%

# seed = 313
# np.random.seed(seed)
# random.seed(seed)
# tf.random.set_seed(seed)

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
# hyperparameters
# ----------------

hidden_units = [19, 19]
learning_rate = 0.002850512
batch_size = 8
activation = "sigmoid"
num_epochs = 700


# %%
# model building
# ----------------

model = BayesianNN(
    model = {"layers": dict(
        hidden_units=hidden_units,
        train_size =len(y_train),
        activation=activation,
        uncertainty_type='aleotoric'
    )},
    batch_size=batch_size,
    lr=learning_rate,
    epochs=num_epochs,
    input_features=input_features,
    optimizer="RMSprop",  # RMSprop quite better than Adam
    loss = negative_loglikelihood,  # todo, why both nll and mse are working?
    #prefix="/mnt/datawaha/hyex/atr/playground/results/abcabc/"
)

reset_seed(142, os=os, tf=tf, random=random)

# %%
# model training
# ----------------

model.update_weights('../models/ProbNN/weights.hdf5')

# model.fit(X_train, y_train, validation_data=(X_test, y_test),
#           verbose=0)

# %%

for i in range(5):
    print(model.predict(X_train[0:2], verbose=False).reshape(-1,))

# %%
train_dist = model._model(X_train)

print(type(train_dist))

# %%

train_mean = train_dist.mean().numpy().reshape(-1,)
train_std = train_dist.stddev().numpy().reshape(-1, )

# %%

maybe_save_prediction(y_train.values, train_mean, 'probnn_train')

# %%

print_metrics(y_train.values, train_mean, "Train")

# %%

test_dist = model._model(X_test)

# %%

test_mean = test_dist.mean().numpy().reshape(-1,)
test_std = test_dist.stddev().numpy().reshape(-1, )

# %%

maybe_save_prediction(y_test.values, test_mean, 'probnn_test')

# %%

print_metrics(y_test.values, test_mean, "Test")

# %%

ax = regression_plot(y_train, train_mean, y_test, test_mean)
ax.set_xlim(-20, ax.get_xlim()[1])
ax.set_ylim(-20, ax.get_ylim()[1])
if SAVE:
    plt.savefig("../manuscript/figures/reg_prob", dpi=600, bbox_inches="tight")
plt.show()

# %%

ax = residual_plot(y_train, train_mean, y_test, test_mean, label='qe')
if SAVE:
    plt.savefig("../manuscript/figures/residue_prob",
                dpi=600, bbox_inches="tight")
plt.show()

# %%

pp = ProcessPredictions('regression', 1, show=False)
output = pp.edf_plot(y_train, train_mean,
                     label=("Absolute Error (Training)", "Prediction (Training)"))
output[1].legend(loc=(0.5, 0.18), frameon=False)
output = pp.edf_plot(y_test, test_mean, marker='*', ax=output[0], pred_axes=output[1],
                     label=("Absolute Error (Test)", "Prediction (Test)"))
output[1].legend(loc=(0.57, 0.18), frameon=False)
output[0].set_xlabel('Absolute Error', fontsize=12)
output[1].set_xlabel('Prediction', fontsize=12)
output[0].set_ylabel('Commulative Probability', fontsize=12)
if SAVE:
    plt.savefig("../manuscript/figures/prob_edf", dpi=600, bbox_inches="tight")
plt.show()
