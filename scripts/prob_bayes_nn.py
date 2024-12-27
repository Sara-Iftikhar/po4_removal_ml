"""
====================================================
6. Probabalistic Bayesian neural network
====================================================
"""

import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from ai4water.utils import TrainTestSplit
from ai4water.postprocessing import ProcessPredictions

from utils import  negative_loglikelihood

from utils import make_data
from utils import BayesianNN
from utils import set_rcParams
from utils import print_metrics
from utils import maybe_save_prediction
from utils import residual_plot, version_info, SAVE, regression_plot

# %%

set_rcParams()

# %%

for lib, ver in version_info().items():
    print(lib, ver)

# %%

seed = 313
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

# %%

data, _, encoders= make_data(encoding='le')
X_train, X_test, y_train, y_test = TrainTestSplit(seed=142).\
    random_split_by_groups(x=data.iloc[:,0:-1], y=data.iloc[:, -1],
    groups=data['Adsorbent'])
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# %%

hidden_units = [31, 31]
learning_rate = 0.003270682279665239
batch_size = 40
activation = 'relu'
num_epochs = 500
alpha = 0.05

# %%

input_features = X_train.columns.tolist()

# %%

model = BayesianNN(
    model = {"layers": dict(
        hidden_units=hidden_units,
        train_size =len(y_train),
        activation=activation,
        uncertainty_type='both'
    )},
    category="DL",
    lr=learning_rate,
    batch_size=batch_size,
    epochs=num_epochs,
    input_features=input_features,
    #loss=negative_loglikelihood,
    #prefix="/mnt/datawaha/hyex/atr/playground/results/abcabc/",
)

# %%

model.update_weights('../models/ProbBayesNN/weights.hdf5')

# model.fit(X_train, y_train, validation_data=(X_test, y_test),
#           verbose=1)

# %%
# Training

tr_prediction_distribution = model._model(X_train)
tr_prediction_mean = tr_prediction_distribution.mean().numpy()
tr_prediction_stdv = tr_prediction_distribution.stddev().numpy()

# The 95% CI is computed as mean ± (1.96 * stdv)
upper = (tr_prediction_mean.tolist() + (1.96 * tr_prediction_stdv)).tolist()
lower = (tr_prediction_mean.tolist() - (1.96 * tr_prediction_stdv)).tolist()
tr_prediction_stdv = tr_prediction_stdv.tolist()

# %%

print_metrics(y_train, np.array(tr_prediction_mean), 'Training')

# %%

tr_df = pd.DataFrame()
tr_df['prediction'] = tr_prediction_mean.reshape(-1,)
tr_df['upper'] = np.array([val[0] for val in upper])
tr_df['lower'] = np.array([val[0] for val in lower])

# %%
# Test

test_prediction_distribution = model._model(X_test)
test_prediction_mean = test_prediction_distribution.mean().numpy()
test_prediction_stdv = test_prediction_distribution.stddev().numpy()

# The 95% CI is computed as mean ± (1.96 * stdv)
upper = (test_prediction_mean.tolist() + (1.96 * test_prediction_stdv))\
    .tolist()
lower = (test_prediction_mean.tolist() - (1.96 * test_prediction_stdv))\
    .tolist()
test_prediction_stdv = test_prediction_stdv.tolist()

# %%

print_metrics(y_test, np.array(test_prediction_mean), 'Test')

# %%

ax = residual_plot(y_train,
              train_prediction=tr_prediction_mean,
              test_true=y_test,
              test_prediction=np.array(test_prediction_mean),
                   label='qe'
              )
if SAVE:
    plt.savefig("../manuscript/figures/residue_probbayesnn", dpi=600, bbox_inches="tight")
plt.show()

# %%
ax = regression_plot(
    y_train,
    tr_prediction_mean,
    y_test,
    test_prediction_mean,
)
ax.set_xlim(-20, ax.get_xlim()[1])
ax.set_ylim(-20, ax.get_ylim()[1])
if SAVE:
    plt.savefig("../manuscript/figures/regression_probbayesnn", dpi=600, bbox_inches="tight")
plt.show()

# %%

pp = ProcessPredictions('regression', 1, show=False)
output = pp.edf_plot(y_train, tr_prediction_mean,
                     label=("Absolute Error (Training)", "Prediction (Training)"))
output[1].legend(loc=(0.5, 0.18), frameon=False)
output = pp.edf_plot(y_test, test_prediction_mean, marker='*', ax=output[0], pred_axes=output[1],
                     label=("Absolute Error (Test)", "Prediction (Test)"))
output[1].legend(loc=(0.5, 0.18), frameon=False)
output[0].set_xlabel('Absolute Error', fontsize=12)
output[1].set_xlabel('Prediction', fontsize=12)
output[0].set_ylabel('Commulative Probability', fontsize=12)
if SAVE:
    plt.savefig("../manuscript/figures/probbayes_edf", dpi=600, bbox_inches="tight")
plt.show()

# %%

maybe_save_prediction(y_train, tr_prediction_mean, 'probbayesnn_train')

# %%

maybe_save_prediction(y_test, test_prediction_mean, 'probbayesnn_test')
