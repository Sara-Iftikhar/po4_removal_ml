"""
========
utils
========
"""

import os
import sys
import warnings
from typing import Union, Tuple, Any

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import seaborn as sns

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from ai4water.eda import EDA
from ai4water.functional import Model
from ai4water.preprocessing import DataSet
from ai4water.utils import LossCurve
from ai4water.utils.utils import get_version_info

from SeqMetrics import RegressionMetrics

from easy_mpl.utils import to_1d_array
from easy_mpl.utils import create_subplots, make_cols_from_cmap
from easy_mpl.utils import make_clrs_from_cmap
from easy_mpl.utils import despine_axes
from easy_mpl import scatter, hist, regplot, pie
from easy_mpl import plot, bar_chart
from easy_mpl.utils import AddMarginalPlots
from mpl_toolkits.axes_grid1 import make_axes_locatable

# %%

SAVE = False

# %%

CAT_COLUMNS = ["Adsorbent", "Feedstock", "ion_type"]

# %%

LABEL_MAP = {
    'Adsorption_time (min)': 'Contact Time (min)',
    'adsorption_temp': 'Ads. Temp. (C)',
    'Ion Concentration (mM)': 'Ion Conc. (mM)',
    'Pyrolysis_time (min)': 'Pyrol. Time (min)',
    'Average pore size': 'Avg. Pore Size (nm)',
    'Heating rate (oC)': 'Heat. Rate (C)',
    'Pyrolysis_temp': 'Pyrol. Temp. (C)',
    'g/L': 'Loading',
    'loading (g)': 'Loading (g)',
    'ion_type': 'Ion Type',
    'O': 'O (%)',
    'C': 'C (%)',
    'S': 'S (%)',
    'Ca': 'Ca (%)',
    'Ash': 'Ash (%)',
    'solution pH': 'Solution pH',
    'Surface area': 'Surf. Area (m2/g)',
    'Ci_ppm': 'Ci (ppm)'
}

# %%

LINE_COLORS = ["#DB0007", "#670E36", "#e30613", "#0057B8", "#6C1D45",
          "#034694", "#1B458F", "#003399", "#FFCD00", "#003090",
          "#C8102E", "#6CABDD", "#DA291C", "#241F20", "#00A650",
          "#D71920", "#132257", "#ED2127", "#7A263A", "#FDB913"
          ]

# %%

def version_info():
    import ngboost
    ver = get_version_info()
    ver['ngboost'] = ngboost.__version__

    try:
        import arfs
        ver['arfs'] = arfs.__version__
    except ModuleNotFoundError:
        pass
    return ver

# %%
def _ohe_column(df:pd.DataFrame, col_name:str)->tuple:
    # function for OHE
    assert isinstance(col_name, str)

    # setting sparse to True will return a scipy.sparse.csr.csr_matrix
    # not a numpy array
    encoder = OneHotEncoder(sparse=False)
    ohe_cat = encoder.fit_transform(df[col_name].values.reshape(-1, 1))
    cols_added = [f"{col_name}_{i}" for i in range(ohe_cat.shape[-1])]

    df[cols_added] = ohe_cat

    df.pop(col_name)

    return df, cols_added, encoder

def le_column(df:pd.DataFrame, col_name)->tuple:
    """label encode a column in dataframe"""
    encoder = LabelEncoder()
    df[col_name] = encoder.fit_transform(df[col_name])
    return df, encoder

# %%

def _load_data(
        input_features:list=None,
)->Tuple[pd.DataFrame, pd.DataFrame, list]:

    default_input_features = ['Adsorbent', 'Feedstock', 'Pyrolysis_temp',
                              'Heating rate (oC)', 'Pyrolysis_time (min)',
                              'C', 'O', 'Surface area', 'Adsorption_time (min)',
                              'Ci_ppm', 'solution pH', 'rpm', 'Volume (L)',
                              'loading (g)', 'adsorption_temp',
                              'Ion Concentration (mM)', 'ion_type']

    if input_features is None:
        input_features = default_input_features

    # read excel
    # our data is on the first sheet
    dirname = os.path.dirname(__file__)
    org_data = pd.read_excel(os.path.join(dirname, 'master_sheet_0802.xlsx'))
    data = org_data
    data.dropna()

    # removing final concentration from our data. As final concentration is used for
    # calculating the true value of target.

    #data = data.drop(columns=['Cf'])

    #removing original index of both dataframes and assigning a new index
    data = data.reset_index(drop=True)

    target = ['qe']

    if input_features is None:
        input_features = data.columns.tolist()[0:-1]
    else:
        assert isinstance(input_features, list)
        assert all([feature in data.columns for feature in input_features])

    data = data[input_features + target]
    data = data.dropna()

    data['Feedstock'] = data['Feedstock'].replace('coagulation–flocculation sludge',
                                                  'CF Sludge')

    data['Feedstock'] = data['Feedstock'].replace('bamboo (Phyllostachys pubescens)',
                                                  'bamboo (PP)')

    data = data[data['qe'] > 0.0]  # removing -ve qe

    return org_data, data, input_features

# %%

def make_data(
        input_features:list = None,
        encoding:Union[str, bool] = None,
)->Tuple[pd.DataFrame, list, dict]:

    _, data, input_features = _load_data(input_features)

    adsorbent_encoder, fs_encoder, it_encoder  = None, None, None
    if encoding=="ohe":
        # applying One Hot Encoding
        data, _, adsorbent_encoder = _ohe_column(data, 'Adsorbent')
        data, _, fs_encoder = _ohe_column(data, 'Feedstock')
        data, _, it_encoder = _ohe_column(data, 'ion_type')

    elif encoding == "le":
        # applying Label Encoding
        data, adsorbent_encoder = le_column(data, 'Adsorbent')
        data, fs_encoder = le_column(data, 'Feedstock')
        data, it_encoder = le_column(data, 'ion_type')

    # moving target to last
    target = data.pop('qe')
    data['qe'] = target

    encoders = {
        "Adsorbent": adsorbent_encoder,
        "Feedstock": fs_encoder,
        "ion_type": it_encoder
    }
    return data, input_features, encoders

# %%

def get_dataset(encoding="ohe"):

    data, input_features, encoders = make_data(encoding=encoding)

    dataset = DataSet(data=data,
                      seed=1575,
                      val_fraction=0.0,
                      split_random=True,
                      input_features=data.columns.tolist()[0:-1],
                      output_features=data.columns.tolist()[-1:],
                      )
    return dataset, input_features, encoders

# %%

def distribution_plot(ax, data, scatter_fc='#045568',
                      box_facecolor='#e6e6e6',
                      width=0.8,
                      add_hist=True,
                      add_ridge=True):

    sns.boxplot(orient='h', data=data, saturation=1, showfliers=False,
                width=width, boxprops={'zorder': 3, 'facecolor': box_facecolor}, ax=ax,
                )
    old_len_collections = len(ax.collections)

    for dots in ax.collections[old_len_collections:]:
        dots.set_offsets(dots.get_offsets() + np.array([0, 0.12]))

    ax = sns.stripplot(orient='h', x=data,
                       edgecolor="gray",
                       #linewidth=0.1,
                       alpha=0.5,
                       c=scatter_fc,
                       size=1.5,
                        ax=ax,
                       jitter=0.2)

    despine_axes(ax, keep=['bottom', 'left', 'right'])

    if add_hist or add_ridge:
        aa = AddMarginalPlots(ax=ax,
                              hist=add_hist,
                              ridge=add_ridge,
                              hist_kws=dict(bins=20, color=box_facecolor),
                              fill_kws=dict(color=box_facecolor),
                              ridge_line_kws=dict(color=scatter_fc))
        aa.divider = make_axes_locatable(ax)
        axHistx = aa.add_ax_marg_x(data.values.reshape(-1,), hist_kws=aa.HIST_KWS[0], ax=None)
        plt.setp(axHistx.get_xticklabels(),visible=False)

    return ax

# %%

def evaluate_model(true, predicted):
    metrics = RegressionMetrics(true, predicted)
    for i in ['mse', 'rmse', 'r2', 'r2_score', 'mape', 'mae']:
        print(i, getattr(metrics, i)())
    return

# %%

def run_experiment(model, loss, train_dataset, test_dataset,
                   learning_rate, num_epochs, plot_loss_curve=False):
    import tensorflow as tf
    from tensorflow import keras

    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate),
        loss=loss,
        metrics=[keras.metrics.RootMeanSquaredError()],
    )

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=50)

    print("Start training the model...")
    h = model.fit(train_dataset, epochs=num_epochs, callbacks=[callback],
                  validation_data=test_dataset, verbose=1)
    print("Model training finished.")
    _, rmse = model.evaluate(train_dataset, verbose=0)
    print(f"Train RMSE: {round(rmse, 3)}")

    print("Evaluating model performance...")
    _, rmse = model.evaluate(test_dataset, verbose=0)
    print(f"Test RMSE: {round(rmse, 3)}")

    if plot_loss_curve:
        LossCurve().plot(h.history)

    return

# %%

def create_model_inputs(FEATURE_NAMES):
    import tensorflow as tf
    from tensorflow.keras import layers

    inputs = {}
    for feature_name in FEATURE_NAMES:
        inputs[feature_name] = layers.Input(
            name=feature_name, shape=(1,), dtype=tf.float32
        )
    return inputs

# Define the prior weight distribution as Normal of mean=0 and stddev=1.
# Note that, in this example, the prior distribution is not trainable,
# as we fix its parameters.
def prior(kernel_size, bias_size, dtype=None):
    import tensorflow as tf
    from tensorflow import keras
    import tensorflow_probability as tfp
    n = kernel_size + bias_size
    prior_model = keras.Sequential(
        [
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.MultivariateNormalDiag(
                    loc=tf.zeros(n), scale_diag=tf.ones(n)
                )
            )
        ]
    )
    return prior_model

# Define variational posterior weight distribution as multivariate Gaussian.
# Note that the learnable parameters for this distribution are the means,
# variances, and covariances.
def posterior(kernel_size, bias_size, dtype=None):

    from tensorflow import keras
    import tensorflow_probability as tfp

    n = kernel_size + bias_size
    posterior_model = keras.Sequential(
        [
            tfp.layers.VariableLayer(
                tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype
            ),
            tfp.layers.MultivariateNormalTriL(n),
        ]
    )
    return posterior_model


class BayesianNN(Model):

    def add_layers(self, *args, **kwargs):
        import tensorflow as tf
        from tensorflow.keras import layers
        import tensorflow_probability as tfp

        num_inputs = len(self.input_features)

        hidden_units = self.config['model']['layers']['hidden_units']
        train_size = self.config['model']['layers']['train_size']
        activation = self.config['model']['layers']['activation']
        uncertainty_type = self.config['model']['layers']['uncertainty_type']

        assert uncertainty_type in ("epistemic", "aleotoric", "both")

        epistemic, aleoteric = False, False
        if uncertainty_type in ("epistemic", "both"):
            epistemic = True

        if uncertainty_type in ("aleotoric", "both"):
            aleoteric = True

        inputs = layers.Input(shape=(num_inputs,), dtype=tf.float32, name='Inputs')

        features = layers.BatchNormalization()(inputs)

        if epistemic:
            # Create hidden layers with weight uncertainty using the DenseVariational layer.
            for units in hidden_units:
                features = tfp.layers.DenseVariational(
                    units=units,
                    make_prior_fn=prior,
                    make_posterior_fn=posterior,
                    kl_weight=1 / train_size,
                    activation=activation,
                )(features)
        else:
            for units in hidden_units:
                features = layers.Dense(units, activation=activation)(features)

        if aleoteric:
            # Create a probabilisticå output (Normal distribution), and use the `Dense` layer
            # to produce the parameters of the distribution.
            # We set units=2 to learn both the mean and the variance of the Normal distribution.
            distribution_params = layers.Dense(units=2)(features)
            outputs = tfp.layers.IndependentNormal(1)(distribution_params)
        else:
            # The output is deterministic: a single point estimate.
            outputs = layers.Dense(units=1)(features)

        self.allow_weight_loading = True
        return inputs, outputs

# We use the tfp.layers.DenseVariational layer instead of
# the standard keras.layers.Dense layer in the neural
# network model.

def create_epistemic_bnn_model(train_size, hidden_units, activation):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    import tensorflow_probability as tfp

    inputs = layers.Input(shape=(26,), dtype=tf.float32, name='model_Inputs')
    features = layers.BatchNormalization()(inputs)

    # Create hidden layers with weight uncertainty using the DenseVariational layer.
    for units in hidden_units:
        features = tfp.layers.DenseVariational(
            units=units,
            make_prior_fn=prior,
            make_posterior_fn=posterior,
            kl_weight=1 / train_size,
            activation=activation,
        )(features)

    # The output is deterministic: a single point estimate.
    outputs = layers.Dense(units=1)(features)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def compute_predictions(model, examples,
                        iterations=100):
    predicted = []
    for _ in range(iterations):
        predicted.append(model.predict(examples, batch_size=len(examples),
                                       verbose=0))
    predicted = np.concatenate(predicted, axis=1)

    prediction_mean = np.mean(predicted, axis=1).tolist()
    prediction_min = np.min(predicted, axis=1).tolist()
    prediction_max = np.max(predicted, axis=1).tolist()
    prediction_range = (np.max(predicted, axis=1) - np.min(predicted, axis=1)).tolist()

    df = pd.DataFrame()
    df['prediction'] = prediction_mean
    df['upper'] = prediction_max
    df['lower'] = prediction_min

    return df


def plot_ci(prediction_dist, dtype, alpha,  show=True):
    # plots the confidence interval

    lower = np.min(prediction_dist, axis=1)
    upper = np.max(prediction_dist, axis=1)
    mean = np.mean(prediction_dist, axis=1)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.fill_between(np.arange(len(lower)), upper, lower, alpha=0.5, color='C1')
    p1 = ax.plot(mean, color="C1", label="Prediction")
    p2 = ax.fill(np.NaN, np.NaN, color="C1", alpha=0.5)
    percent = int((1 - alpha) * 100)
    ax.legend([(p2[0], p1[0]), ], [f'{percent}% Confidence Interval'],
              fontsize=12)

    ax.set_xlabel(f"{dtype} Samples", fontsize=12)
    ax.set_ylabel("qe", fontsize=12)

    if show:
        plt.tight_layout()
        plt.show()

    return ax

class AleatoricBNN(Model):

    def add_layers(self, *args, **kwargs):
        import tensorflow as tf
        from tensorflow.keras import layers
        import tensorflow_probability as tfp

        num_inputs: int = len(self.input_features)

        hidden_units = self.config['model']['layers']['hidden_units']
        train_size = self.config['model']['layers']['train_size']
        activation = self.config['model']['layers']['activation']

        inputs = layers.Input(shape=(num_inputs,), dtype=tf.float32, name='Inputs')
        features = layers.BatchNormalization()(inputs)

        # Create hidden layers with weight uncertainty using the DenseVariational layer.
        for units in hidden_units:
            features = tfp.layers.DenseVariational(
                units=units,
                make_prior_fn=prior,
                make_posterior_fn=posterior,
                kl_weight=1 / train_size,
                activation=activation,
            )(features)

        # Create a probabilisticå output (Normal distribution), and use the `Dense` layer
        # to produce the parameters of the distribution.
        # We set units=2 to learn both the mean and the variance of the Normal distribution.
        distribution_params = layers.Dense(units=2)(features)
        outputs = tfp.layers.IndependentNormal(1)(distribution_params)

        return inputs, outputs

def create_aleatoric_bnn_model(num_inputs, train_size, hidden_units, activation):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    import tensorflow_probability as tfp

    inputs = layers.Input(shape=(num_inputs,), dtype=tf.float32, name='model_Inputs')
    features = layers.BatchNormalization()(inputs)

    # Create hidden layers with weight uncertainty using the DenseVariational layer.
    for units in hidden_units:
        features = tfp.layers.DenseVariational(
            units=units,
            make_prior_fn=prior,
            make_posterior_fn=posterior,
            kl_weight=1 / train_size,
            activation=activation,
        )(features)

    # Create a probabilisticå output (Normal distribution), and use the `Dense` layer
    # to produce the parameters of the distribution.
    # We set units=2 to learn both the mean and the variance of the Normal distribution.
    distribution_params = layers.Dense(units=2)(features)
    outputs = tfp.layers.IndependentNormal(1)(distribution_params)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def negative_loglikelihood(targets, estimated_distribution):
    return -estimated_distribution.log_prob(targets)

# %%

class BayesModel(Model):
    """
    A model which can be used to quantify aleotoric uncertainty, or epsitemic
    uncertainty or both. Following parameters must be defined in a dictionary
    called ``layers``.
    >>> model = BayesModel(model={"layers": {'hidden_units': [1,], 'train_size': 100,
    ...           'activation': 'sigmoid'}})


    hidden_units : List[int]
    train_size : int
    activation : str
    uncertainty_type : str
        either ``epistemic`` or ``aleoteric`` or ``both``
    """
    def add_layers(self, *args, **kwargs)->tuple:
        import tensorflow as tf
        from tensorflow.keras import layers
        import tensorflow_probability as tfp

        hidden_units = self.config['model']['layers']['hidden_units']
        train_size = self.config['model']['layers']['train_size']
        activation = self.config['model']['layers']['activation']
        uncertainty_type = self.config['model']['layers'].get('uncertainty_type',
                                                              'epistemic')

        assert uncertainty_type in ("epistemic", "aleoteric", "both")
        epistemic = False
        aleoteric = False

        if uncertainty_type in ("epistemic", "both"):
            epistemic = True

        if uncertainty_type in ("aleoteric", "both"):
            aleoteric = True

        inputs = layers.Input(shape=len(self.input_features, ), dtype=tf.float32)
        features = layers.BatchNormalization()(inputs)

        if epistemic:
            # Create hidden layers with weight uncertainty using the
            # DenseVariational layer.
            for units in hidden_units:
                features = tfp.layers.DenseVariational(
                    units = units,
                    make_prior_fn = prior,
                    make_posterior_fn = posterior,
                    kl_weight = 1 / train_size,
                    activation = activation,
                )(features)
        else:
            for units in hidden_units:
                features = layers.Dense(units, activation=activation)(features)

        if aleoteric:
            # Create a probabilisticå output (Normal distribution), and use
            # the `Dense` layer
            # to produce the parameters of the distribution.
            # We set units=2 to learn both the mean and the variance of the
            # Normal distribution.
            distribution_params = layers.Dense(units=2)(features)
            outputs = tfp.layers.IndependentNormal(1)(distribution_params)
        else:
            # The output is deterministic: a single point estimate.
            outputs = layers.Dense(units=1)(features)

        return inputs, outputs

# %%

def shap_scatter(
        feature_shap_values,
        feature_data,
        color_feature:pd.Series=None,
        color_feature_is_categorical:bool = False,
        feature_name:str = '',
        show_hist:bool = True,
        palette_name = "magma",
        s:int = 70,
        ax:plt.Axes = None,
        edgecolors='black',
        linewidth=0.8,
        alpha=0.8,
        show:bool = True,
        process_ticks:bool = True,
        leg_ncol=1,
        bbox = (1.05, 1),
        **scatter_kws,
):
    if ax is None:
        fig, ax = plt.subplots()

    if color_feature is None:
        c = None
    else:
        if color_feature_is_categorical:
            if isinstance(palette_name, (tuple, list)):
                assert len(palette_name) == len(color_feature.unique())
                rgb_values = palette_name
            else:
                rgb_values = sns.color_palette(palette_name, color_feature.unique().__len__())
            color_map = dict(zip(color_feature.unique(), rgb_values))
            c= color_feature.map(color_map)
        else:
            c = color_feature.values.reshape(-1,)

    _, pc = scatter(
        feature_data,
        feature_shap_values,
        c=c,
        s=s,
        marker="o",
        edgecolors=edgecolors,
        linewidth=linewidth,
        alpha=alpha,
        ax=ax,
        show=False,
        **scatter_kws
    )

    if color_feature is not None:
        feature_wrt_name = ' '.join(color_feature.name.split('_'))
        if color_feature_is_categorical:
            # add a legend
            handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=v,
                              label=k, markersize=8) for k, v in color_map.items()]

            ax.legend(title=feature_wrt_name,
                  handles=handles, bbox_to_anchor=bbox, loc='upper left',
                      title_fontsize=14,
                      ncol=leg_ncol,
                      )
        else:
            cbar = plt.colorbar(pc, aspect=40)
            cbar.ax.set_ylabel(feature_wrt_name, rotation=90, labelpad=14,
                               fontsize=14)

            if process_ticks:
                set_ticks(cbar.ax, "y")

            cbar.set_alpha(1)
            cbar.outline.set_visible(False)

    ax.set_xlabel(feature_name, fontsize=14)
    ax.set_ylabel(f"SHAP value for {feature_name}")
    ax.axhline(0, color='grey', linewidth=1.3, alpha=0.3, linestyle='--')

    if process_ticks:
        set_ticks(ax)
        set_ticks(ax, "y")

    if show_hist:
        if isinstance(feature_data, (pd.Series, pd.DataFrame)):
            feature_data = feature_data.values
        x = feature_data

        if len(x) >= 500:
            bin_edges = 50
        elif len(x) >= 200:
            bin_edges = 20
        elif len(x) >= 100:
            bin_edges = 10
        else:
            bin_edges = 5

        ax2 = ax.twinx()

        xlim = ax.get_xlim()

        ax2.hist(x.reshape(-1,), bin_edges,
                 range=(xlim[0], xlim[1]),
                 density=False, facecolor='#000000', alpha=0.1, zorder=-1)
        ax2.set_ylim(0, len(x))
        ax2.set_yticks([])

    if show:
        plt.show()

    return ax


def set_ticks(axes:plt.Axes, which="x", size=12):
    ticks = getattr(axes, f"get_{which}ticks")()
    ticks = np.array(ticks)

    if 'float' in ticks.dtype.name:
        ticks = np.round(ticks, 2)
    else:
        ticks = ticks.astype(int)

    getattr(axes, f"set_{which}ticklabels")(ticks, size=size, weight="bold")
    return

def set_xticklabels(
        ax:plt.Axes,
        max_ticks:Union[int, Any] = 5,
        dtype = int,
        weight = "bold",
        fontsize:Union[int, float]=12,
        max_xtick_val=None,
        min_xtick_val=None,
):
    """

    :param ax:
    :param max_ticks:
        maximum number of ticks, if not set, all the default ticks will be used
    :param dtype:
    :param weight:
    :param fontsize:
    :param max_xtick_val:
        maxikum value of tick
    :param min_xtick_val:
    :return:
    """
    return set_ticklabels(ax, "x", max_ticks, dtype, weight, fontsize,
                          max_tick_val=max_xtick_val,
                          min_tick_val=min_xtick_val)


def set_yticklabels(
        ax:plt.Axes,
        max_ticks:Union[int, Any] = 5,
        dtype=int,
        weight="bold",
        fontsize:int=12,
        max_ytick_val = None,
        min_ytick_val = None
):
    return set_ticklabels(
        ax, "y", max_ticks, dtype, weight,
        fontsize=fontsize,
        max_tick_val=max_ytick_val,
        min_tick_val=min_ytick_val,
    )


def set_ticklabels(
        ax:plt.Axes,
        which:str = "x",
        max_ticks:int = 5,
        dtype=int,
        weight="bold",
        fontsize:int=12,
        max_tick_val = None,
        min_tick_val = None,
):
    ticks_ = getattr(ax, f"get_{which}ticks")()
    ticks = np.array(ticks_)
    if len(ticks)<1:
        warnings.warn(f"can not get {which}ticks {ticks_}")
        return

    if max_ticks:
        ticks = np.linspace(min_tick_val or min(ticks), max_tick_val or max(ticks), max_ticks)

    ticks = ticks.astype(dtype)

    getattr(ax, f"set_{which}ticks")(ticks)

    getattr(ax, f"set_{which}ticklabels")(ticks, weight=weight, fontsize=fontsize)
    return ax

# %%

def set_rcParams(kwargs:dict = None):

    _kwargs = {'axes.labelsize': '14',
               'axes.labelweight': 'bold',
               'xtick.labelsize': '12',
               'ytick.labelsize': '12',
               'font.weight': 'bold',
               'legend.title_fontsize': '12',
               'axes.titleweight': 'bold',
               'axes.titlesize': '14',
               #"font.family" : "Times New Roman"
               }

    if sys.platform == "linux":
       
        _kwargs['font.family'] = 'serif'
        _kwargs['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    else:
        _kwargs['font.family'] = "Times New Roman"
    
    if kwargs:
        _kwargs.update(kwargs)

    for k,v in _kwargs.items():
        plt.rcParams[k] = v

    return

# %%

def plot_correlation(df, **kwargs):
    eda = EDA(data=df, save=False, show=False)
    ax = eda.correlation(figsize=(12, 10), cmap="rocket",
                         square=True, **kwargs)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=12, weight='bold')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=12, weight='bold')
    plt.tight_layout()
    plt.show()
    return

# %%
def residual_plot(
        train_true,
        train_prediction,
        test_true,
        test_prediction,
        label='',
        show:bool = False
)->np.ndarray:

    train_true = to_1d_array(train_true)
    train_prediction = to_1d_array(train_prediction)
    test_true = to_1d_array(test_true)
    test_prediction = to_1d_array(test_prediction)

    fig, axis = plt.subplots(1, 2, sharey="all"
                             , gridspec_kw={'width_ratios': [2, 1]})
    test_y = test_true.reshape(-1, ) - test_prediction.reshape(-1, )
    train_y = train_true.reshape(-1, ) - train_prediction.reshape(-1, )
    train_hist_kws = dict(bins=20, linewidth=0.7,
                          edgecolor="k", grid=False, color="#fe8977",
                          orientation='horizontal')
    hist(train_y, show=False, ax=axis[1],
         label="Training", **train_hist_kws)
    plot(train_prediction, train_y, 'o', show=False,
         ax=axis[0],
         color="#fe8977",
         markerfacecolor="#fe8977",
         markeredgecolor="black", markeredgewidth=0.7,
         alpha=0.9, label="Training"
         )

    _hist_kws = dict(bins=40, linewidth=0.7,
                     edgecolor="k", grid=False,
                     color="#9acad4",
                     orientation='horizontal')
    hist(test_y, show=False, ax=axis[1],
         **_hist_kws)

    set_xticklabels(axis[1], 3)

    plot(test_prediction, test_y, 'o', show=False,
         ax=axis[0],
         color="darksalmon",
         markerfacecolor="#9acad4",
         markeredgecolor="black", markeredgewidth=0.7,
         ax_kws=dict(
             #xlabel=f"Predicted {label}",
             #ylabel="Residual",
             legend_kws=dict(loc="upper left"),
         ),
         alpha=0.9, label="Test",
         )

    axis[0].set_xlabel(f'Predicted {label}', fontsize=14)
    axis[0].set_ylabel('Residual', fontsize=14)
    set_yticklabels(axis[0], 5)
    axis[0].axhline(0.0, color="black", ls="--")
    plt.subplots_adjust(wspace=0.15)

    if show:
       plt.show()
    return axis

# %%

def regression_plot(
        train_true,
        train_pred,
        test_true,
        test_pred,
        label = 'qe',
        max_xtick_val = None,
        max_ytick_val = None,
        min_xtick_val=None,
        min_ytick_val=None,
        max_ticks = 5,
        show=False
)->plt.Axes:
    TRAIN_RIDGE_LINE_KWS = [{'color': '#9acad4', 'lw': 1.0},
                            {'color': '#9acad4', 'lw': 1.0}]
    TRAIN_HIST_KWS = [{'color': '#9acad4', 'bins': 50},
                      {'color': '#9acad4', 'bins': 50}]

    ax = regplot(train_true, train_pred,
                 marker_size=35,
                 marker_color="#9acad4",
                 line_color='k',
                 fill_color='k',
                 scatter_kws={'edgecolors': 'black',
                              'linewidth': 0.7,
                              'alpha': 0.9,
                              },
                 label="Training",
                 show=False
                 )

    axHistx, axHisty = AddMarginalPlots(
        ax,
        ridge=False,
        pad=0.25,
        size=0.7,
        ridge_line_kws=TRAIN_RIDGE_LINE_KWS,
        hist_kws=TRAIN_HIST_KWS
    )(train_true, train_pred)

    train_r2 = RegressionMetrics(train_true, train_pred).r2()
    test_r2 = RegressionMetrics(test_true, test_pred).r2()
    ax.annotate(f'Training $R^2$= {round(train_r2, 2)}',
                xy=(0.95, 0.30),
                xycoords='axes fraction',
                horizontalalignment='right',
                verticalalignment='top',
                fontsize=12, weight="bold")
    ax.annotate(f'Test $R^2$= {round(test_r2, 2)}',
                xy=(0.95, 0.20),
                xycoords='axes fraction',
                horizontalalignment='right', verticalalignment='top',
                fontsize=12, weight="bold")

    ax_ = regplot(test_true, test_pred,
                  marker_size=35,
                  marker_color="#fe8977",
                  line_style=None,
                  scatter_kws={'edgecolors': 'black',
                               'linewidth': 0.7,
                               'alpha': 0.9,
                               },
                  show=False,
                  label="Test",
                  ax=ax
                  )

    ax_.legend(fontsize=12, prop=dict(weight="bold"))
    TEST_RIDGE_LINE_KWS = [{'color': "#fe8977", 'lw': 1.0},
                           {'color': "#fe8977", 'lw': 1.0}]
    TEST_HIST_KWS = {'color': "#fe8977", 'bins': 50}
    AddMarginalPlots(
        ax,
        ridge=False,
        pad=0.25,
        size=0.7,
        ridge_line_kws=TEST_RIDGE_LINE_KWS,
        hist_kws=TEST_HIST_KWS
    )(test_true, test_pred, axHistx, axHisty)

    set_xticklabels(
        ax_,
        max_xtick_val=max_xtick_val,
        min_xtick_val=min_xtick_val,
        max_ticks=max_ticks,
    )
    set_yticklabels(
        ax_,
        max_ytick_val=max_ytick_val,
        min_ytick_val=min_ytick_val,
        max_ticks=max_ticks
    )
    ax.set_xlabel(f"Experimental {label}")
    ax.set_ylabel(f"Predicted {label}")

    if show:
        plt.show()
    return ax

# %%

def shape_scatter_plots_for_cat(
        shap_values: np.ndarray,
        TrainX: pd.DataFrame,
        feature_name: str,
        encoders: dict,
        figsize: tuple = None,
        n_to_keep:int = 10,
        label_original:bool = True,
        cmap=None,
        show: bool = True
):
    f, axes = plt.subplots(1, 3,
                              figsize=figsize or (12, 9))

    index = TrainX.columns.to_list().index(feature_name)


    for idx, (feature, ax) in enumerate(zip(CAT_COLUMNS, axes.flat)):

        enc = encoders[feature]
        dec_feature = pd.Series(
            enc.inverse_transform(TrainX.loc[:, feature].values.astype(int)),
                                name=feature)

        dec_feature = merge_uniques(dec_feature, n_to_keep=n_to_keep)
        color_feature = dec_feature

        if not label_original:
            # instead of showing the actual names, we still prefer to
            # label encode them because actual names takes very large
            # space in figure/axes
            encoder_ = LabelEncoder()
            color_feature = pd.Series(
                encoder_.fit_transform(color_feature),
                                      name=feature)
            print(feature, encoder_.classes_)


        color_feature.name = LABEL_MAP.get(color_feature.name, color_feature.name)


        ax = shap_scatter(
            shap_values[:, index],
            feature_data=TrainX.loc[:, feature_name].values,
            feature_name=LABEL_MAP.get(feature_name, feature_name),
            color_feature=color_feature,
            color_feature_is_categorical=True,
            show=False,
            alpha=0.5,
            ax=ax,
            process_ticks=False,
            cmap=cmap,
        )

    if show:
        plt.tight_layout()
        plt.show()

    return

# %%

def shap_scatter_plots(
        shap_values:np.ndarray,
        TrainX:pd.DataFrame,
        feature_name:str,
        encoders:dict,
        cmap = None,
        figsize:tuple=None,
        show:bool = True
):
    """
    It is expected that the columns in TrainX and shap_values have same order.
    :param shap_values:
    :param TrainX:
    :param feature_name:
    :param encoders:
    :param figsize
    :param show

    :return:
    """

    NUM_COLUMNS = [col for col in TrainX.columns if col not in CAT_COLUMNS]

    f, axes = create_subplots(len(NUM_COLUMNS),
                              figsize=figsize or (12, 9))

    index = TrainX.columns.to_list().index(feature_name)


    for idx, (feature, ax) in enumerate(zip(NUM_COLUMNS, axes.flat)):

        clr_f_is_cat = False
        if feature in CAT_COLUMNS:
            clr_f_is_cat = True

        if feature in CAT_COLUMNS:
            enc = encoders[feature]
            dec_feature = pd.Series(
                enc.inverse_transform(TrainX.loc[:, feature].values.astype(int)),
                                    name=feature)

            dec_feature = merge_uniques(dec_feature, n_to_keep=4)
            color_feature = dec_feature

            # instead of showing the actual names, we still prefer to
            # label encode them because actual names takes very large
            # space in figure/axes
            encoder_ = LabelEncoder()
            color_feature = pd.Series(
                encoder_.fit_transform(color_feature),
                                      name=feature)
            print(feature, encoder_.classes_)
        else:
            color_feature = TrainX.loc[:, feature]

        color_feature.name = LABEL_MAP.get(color_feature.name, color_feature.name)

        ax = shap_scatter(
            shap_values[:, index],
            feature_data=TrainX.loc[:, feature_name].values,
            feature_name=LABEL_MAP.get(feature_name, feature_name),
            color_feature=color_feature,
            color_feature_is_categorical=clr_f_is_cat,
            show=False,
            alpha=0.5,
            ax=ax,
            cmap=cmap,
        )
        ax.set_ylabel('')

        if idx < 10:
            ax.set_xlabel('')

    if show:
        plt.tight_layout()
        plt.show()
    return

def pie_from_series(
        data:pd.Series,
        cmap="tab20",
        label_percent:bool = True,
        n_to_merge:int = None,
        leg_pos=None,
        show=True,
        fontsize=14
):

    d:pd.Series = data.value_counts()
    labels = d.index.tolist()
    vals = d.values
    colors = make_cols_from_cmap(cm=cmap, num_cols=len(vals))
    percent = 100. * vals / vals.sum()

    outs = pie(fractions=percent, autopct=None,
               colors=colors, show=False)
    patches, texts = outs

    if label_percent:
        labels = ['{0}: {1:1.2f} %'.format(i, j) for i, j in zip(labels, percent)]
    else:
        labels = ['{0} (n={1:4})'.format(i, j) for i, j in zip(labels, vals)]

    patches, labels, dummy = zip(*sorted(zip(patches, labels, vals),
                                         key=lambda x: x[2],
                                         reverse=True))

    plt.legend(patches, labels, bbox_to_anchor=leg_pos or (1.1, 1.),
               fontsize=fontsize)

    if show:
        plt.tight_layout()
        plt.show()
    return

def merge_uniques(
        series:pd.Series,
        n_to_keep:int=5,
        replace_with="Rest"
):
    counts = series.value_counts()

    values = []
    for idx, (value, count) in enumerate(counts.items()):
        if idx >= n_to_keep:
            values.append(value)

    series = series.replace(values, replace_with)
    return series

def scatter_(col, first, second, label,
             ax, fig, cmap="Spectral"):


    data, _, encoders = make_data(encoding="le")
    CAT_COLUMNS = ["Adsorbent", "Feedstock", "ion_type"]

    c = data[col].values

    if label in CAT_COLUMNS:
        feature_wrt = data[label].values.reshape(-1,).astype(int)
        feature_wrt = encoders[label].inverse_transform(feature_wrt)
        feature_wrt = merge_uniques(pd.Series(feature_wrt), 10)
        rgb_values = sns.color_palette("tab20", feature_wrt.unique().__len__())
        color_map = dict(zip(feature_wrt.unique(), rgb_values))
        c = feature_wrt.map(color_map)
        cmap=None

    pc = ax.scatter(first, second,
                    c=c,
                    s = 2,
            cmap=cmap,
            )

    if label in CAT_COLUMNS:
        handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=v,
                          label=k, markersize=8) for k, v in color_map.items()]

        ax.legend(title=LABEL_MAP.get(label, label),
              handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left',
                  title_fontsize=14, ncols=2
                  )
    else:
        colorbar = fig.colorbar(pc, ax=ax)

        colorbar.set_label(LABEL_MAP.get(label, label))
        despine_axes(colorbar.ax)

    return

# %%

def plot_ci_from_dist(
        distribution,
        mean,
        n:int = None,
        show=True,
        ci=0.95,
        figsize=None,
        line_color="teal",
        fill_color = "teal",
        fill_alpha=0.4,
        ylabel = "Adsorption",
        name=""
):
    """plots ci from distribution"""
    lower, upper = distribution.interval(ci)

    assert len(mean) == len(lower) == len(upper)
    n = n or len(mean)

    fig, ax = plt.subplots(figsize=figsize or (8, 6))
    ax = plot(mean[0:n], show=False, ax=ax,
              label='True', color=line_color)

    x = np.arange(len(lower))
    ax.fill_between(x[0:n],
                    lower[0:n],
                    upper[0:n],
                    color=fill_color,
                    label=f"{int(ci * 100)}% CI",
                    alpha=fill_alpha,
                    )
    ax.legend()
    ax.set_xlabel('Samples')
    ax.set_ylabel(ylabel)
    ax.grid(visible=True, ls='--', color='lightgrey')
    if SAVE:
        plt.savefig(f"../manuscript/figures/ngb_ci_95_{name}", dpi=600, bbox_inches="tight")
    plt.tight_layout()
    if show:
        plt.show()
    return ax

# %%

def local_pdfs_for_nbg(
        model,
        X,
        y,
        cmap="tab10",
        indices = None,
        show=True,
):
    indices = indices or [28, 35, 45, 54, 90, 100, 150, 200, 250, 300]

    clrs = make_clrs_from_cmap(cm=cmap, num_cols=len(indices))

    y_pred_all = model.predict(X)
    Y_dists = model.pred_dist(X)
    y_range = np.linspace(y.min(), y.max(), len(y))

    dist_values = Y_dists.pdf(y_range.reshape(-1, 1)).transpose()  # plot index 0 and 114

    fig, all_axes = plt.subplots(len(indices), 1, sharex="all", figsize=(6, 8))

    idx1 = 0
    for idx, ax in zip(indices, all_axes.flat):
        ax = plot(y_range, dist_values[idx],
                  color=clrs[idx1],
                  ax=ax, show=False)
        ax.axvline(y_pred_all[idx], color="darkgray")
        ax.text(x=0.5, y=0.7, s=f"Sample ID: {idx}",
                transform=ax.transAxes)

        ax.set_yticks([])

        idx1 += 1

    xticks = np.array(ax.get_xticks()).astype(int)
    ax.set_xticklabels(xticks)
    ax.set_xlabel("Adsorption")
    if SAVE:
        plt.savefig("../manuscript/figures/ngb_local_pdfs", dpi=600, bbox_inches="tight")
    if show:
        plt.show()
    return ax

# %%

def plot_1d_pdp(pdp, X:pd.DataFrame, show=True):
    f, axes = create_subplots(X.shape[1], figsize=(10, 12))

    for ax, feature, clr in zip(axes.flat, X.columns, LINE_COLORS):
        pdp_vals, ice_vals = pdp.calc_pdp_1dim(X.values, feature)

        ax = pdp.plot_pdp_1dim(pdp_vals, ice_vals, X.values,
                               feature,
                               pdp_line_kws={
                                   'color': clr, 'zorder': 3},
                               ice_color="gray",
                               ice_lines_kws=dict(zorder=2, alpha=0.15),
                               ax=ax,
                               show=False,
                               )
        ax.set_xlabel(LABEL_MAP.get(feature, feature), fontsize=14)
        ax.set_ylabel(f"E[f(x) | " + feature + "]", fontsize=14)

    plt.tight_layout()
    if show:
        plt.show()
    return axes

# %%

def plot_feature_importance(importances, labels, title:str=''):
    fig, ax = plt.subplots(figsize=(6,7))
    ax = bar_chart(
        importances,
        labels=labels,
        color="tan",
        sort=True,
        ax_kws=dict(
            title=title,
            ylabel="Input Features",
            xlabel="Importance",
            ylabel_kws={"fontsize": 12, 'weight': 'bold'},
            xlabel_kws={"fontsize": 12, 'weight': 'bold'},
        ),
        ax=ax,
        show=False
    )
    plt.tight_layout()
    if SAVE:
        plt.savefig(f"../manuscript/figures/feature_imortance_bar_chart_{title}",
                    dpi=600, bbox_inches="tight")
    plt.show()
    return ax


# %%

def pdp_interaction_plot(pdp, feature_1:str, feature_2):
    pdp.plot_interaction(features=[feature_1, feature_2])
    feature_1 = feature_1.replace('(', '_')
    feature_1 = feature_1.replace(')', '_')
    feature_1 = feature_1.replace('/', '_')

    feature_2 = feature_2.replace('(', '_')
    feature_2 = feature_2.replace(')', '_')
    feature_2 = feature_2.replace('/', '_')

    if SAVE:
        plt.savefig(f"../manuscript/figures/{feature_1}_{feature_2}",
                    dpi=500, bbox_inches="tight")
    plt.show()
    return

# %%
def print_metrics(true, prediction, prefix:str):
    metrics = RegressionMetrics(true, prediction)
    print(f"{prefix} R2: {metrics.r2()}")
    print(f"{prefix} R2 Score: {metrics.r2_score()}")
    print(f"{prefix} RMSE Score: {metrics.rmse()}")
    print(f"{prefix} MAE: {metrics.mae()}")
    return

def maybe_save_prediction(true, prediction, name, save=False):

    true = to_1d_array(true)
    prediction = to_1d_array(prediction)

    if save or SAVE:
        pd.DataFrame(
            np.column_stack([true, prediction]),
            columns=['true', 'prediction']
        ).to_csv(os.path.join('../manuscript/figures/data/', f'{name}.csv'), index=False)
    return