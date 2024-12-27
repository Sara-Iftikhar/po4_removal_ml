"""
===========
3. NGBoost
===========
"""

import random

import shap

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

from ngboost.distns import Exponential

from ai4water import Model
from ai4water.utils.utils import TrainTestSplit
from ai4water.postprocessing import PartialDependencePlot, ProcessPredictions

from ngboost import NGBRegressor

from utils import plot_1d_pdp
from utils import set_rcParams
from utils import LABEL_MAP
from utils import plot_ci_from_dist
from utils import local_pdfs_for_nbg
from utils import pdp_interaction_plot
from utils import version_info, make_data
from utils import plot_feature_importance
from utils import maybe_save_prediction
from utils import shape_scatter_plots_for_cat
from utils import shap_scatter
from utils import residual_plot, SAVE, regression_plot, shap_scatter_plots

# %%

seed = 313
np.random.seed(seed)
random.seed(seed)

# %%

for lib, ver in version_info().items():
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
# Model Building
# ===============

ngb_rgr = NGBRegressor(Dist=Exponential,
                       n_estimators=204,
                       learning_rate=0.058293744434511935,
                       minibatch_frac=0.7209642683250839,
                       col_sample=0.5341159717834566,
                       random_state=313
                       )
model = Model(
    model = ngb_rgr,
    mode="regression",
    category="ML",
    input_features=data.columns.tolist()[:-1],
    output_features=data.columns.tolist()[-1],
)

# %%
# Model training
# ===============

ngb = model.fit(X_train.values, y_train.values,
                X_val=X_test.values,
                Y_val=y_test.values)

# %%

_ = model.prediction_analysis(
    features= ['Adsorption_time (min)',  'loading (g)'],
    x=np.concatenate([X_train, X_test], axis=0),
    y=np.concatenate([y_train, y_test], axis=0),
    num_grid_points=(6, 6),
    border=True,
    show=False,
    annotate_kws=dict(fontsize=12)
)
plt.tight_layout()
plt.show()


# %%
# Performance evaluation
# =============================

# %%
# Evaluating model performance on training data

print(model.evaluate(X_train, y_train,
                     metrics=['r2', 'nse', 'rmse', 'mae']))
# %%

tr_preds = model.predict(X_train)

# %%

train_dist = ngb.pred_dist(X_train)

plot_ci_from_dist(train_dist,
                  y_train.values, line_color="peru",
                  fill_color="peachpuff", n = 200,
                  fill_alpha=0.99,
                  name="train")

# %%
# Now we get the negative log likelihood for training data.
# Negative log likelihood is another way of quantification of
# model performance.

print(-train_dist.logpdf(y_train).mean())

# %%
# Prediction on test data

test_preds = model.predict(X_test)

# %%

print(model.evaluate(X_test, y_test,
                     metrics=['r2', 'nse', 'rmse', 'mae', 'pbias'],
                     max_iter=ngb.best_val_loss_itr))

# %%

residual_plot(
    y_train.values,
    tr_preds,
    y_test.values,
    test_preds,
    label="qe",
)
if SAVE:
    plt.savefig("../manuscript/figures/residue_ngb", dpi=600, bbox_inches="tight")
plt.show()

# %%

ax = regression_plot(
    y_train, tr_preds,
    y_test, test_preds)
ax.set_xlim(-20, ax.get_xlim()[1])
ax.set_ylim(-20, ax.get_ylim()[1])
if SAVE:
    plt.savefig("../manuscript/figures/reg_ngb", dpi=600, bbox_inches="tight")
plt.show()

# %%

pp = ProcessPredictions('regression', 1, show=False)
output = pp.edf_plot(y_train, tr_preds,
                     label=("Absolute Error (Training)", "Prediction (Training)"))
output[1].legend(loc=(0.5, 0.18), frameon=False)
output = pp.edf_plot(y_test, test_preds, marker='*', ax=output[0], pred_axes=output[1],
                     label=("Absolute Error (Test)", "Prediction (Test)"))
output[1].legend(loc=(0.57, 0.18), frameon=False)
output[0].set_xlabel('Absolute Error', fontsize=12)
output[1].set_xlabel('Prediction', fontsize=12)
output[0].set_ylabel('Commulative Probability', fontsize=12)
if SAVE:
    plt.savefig("../manuscript/figures/ngb_edf", dpi=600, bbox_inches="tight")
plt.show()

# %%
# get negative log likelihood

test_dist = ngb.pred_dist(X_test)

print(-test_dist.logpdf(y_test).mean())

# %%

plot_ci_from_dist(
    test_dist,
    y_test.values,
    line_color="peru",
    fill_color="peachpuff",
    fill_alpha=0.99,
    n=200
)

# %%

maybe_save_prediction(y_train, tr_preds, 'ngb_train')

# %%

maybe_save_prediction(y_test, test_preds, 'ngb_test')

# %%
# feature importance for loc trees

feature_importance_loc = ngb.feature_importances_[0]

plot_feature_importance(
    feature_importance_loc,
    labels=[LABEL_MAP.get(feature, feature) for feature in data.columns.tolist()[:-1]],
    title='loc')

# %%
# feature importance for scale trees

if len(ngb.feature_importances_)==2:
    feature_importance_scale = ngb.feature_importances_[1]

    plot_feature_importance(
        feature_importance_scale,
        labels=[LABEL_MAP.get(feature, feature) for feature in data.columns.tolist()[:-1]],
        title='scale')

# %%

X = np.concatenate((X_train, X_test), axis=0)
y = np.concatenate((y_train, y_test), axis=0)

local_pdfs_for_nbg(ngb, X, y)

# %%

all_y = ngb.predict(X)

Y_dists = ngb.pred_dist(X)

# %%

_ = plot_ci_from_dist(Y_dists, all_y, n=200)

# %%
# SHAP
# ======

# %%
# SHAP plot for ``loc``
# ---------------------

explainer = shap.TreeExplainer(ngb, model_output=0)
shap_values = explainer.shap_values(X_train)
shap.summary_plot(
    shap_values, X_train,
    feature_names=[LABEL_MAP.get(feature, feature) for feature in data.columns.tolist()[:-1]],
    plot_size=(6, 7),
    show=False)
if SAVE:
    plt.savefig("../manuscript/figures/ngb_shap_summary_loc", dpi=600, bbox_inches="tight")
plt.show()

# %%

sv_bar = np.mean(np.abs(shap_values), axis=0)
plot_feature_importance(
    sv_bar,
    labels=[LABEL_MAP.get(feature, feature) for feature in data.columns.tolist()[:-1]],
    title='loc')

# %%
# SHAP plot for ``scale``
# --------------------------

explainer = shap.TreeExplainer(ngb,
                               model_output=1 if len(ngb.feature_importances_)==2 else 0)
shap_values = explainer.shap_values(X_train)
shap.summary_plot(
    shap_values, X_train,
    feature_names=[LABEL_MAP.get(feature, feature) for feature in data.columns.tolist()[:-1]],
    plot_size=(6, 7),
    show=False)
if SAVE:
    plt.savefig("../manuscript/figures/ngb_shap_summary_scale", dpi=600, bbox_inches="tight")
plt.show()

# %%
sv_bar = np.mean(np.abs(shap_values), axis=0)
plot_feature_importance(
    sv_bar,
    labels=[LABEL_MAP.get(feature, feature) for feature in data.columns.tolist()[:-1]],
    title='scale')

# %%

# %%
TrainX = pd.DataFrame(X_train,
                      columns=data.columns.tolist()[:-1])
# %%
# Pyrolysis Temperature
feature_name = 'Pyrolysis_temp'

shap_scatter_plots(shap_values, TrainX, feature_name,
                   figsize=(14, 12),
                   encoders=encoders, show=False)
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()

# %%

shape_scatter_plots_for_cat(shap_values, TrainX, feature_name,
                   figsize=(14, 5),
                   encoders=encoders, show=False)
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.tight_layout()
plt.show()

# %%

feature_name = 'Ci_ppm'
shap_scatter_plots(shap_values, TrainX, feature_name,
                   encoders=encoders,
                   figsize=(16, 14),show=False)
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()

# %%

shape_scatter_plots_for_cat(shap_values, TrainX, feature_name,
                   figsize=(14, 5),
                   encoders=encoders, show=False)
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.tight_layout()
plt.show()

# %%

feature_name = 'solution pH'
shap_scatter_plots(shap_values, TrainX, feature_name,
                   encoders=encoders,
                   figsize=(16, 14),show=False)
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()

# %%

shape_scatter_plots_for_cat(shap_values, TrainX, feature_name,
                   figsize=(14, 5),
                   encoders=encoders, show=False)
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.tight_layout()
plt.show()

# %%

feature_name = 'adsorption_temp'
shap_scatter_plots(shap_values, TrainX, feature_name,
                   encoders=encoders,
                   figsize=(16, 14),show=False)
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()

# %%
shape_scatter_plots_for_cat(shap_values, TrainX, feature_name,
                   figsize=(14, 5),
                   encoders=encoders, show=False)
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.tight_layout()
plt.show()

# %%

feature_name = 'Adsorbent'
shap_scatter_plots(shap_values, TrainX, feature_name,
                   encoders=encoders,
                   figsize=(16, 14),show=False)
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()

# %%
shape_scatter_plots_for_cat(shap_values, TrainX, feature_name,
                   figsize=(14, 5),
                   encoders=encoders, show=False)
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.tight_layout()
plt.show()

# %%

feature_name = 'Surface area'
shap_scatter_plots(shap_values, TrainX, feature_name,
                   encoders=encoders,
                   figsize=(16, 14),show=False)
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()

# %%
shape_scatter_plots_for_cat(shap_values, TrainX, feature_name,
                   figsize=(14, 5),
                   encoders=encoders, show=False)
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.tight_layout()
plt.show()

# %%

feature_name = 'Adsorption_time (min)'
shap_scatter_plots(shap_values, TrainX, feature_name,
                   encoders=encoders,
                   figsize=(16, 14),show=False)
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()

# %%
shape_scatter_plots_for_cat(shap_values, TrainX, feature_name,
                   figsize=(14, 5),
                   encoders=encoders, show=False)
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.tight_layout()
plt.show()


# %%
feature_name = 'Heating rate (oC)'
shap_scatter_plots(shap_values, TrainX, feature_name,
                   encoders=encoders,
                   figsize=(16, 14),show=False)
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()

# %%
shape_scatter_plots_for_cat(shap_values, TrainX, feature_name,
                   figsize=(14, 5),
                   encoders=encoders, show=False)
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.tight_layout()
plt.show()

# %%
feature_name = 'C'
shap_scatter_plots(shap_values, TrainX, feature_name,
                   encoders=encoders,
                   figsize=(16, 14),show=False)
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()

# %%
shape_scatter_plots_for_cat(shap_values, TrainX, feature_name,
                   figsize=(14, 5),
                   encoders=encoders, show=False)
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.tight_layout()
plt.show()

# %%
feature_name = 'O'
shap_scatter_plots(shap_values, TrainX, feature_name,
                   encoders=encoders,
                   figsize=(16, 14),show=False)
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()

# %%
shape_scatter_plots_for_cat(shap_values, TrainX, feature_name,
                   figsize=(14, 5),
                   encoders=encoders, show=False)
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.tight_layout()
plt.show()

# %%

f, axes = plt.subplots(2, ncols=4, figsize=(14, 7))
axes = axes.flatten()

## axes 1
feature1 = 'Adsorption_time (min)'
feature2 = 'C'

index = TrainX.columns.to_list().index(feature1)
color_feature = TrainX.loc[:, feature2]
color_feature.name = LABEL_MAP.get(color_feature.name, color_feature.name)

ax = shap_scatter(
    shap_values[:, index],
    feature_data=TrainX.loc[:, feature1].values,
    feature_name=LABEL_MAP.get(feature1, feature1),
    color_feature=color_feature,
    show=False,
    alpha=0.5,
    ax=axes[0],
    process_ticks=False
)
ax.set_ylabel(f"SHAP ({LABEL_MAP.get(feature1, feature1)})", fontsize=14)

## axes 2

feature1 = 'Adsorption_time (min)'
feature2 = 'solution pH'

index = TrainX.columns.to_list().index(feature1)
color_feature = TrainX.loc[:, feature2]
color_feature.name = LABEL_MAP.get(color_feature.name, color_feature.name)

ax = shap_scatter(
    shap_values[:, index],
    feature_data=TrainX.loc[:, feature1].values,
    feature_name=LABEL_MAP.get(feature1, feature1),
    color_feature=color_feature,
    show=False,
    alpha=0.5,
    ax=axes[1],
    process_ticks=False
)
ax.set_ylabel(f"SHAP ({LABEL_MAP.get(feature1, feature1)})", fontsize=14)

## axes 3
feature1 = 'Pyrolysis_temp'
feature2 = 'Pyrolysis_temp'
index = TrainX.columns.to_list().index(feature1)
color_feature = TrainX.loc[:, feature2]
color_feature.name = LABEL_MAP.get(color_feature.name, color_feature.name)

ax = shap_scatter(
    shap_values[:, index],
    feature_data=TrainX.loc[:, feature1].values,
    feature_name=LABEL_MAP.get(feature1, feature1),
    color_feature=color_feature,
    show=False,
    alpha=0.5,
    ax=axes[2],
    process_ticks=False
)
ax.set_ylabel(f"SHAP ({LABEL_MAP.get(feature1, feature1)})", fontsize=14)

## axes 4
feature1 = 'C'
feature2 = 'Pyrolysis_temp'

index = TrainX.columns.to_list().index(feature1)
color_feature = TrainX.loc[:, feature2]
color_feature.name = LABEL_MAP.get(color_feature.name, color_feature.name)

ax = shap_scatter(
    shap_values[:, index],
    feature_data=TrainX.loc[:, feature1].values,
    feature_name=LABEL_MAP.get(feature1, feature1),
    color_feature=color_feature,
    show=False,
    alpha=0.5,
    ax=axes[3],
    process_ticks=False
)
ax.set_ylabel(f"SHAP ({LABEL_MAP.get(feature1, feature1)})", fontsize=14)

## axes 5
feature1 = 'O'
feature2 = 'C'
index = TrainX.columns.to_list().index(feature1)
color_feature = TrainX.loc[:, feature2]
color_feature.name = LABEL_MAP.get(color_feature.name, color_feature.name)

ax = shap_scatter(
    shap_values[:, index],
    feature_data=TrainX.loc[:, feature1].values,
    feature_name=LABEL_MAP.get(feature1, feature1),
    color_feature=color_feature,
    show=False,
    alpha=0.5,
    ax=axes[4],
    process_ticks=False
)
ax.set_ylabel(f"SHAP ({LABEL_MAP.get(feature1, feature1)})", fontsize=14)

## axes 6
feature1 = 'C'
feature2 = 'C'
index = TrainX.columns.to_list().index(feature1)
color_feature = TrainX.loc[:, feature2]
color_feature.name = LABEL_MAP.get(color_feature.name, color_feature.name)

ax = shap_scatter(
    shap_values[:, index],
    feature_data=TrainX.loc[:, feature1].values,
    feature_name=LABEL_MAP.get(feature1, feature1),
    color_feature=color_feature,
    show=False,
    alpha=0.5,
    ax=axes[5],
    process_ticks=False
)
ax.set_ylabel(f"SHAP ({LABEL_MAP.get(feature1, feature1)})", fontsize=14)

## axes 7

feature1 = 'O'
feature2 = 'O'
index = TrainX.columns.to_list().index(feature1)
color_feature = TrainX.loc[:, feature2]
color_feature.name = LABEL_MAP.get(color_feature.name, color_feature.name)

ax = shap_scatter(
    shap_values[:, index],
    feature_data=TrainX.loc[:, feature1].values,
    feature_name=LABEL_MAP.get(feature1, feature1),
    color_feature=color_feature,
    show=False,
    alpha=0.5,
    ax=axes[6],
    process_ticks=False
)
ax.set_ylabel(f"SHAP ({LABEL_MAP.get(feature1, feature1)})", fontsize=14)

## axes 8
feature1 = 'Surface area'
feature2 = 'Pyrolysis_temp'

index = TrainX.columns.to_list().index(feature1)
color_feature = TrainX.loc[:, feature2]
color_feature.name = LABEL_MAP.get(color_feature.name, color_feature.name)

ax = shap_scatter(
    shap_values[:, index],
    feature_data=TrainX.loc[:, feature1].values,
    feature_name=LABEL_MAP.get(feature1, feature1),
    color_feature=color_feature,
    show=False,
    alpha=0.5,
    ax=axes[7],
    process_ticks=False
)
ax.set_ylabel(f"SHAP ({LABEL_MAP.get(feature1, feature1)})", fontsize=14)
##
plt.tight_layout()
plt.show()

# %%
# Partial Dependence Plot
# ===========================

mpl.rcParams.update(mpl.rcParamsDefault)


pdp = PartialDependencePlot(
    model.predict,
    TrainX,
    num_points=10,
    feature_names=TrainX.columns.tolist(),
    show=False,
    save=False
)

# %%

pdp_interaction_plot(pdp, 'Pyrolysis_temp', 'Heating rate (oC)')

# %%

pdp_interaction_plot(pdp,'Pyrolysis_temp', 'Ci_ppm')

# %%

pdp_interaction_plot(pdp,'Adsorption_time (min)', 'Ci_ppm')

# %%

pdp_interaction_plot(pdp,'adsorption_temp', 'Ci_ppm')

# %%

pdp_interaction_plot(pdp,'adsorption_temp', 'Adsorption_time (min)')

# %%

pdp_interaction_plot(pdp,'solution pH', 'Ci_ppm')

# %%

pdp_interaction_plot(pdp,'solution pH', 'Adsorption_time (min)')

# %%

pdp_interaction_plot(pdp,'solution pH', 'adsorption_temp')

# %%

plot_1d_pdp(pdp, TrainX)
