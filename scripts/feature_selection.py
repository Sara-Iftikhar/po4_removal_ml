"""
====================
2. Feature Selection
====================
This file compares different feature selection methods.
"""

import arfs
import arfs.feature_selection as arfsfs
import arfs.feature_selection.allrelevant as arfsgroot
import numpy as np
from arfs.benchmark import highlight_tick, compare_varimp, sklearn_pimp_bench
from arfs.feature_selection import MinRedundancyMaxRelevance

from ngboost import NGBRegressor
from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt

from utils import version_info
from utils import make_data
from utils import set_rcParams

# %%

for lib, ver in version_info().items():
    print(lib, ver)

# %%

set_rcParams()

# %%

data, _, encoders = make_data(encoding="le")

X = data.iloc[:, 0:-1]
y = data.iloc[:, -1]

print(X.shape, y.shape)
# %%

selector = arfsfs.CollinearityThreshold(threshold=0.95)
X_filtered = selector.fit_transform(
    data.iloc[:, 0:-1]
)
print(f"The features going in the selector are : {selector.feature_names_in_}")
print(f"The support is : {selector.support_}")
print(f"The selected features are : {selector.get_feature_names_out()}")

selector.plot_association()

# %%
# Leshy
# ================
model = RandomForestRegressor(random_state=313)

# # Leshy, all the predictors, no-preprocessing
# feat_selector = arfsgroot.Leshy(
#     model, n_estimators=30, verbose=1, max_iter=10,
#     random_state=42, importance="pimp"
# )
#
# feat_selector.fit(X, y, sample_weight=None)
# print(f"The selected features: {feat_selector.get_feature_names_out()}")
# print(f"The agnostic ranking: {feat_selector.ranking_}")
# print(f"The naive ranking: {feat_selector.ranking_absolutes_}")
# fig = feat_selector.plot_importance(n_feat_per_inch=5)
#
# # highlight synthetic random variable
# fig = highlight_tick(figure=fig, str_match="random")
# fig = highlight_tick(figure=fig, str_match="genuine", color="green")
# plt.show()

# %%

# # Leshay with SHAP
# # ====================
# model = RandomForestRegressor(random_state=313)
# # Leshy
# feat_selector = arfsgroot.Leshy(
#     model, n_estimators=50, verbose=1, max_iter=10,
#     random_state=313, importance="shap"
# )
# feat_selector.fit(X, y, sample_weight=None)
# print(f"The selected features: {feat_selector.get_feature_names_out()}")
# print(f"The agnostic ranking: {feat_selector.ranking_}")
# print(f"The naive ranking: {feat_selector.ranking_absolutes_}")
# fig = feat_selector.plot_importance(n_feat_per_inch=5)
#
# # highlight synthetic random variable
# fig = highlight_tick(figure=fig, str_match="random")
# fig = highlight_tick(figure=fig, str_match="genuine", color="green")
# plt.show()
#
# # %%
# # BoostAGroota
# # ===============
#
# model = RandomForestRegressor(random_state=313)
# # BoostAGroota
# feat_selector = arfsgroot.BoostAGroota(
#     est=model,
#     cutoff=1,
#     iters=10,
#     max_rounds=10,
#     delta=0.1,
#     silent=True,
#     importance="shap",
# )
# feat_selector.fit(X, y, sample_weight=None)
# print(f"The selected features: {feat_selector.get_feature_names_out()}")
# print(f"The agnostic ranking: {feat_selector.ranking_}")
# print(f"The naive ranking: {feat_selector.ranking_absolutes_}")
# fig = feat_selector.plot_importance(n_feat_per_inch=5)
#
# # highlight synthetic random variable
# fig = highlight_tick(figure=fig, str_match="random")
# fig = highlight_tick(figure=fig, str_match="genuine", color="green")
# plt.show()
#
# # %%
# # GrootCV
# # ================
#
# feat_selector = arfsgroot.GrootCV(
#     objective="mse", cutoff=1, n_folds=5, n_iter=10, silent=True
# )
# feat_selector.fit(X, y, sample_weight=None)
# print(f"The selected features: {feat_selector.get_feature_names_out()}")
# print(f"The agnostic ranking: {feat_selector.ranking_}")
# print(f"The naive ranking: {feat_selector.ranking_absolutes_}")
# fig = feat_selector.plot_importance(n_feat_per_inch=5)
#
# # highlight synthetic random variable
# fig = highlight_tick(figure=fig, str_match="random")
# fig = highlight_tick(figure=fig, str_match="genuine", color="green")
# plt.show()
#
# # %%
# # Maximal relevance minimal redundancy
# # ========================================
#
# fs_mrmr = MinRedundancyMaxRelevance(
#     n_features_to_select=10,
#     relevance_func=None,
#     redundancy_func=None,
#     task="regression",  # "classification",
#     denominator_func=np.mean,
#     only_same_domain=False,
#     return_scores=False,
#     show_progress=True,
#     n_jobs=-1,
# )
#
# fs_mrmr.fit(X=X, y=y, sample_weight=None)
# print(f"The selected features: {fs_mrmr.get_feature_names_out()}")