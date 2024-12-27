"""
===============================
1. Exploratory data analysis
===============================
"""
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

from ai4water.utils import TrainTestSplit
from ai4water.utils.utils import get_version_info

from sklearn.manifold import TSNE
from umap import UMAP

from easy_mpl import ridge
from easy_mpl import boxplot, plot
from easy_mpl.utils import make_clrs_from_cmap
from easy_mpl.utils import create_subplots

from utils import SAVE, LABEL_MAP, pie_from_series
from utils import make_data, _load_data, \
    set_rcParams, distribution_plot, plot_correlation
from utils import CAT_COLUMNS
from utils import merge_uniques, scatter_
# %%

for lib, ver in get_version_info().items():
    print(lib, ver)

# %%

set_rcParams()

# %%

data, *_ = _load_data()

print(data.isna().sum())

# %%
cols = ['Adsorbent', 'Feedstock',
        'Pyrolysis_temp',
       'Heating rate (oC)', 'Pyrolysis_time (min)', 'C',
        'O',
        'Surface area',
        'Adsorption_time (min)',
        'Ci_ppm',
       'solution pH', 'rpm', 'Volume (L)', 'loading (g)',
       'adsorption_temp', 'Ion Concentration (mM)', 'ion_type','qe']
# %%
data = data[cols]

# %%
total = len(data)
print(data.isna().sum())
# %%
# Data filtering procedure

print('Initial data points: ', len(data))
df1 = data.dropna(subset=['Pyrolysis_temp'])
print('After dropping Pyrolysis Temp.: ', len(df1))
df2 = df1.dropna(subset=['Pyrolysis_time (min)'])
print('After dropping Pyrolysis Time: ', len(df2))
df3 = df2.dropna(subset=['Surface area'])
print('After dropping Surface area: ', len(df3))
df4 = df3.dropna(subset=['C'])
print('After dropping C: ', len(df4))
df5 = df4.dropna(subset=['O'])
print('After dropping O: ', len(df5))
df6 = df5.dropna(subset=['rpm'])
print('After dropping rpm: ', len(df6))
df7 = df6.dropna(subset=['solution pH'])
print('After dropping Solution pH: ', len(df7))
df8 = df7.dropna(subset=['Adsorption_time (min)'])
print('After dropping Adsorption Time: ', len(df8))
df9 = df8.dropna(subset=['Ion Concentration (mM)'])
print('After dropping Ion Conc: ', len(df9))
# %%
df9.isna().sum()

# %%
# Loading the original dataset

data, _, _ = make_data()
input_features = data.columns.tolist()

# %%

# Removing the categorical features from our dataframe
data_num = data.drop(columns=CAT_COLUMNS)
data, _, encoders = make_data()

NUM_COLUMNS = ['Pyrolysis_temp',
                'Heating rate (oC)', 'Pyrolysis_time (min)',
                'C', 'O', 'Surface area', 'Adsorption_time (min)',
                'Ci_ppm', 'solution pH', 'rpm', 'Volume (L)',
                'loading (g)', 'adsorption_temp', 'Ion Concentration (mM)',
                 'qe']

# %%
# Here, we are printing the shape of original dataset.
# The first value shows the number of samples/examples/datapoints
# and the second one shows the number of features.

print(data.shape)

# %%

print(data.columns)

# %%
# The first five samples are

data.head()

# %%
# The last five samples are

data.tail()

# %%
# The names of different adsorbents and their counts

data['Adsorbent'].value_counts()

# %%

data['Adsorbent'].unique()

# %%

print(len(data['Adsorbent'].unique()))

# %%
# The names of different Feedstock and their counts

data['Feedstock'].value_counts()

# %%

print(len(data['Feedstock'].unique()))

# %%
# The names of different Ion_type and their counts

data['ion_type'].value_counts()

# %%

print(len(data['ion_type'].unique()))

# %%
# get statistical summary of data

pd.set_option('display.max_columns', None)

print(data_num.describe())

# %%
# Associations
# =============

# plot correlation between numerical features

plot_correlation(data_num)

# %%
# plotting only those where correlation is higher than 0.6
plot_correlation(data_num, threshold=0.6, split="pos", method="spearman")

# %%
# plotting only those where correlation is below -0.4

plot_correlation(data_num, threshold=-0.4, split="neg", method="spearman")

# %%
# Line plot for numerical features
# ================================

fig, axes = create_subplots(data_num.shape[1], figsize=(10, 8))

for ax, col, label  in zip(axes.flat, data_num, data.columns):

    plot(data_num[col].values, ax=ax, ax_kws=dict(ylabel=LABEL_MAP.get(col, col)),
         lw=0.9,
         color='darkcyan', show=False)
plt.tight_layout()
plt.show()

# %%
cols = ['Pyrolysis_temp', 'Heating rate (oC)',
              'Pyrolysis_time (min)']
fig, axes = create_subplots(len(cols), figsize=(10, 8))

for idx, (ax, col)  in enumerate(zip(axes.flat, cols)):
    ax.grid(visible=True, ls='--', color='lightgrey')
    plot(data_num[col].values, ax=ax,
         ax_kws=dict(ylabel=LABEL_MAP.get(col, col)),
         lw=3.0,
         color='darkcyan', show=False)

    if idx in [3, 4]:
        ax.set_xlabel("Samples")
if SAVE:
    plt.savefig("../manuscript/figures/line_corr_feats", dpi=500, bbox_inches="tight")
plt.tight_layout()
plt.show()

# %%
# Clustring
# ==========

# %%
# TSNE
# ------

data, _, encoders = make_data(encoding="le")

tsne = TSNE(random_state=313, perplexity=30)
comp = tsne.fit_transform(data[input_features])

# %%
# TSNE plot for whole data

f, axes = create_subplots(len(input_features), sharex="all", sharey="all",
                       figsize=(9, 8))

for col, ax in zip(input_features, axes.flat):

    scatter_(col, comp[:, 0], comp[:, 1],
         label=col, ax=ax, fig=f)

plt.tight_layout()
plt.show()

set_rcParams()

# %%
# TSNE plot for categorical features

f, axes = create_subplots(len(CAT_COLUMNS), sharex="all", sharey="all",
                       figsize=(14, 9))

for col, ax in zip(CAT_COLUMNS, axes.flat):

    scatter_(col, comp[:, 0], comp[:, 1],
         label=col, ax=ax, fig=f)

plt.tight_layout()
plt.show()

# %%
# TSNE plot for numerical features


f, axes = create_subplots(len(NUM_COLUMNS), sharex="all", sharey="all",
                       figsize=(12, 10))

for col, ax in zip(NUM_COLUMNS, axes.flat):

    scatter_(col, comp[:, 0], comp[:, 1],
         label=col, ax=ax, fig=f)

plt.tight_layout()
plt.show()

# %%
# UMAP
# -----
mpl.rcParams.update(mpl.rcParamsDefault)
data, _, encoders = make_data(encoding="le")
umap = UMAP(random_state=313, n_neighbors=30, spread=25)
comp = umap.fit_transform(data[input_features])

# %%
# UMAP for numerical features
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['axes.titleweight'] = "bold"
f, axes = create_subplots(len(NUM_COLUMNS), sharex="all", sharey="all",
                       figsize=(9, 8))

for col, ax in zip(NUM_COLUMNS, axes.flat):

    scatter_(col, comp[:, 0], comp[:, 1],
         label=col, ax=ax, fig=f)
plt.subplots_adjust(wspace=0.5, hspace=0.3)
plt.show()

# %%
# UMAP for categorical features

f, axes = create_subplots(len(CAT_COLUMNS),
                          ncols=1,
                          sharex="all", sharey="all",
                       figsize=(6, 6))

for col, ax in zip(CAT_COLUMNS, axes.flat):

    scatter_(col, comp[:, 0], comp[:, 1],
         label=col, ax=ax, fig=f)

plt.tight_layout()
plt.show()

set_rcParams()

# %%
# Distributions
# =============

# %%
# Condition box-plots
# --------------------

data, *_ = make_data(encoding=False)

# %%
# distribution of all features with respect to a particular Feedstock

grps = data.groupby(by="Feedstock")

names = []
groups = []

for name, grp in grps:
    if len(grp)>100:
        names.append(name)
        groups.append(grp)

colors = make_clrs_from_cmap('tab20', num_cols=len(names))

f, axes = create_subplots(len(data_num.columns), figsize=(14, 14))

for col, ax in zip(data_num.columns, axes.flat):

    _, out = boxplot(
    [grp[col].values for grp in groups],
        flierprops=dict(ms=2.0),
        medianprops={"color": "black"},
        fill_color=colors,
        widths=0.7,
        patch_artist=True,
        ax=ax,
        show=False
    )
    ax.set_title(LABEL_MAP.get(col, col))
    ax.set_xticks([])

ax.legend(out["boxes"][0:10],
          names,
         loc=(1.1, 0.1))
plt.subplots_adjust(wspace=0.3, hspace=0.3)
if SAVE:
    plt.savefig('../manuscript/figures/dist_feedstock', bbox_inches="tight", dpi=600)
plt.show()

# %%

cols = ['Pyrolysis_temp', 'Heating rate (oC)',
              'Pyrolysis_time (min)',  'Surface area',
            'Adsorption_time (min)']

f, axes = create_subplots(len(cols), figsize=(8, 7))

for col, ax in zip(cols, axes.flat):

    _, out = boxplot(
    [grp[col].values for grp in groups],
        flierprops=dict(ms=2.0),
        medianprops={"color": "black"},
        fill_color=colors,
        widths=0.7,
        patch_artist=True,
        ax=ax,
        show=False
    )
    ax.set_title(LABEL_MAP.get(col, col))
    ax.set_xticks([])

ax.legend(out["boxes"][0:10],
          names,
         loc=(1.1, -0.2))

plt.subplots_adjust(wspace=0.3, hspace=0.3)
if SAVE:
    plt.savefig("../manuscript/figures/dist_feestock_short", dpi=500, bbox_inches="tight")
plt.show()

# %%
# distribution of all features with respect to a particular Ion type

data, *_ = make_data(encoding=False)

grps = data.groupby(by="ion_type")

names = []
groups = []

for name, grp in grps:
    if len(grp)>10:
        names.append(name)
        groups.append(grp)

colors = make_clrs_from_cmap('tab20', num_cols=len(names))

f, axes = create_subplots(len(data_num.columns), figsize=(11, 9))
for col, ax in zip(data_num.columns, axes.flat):

    _, out = boxplot(
        [grp[col].values for grp in groups],
        flierprops=dict(ms=2.0),
        medianprops={"color": "black"},
        fill_color=colors,
        widths=0.7,
        patch_artist=True,
        ax=ax,
        show=False
    )
    ax.set_title(LABEL_MAP.get(col, col))
    ax.set_xticks([])

ax.legend(out["boxes"][0:8],
          names,
          loc=(1.1, 0.1)
          )
plt.subplots_adjust(wspace=0.55, hspace=0.3)
if SAVE:
    plt.savefig('../manuscript/figures/dist_ion_type', bbox_inches="tight", dpi=600)
plt.show()

# %%
# distribution of all features with respect to a particular Adsorbent

grps = data.groupby(by="Adsorbent")
names = []
groups = []

for name, grp in grps:
    if len(grp)>70:
        names.append(name)
        groups.append(grp)

colors = make_clrs_from_cmap('tab20', num_cols=len(names))

f, axes = create_subplots(len(data_num.columns), figsize=(12, 10))
for col, ax in zip(data_num.columns, axes.flat):

    _, out = boxplot(
        [grp[col].values for grp in groups],
        flierprops=dict(ms=2.0),
        medianprops={"color": "black"},
        fill_color=colors,
        widths=0.7,
        patch_artist=True,
        ax=ax,
        show=False
    )
    ax.set_title(LABEL_MAP.get(col, col))
    ax.set_xticks([])
ax.legend(out["boxes"][0:8],
          names,
          loc=(1.1, 0.1)
          )
plt.subplots_adjust(wspace=0.4, hspace=0.3)
if SAVE:
    plt.savefig('../manuscript/figures/dist_adsorbent.png', bbox_inches="tight", dpi=600)
plt.show()


# %%
# Box plot for numerical features
# ----------------------------------

# %%
# show the box and (half) violin plots together

fig, axes = create_subplots(data_num.shape[1], figsize=(9, 8))
for ax, col in zip(axes.flat, data_num.columns):
    distribution_plot(ax=ax, data=data_num[col])
    ax.set_xlabel(xlabel=LABEL_MAP.get(col, col), weight='bold', fontsize=14)
    ax.set_yticklabels('')
plt.tight_layout()
if SAVE:
    plt.savefig("../manuscript/figures/distribution_num.png", dpi=600,
                bbox_inches="tight")
plt.show()

# %%
# Pie chart
# ---------

# %%
# here is a pie chart for unique values of 'Adsorbent'

merged_series = merge_uniques(data['Adsorbent'], 18)
pie_from_series(merged_series, cmap="coolwarm",  show=False, leg_pos=(0.85, 0.7))
if SAVE:
    plt.savefig('../manuscript/figures/pie_adsorbent', bbox_inches="tight", dpi=600)
plt.show()

# %%
# here is a pie chart for unique values of 'Feedstock'

merged_series = merge_uniques(data['Feedstock'], 10)
pie_from_series(merged_series, cmap="coolwarm",  show=False, leg_pos=(0.9, 0.9))
if SAVE:
    plt.savefig('../manuscript/figures/pie_feedstock', bbox_inches="tight", dpi=600)
plt.show()

# %%
# here is a pie chart for unique values of 'ion type'

pie_from_series(data['ion_type'], cmap="coolwarm",  show=False, leg_pos=(0.85, 0.75))
if SAVE:
    plt.savefig('../manuscript/figures/pie_ion_type', bbox_inches="tight", dpi=600)
plt.show()

# %%

data, _, encoders= make_data(encoding='le')
X_train, X_test, y_train, y_test = TrainTestSplit(seed=142).\
    random_split_by_groups(x=data.iloc[:,0:-1], y=data.iloc[:, -1],
    groups=data['Adsorbent'])
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# %%

print(f"""
Total adsorbent categories are {data['Adsorbent'].nunique()},
{X_train['Adsorbent'].nunique()} are used for training, while
{X_test['Adsorbent'].nunique()} are considered for test.
""")


ax = ridge([y_train.values.reshape(-1,),
       y_test.values.reshape(-1,)],
      labels=["Training", "Test"],
      share_axes=True,
      color = ['c', 'w'],
           show=False
      )
handles, labels = ax[0].get_legend_handles_labels()
for ha in handles:
    ha.set_edgecolor("black")
lgd = ax[0].legend(
    handles,
    labels,
    loc="upper right",
)
plt.show()