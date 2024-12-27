.. Adsorption_BNN documentation master file, created by
   sphinx-quickstart on Fri Feb  3 21:18:16 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Probabalistic modeling of Adsorption capacity of Phosphate onto biochars
=========================================================================

This project shows how to perform probabilistic modeling for tabular data.
To this end, we employ one decision tree based
approach ngboost and two neural network (NN)
based approaches. The first NN based approach is aleoteric. This method
qunatifies uncertainty in data generating process. To record aleoteric
uncertainty in a NN, we have to modify the output layer. In this case, the output
layer is a probability distribution instead of a fully connected layer. The
parameters of this distribution are learned during model training. The
other kind of uncertainty is epistemic uncertainty. This method quantifies
uncertainty in our knowledge. It should be noted that recording epistemic
uncertainty increases the learnable parameters of NN far more than vanilla
or aleoteric NN. However, the good news is that this type of uncertainty
can be reduced by
collecting more data. Such NNs are also known as bayesian NNs.
In such NNs, the weights (and biases) are distributions instead of scaler
values. The parameter of these distributions are again learned during model
training. We can also combine both aleoteric and epistemic uncertainties to make
our NNs into a probablistic NN. Such a NNs have weights as distributions and
output layer as distribution as well.

Data
----
The input data consists of eighteen parameters corresponding to removal efficiency
of biochar for PO4 from wastewater using adsorption such as adsorption experimental
conditions, elemental composition of adsorbent and physical characteristics of adsorbent.
Four of the total 18 parameters were categorical in nature, which included adsorbent type,
feedstock type, ion type and activation.
Our target is adsorption capacity (mg/g).
A comprehensive analysis of data is given in
:ref:`sphx_glr_auto_examples_eda.py`

Reproducibility
---------------
To replicate the experiments, you need to install all requirements given in
requirements file .
If your results are quite different from what are presented here, then make sure
that you are using the exact versions of the libraries which were used
at the time of running of these scripts. These versions are given printed
at the start of each script. Download all the .py files in the scripts including
utils.py (:ref:`sphx_glr_auto_examples_utils.py`) file. The data is expected to be
in the data folder under the scripts folder.

Some code in this project is inspired by following resoures

   - Probabilitic Bayesian Neural Networks code recipies from keras [1]_
   - Probabilistic Deep Learning Book [2]_, [3]_
   - Medium Blog [4]_
   - Tensorflow Probability tutorial [5]_, [6]_


Code
------
.. toctree::
   :maxdepth: 4

   auto_examples/index
   chord.ipynb
   BNN_taylor.ipynb


.. [1] https://keras.io/examples/keras_recipes/bayesian_neural_networks/
.. [2] https://www.manning.com/books/probabilistic-deep-learning
.. [3] https://github.com/tensorchiefs/dl_book/tree/master
.. [4] https://github.com/Frightera/Medium_Notebooks_English/tree/main
.. [5] https://github.com/tensorflow/probability/blob/main/tensorflow_probability/examples/jupyter_notebooks/Probabilistic_Layers_Regression.ipynb
.. [6] https://github.com/tensorflow/probability/blob/main/tensorflow_probability/examples/bayesian_neural_network.py