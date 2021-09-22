import numpy as np
from numpy import ndarray

import shap
import networkx as nx

np.random.seed(0)
data = np.random.randint(0, 2, size=(10, 2))


def model(data):
    return data[:, 0] + data[:, 1] + data[:, 0] * data[:, 1]


right_answer_sym: ndarray = np.zeros(data.shape)

right_answer_sym[:, 0] += data[:, 0]
right_answer_sym[:, 1] += data[:, 1]
right_answer_sym[:, 0] += (data[:, 0] * data[:, 1]) / 2
right_answer_sym[:, 1] += (data[:, 0] * data[:, 1]) / 2
shap_values = shap.explainers.Permutation(model, np.zeros((1,2)))(data)

assert np.allclose(right_answer_sym, shap_values.values)

print("yo symm coorect")

G = nx.DiGraph()
G.add_edges_from([(1, 2)])
H = nx.convert_node_labels_to_integers(G)

right_answer_asy = np.zeros(data.shape)
right_answer_asy[:, 0] += data[:, 0]
right_answer_asy[:, 1] += data[:, 1]
right_answer_asy[:, 0] += (data[:, 0] * data[:, 1]) / 2
right_answer_asy[:, 1] += (data[:, 0] * data[:, 1]) / 2
shap_values_asy = shap.explainers.Asymmetric(model, data)(data)
#, causal_info = H

print("computed some new shaps")

