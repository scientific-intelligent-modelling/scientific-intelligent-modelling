"""Testing the examples from the documentation."""

# Author: Trevor Stephens <trevorstephens.com>
#
# License: BSD 3 clause

import numpy as np

from sklearn.utils._testing import assert_almost_equal
from sklearn.utils.validation import check_random_state

# import scientific_intelligent_modelling as sim
from scientific_intelligent_modelling import SymbolicRegressor

def test_symbolic_regressor():
    """Check that SymbolicRegressor example works"""

    rng = check_random_state(0)
    X_train = rng.uniform(-1, 1, 100).reshape(50, 2)
    y_train = X_train[:, 0] ** 2 - X_train[:, 1] ** 2 + X_train[:, 1] - 1
    X_test = rng.uniform(-1, 1, 100).reshape(50, 2)
    y_test = X_test[:, 0] ** 2 - X_test[:, 1] ** 2 + X_test[:, 1] - 1

    est_gp = SymbolicRegressor(population_size=5000, generations=20,
                               stopping_criteria=0.01, p_crossover=0.7,
                               p_subtree_mutation=0.1, p_hoist_mutation=0.05,
                               p_point_mutation=0.1, max_samples=0.9,
                               parsimony_coefficient=0.01, random_state=0)
    est_gp.fit(X_train, y_train)

    assert(len(est_gp._programs) == 7)
    expected = 'sub(add(-0.999, X1), mul(sub(X1, X0), add(X0, X1)))'
    assert(est_gp.__str__() == expected)
    assert_almost_equal(est_gp.score(X_test, y_test), 0.99999, decimal=5)
    dot_data = est_gp._program.export_graphviz()
    expected = ('digraph program {\nnode [style=filled]\n0 [label="sub", '
                'fillcolor="#136ed4"] ;\n1 [label="add", fillcolor="#136ed4"] '
                ';\n2 [label="-0.999", fillcolor="#60a6f6"] ;\n3 [label="X1", '
                'fillcolor="#60a6f6"] ;\n1 -> 3 ;\n1 -> 2 ;\n4 [label="mul", '
                'fillcolor="#136ed4"] ;\n5 [label="sub", fillcolor="#136ed4"] '
                ';\n6 [label="X1", fillcolor="#60a6f6"] ;\n7 [label="X0", '
                'fillcolor="#60a6f6"] ;\n5 -> 7 ;\n5 -> 6 ;\n8 [label="add", '
                'fillcolor="#136ed4"] ;\n9 [label="X0", fillcolor="#60a6f6"] '
                ';\n10 [label="X1", fillcolor="#60a6f6"] ;\n8 -> 10 ;\n8 -> 9 '
                ';\n4 -> 8 ;\n4 -> 5 ;\n0 -> 4 ;\n0 -> 1 ;\n}')
    assert(dot_data == expected)
    assert(est_gp._program.parents == {'method': 'Crossover',
                                       'parent_idx': 1555,
                                       'parent_nodes': range(1, 4),
                                       'donor_idx': 78,
                                       'donor_nodes': []})
    idx = est_gp._program.parents['donor_idx']
    fade_nodes = est_gp._program.parents['donor_nodes']
    assert(est_gp._programs[-2][idx].__str__() == 'add(-0.999, X1)')
    assert_almost_equal(est_gp._programs[-2][idx].fitness_, 0.351803319075)
    dot_data = est_gp._programs[-2][idx].export_graphviz(fade_nodes=fade_nodes)
    expected = ('digraph program {\nnode [style=filled]\n0 [label="add", '
                'fillcolor="#136ed4"] ;\n1 [label="-0.999", '
                'fillcolor="#60a6f6"] ;\n2 [label="X1", fillcolor="#60a6f6"] '
                ';\n0 -> 2 ;\n0 -> 1 ;\n}')
    assert(dot_data == expected)
    idx = est_gp._program.parents['parent_idx']
    fade_nodes = est_gp._program.parents['parent_nodes']
    expected = 'sub(sub(X1, 0.939), mul(sub(X1, X0), add(X0, X1)))'
    assert(est_gp._programs[-2][idx].__str__() == expected)
    assert_almost_equal(est_gp._programs[-2][idx].fitness_, 0.17080204042)
    dot_data = est_gp._programs[-2][idx].export_graphviz(fade_nodes=fade_nodes)
    expected = ('digraph program {\nnode [style=filled]\n0 [label="sub", '
                'fillcolor="#136ed4"] ;\n1 [label="sub", fillcolor="#cecece"] '
                ';\n2 [label="X1", fillcolor="#cecece"] ;\n3 [label="0.939", '
                'fillcolor="#cecece"] ;\n1 -> 3 ;\n1 -> 2 ;\n4 [label="mul", '
                'fillcolor="#136ed4"] ;\n5 [label="sub", fillcolor="#136ed4"] '
                ';\n6 [label="X1", fillcolor="#60a6f6"] ;\n7 [label="X0", '
                'fillcolor="#60a6f6"] ;\n5 -> 7 ;\n5 -> 6 ;\n8 [label="add", '
                'fillcolor="#136ed4"] ;\n9 [label="X0", fillcolor="#60a6f6"] '
                ';\n10 [label="X1", fillcolor="#60a6f6"] ;\n8 -> 10 ;\n8 -> 9 '
                ';\n4 -> 8 ;\n4 -> 5 ;\n0 -> 4 ;\n0 -> 1 ;\n}')
    assert(dot_data == expected)


if __name__ == '__main__':
    print(1)
    rng = check_random_state(0)
    X_train = rng.uniform(-1, 1, 100).reshape(50, 2)
    y_train = X_train[:, 0] ** 2 - X_train[:, 1] ** 2 + X_train[:, 1] - 1
    X_test = rng.uniform(-1, 1, 100).reshape(50, 2)
    y_test = X_test[:, 0] ** 2 - X_test[:, 1] ** 2 + X_test[:, 1] - 1

    est_gp = SymbolicRegressor(population_size=5000, generations=20,
                               stopping_criteria=0.01, p_crossover=0.7,
                               p_subtree_mutation=0.1, p_hoist_mutation=0.05,
                               p_point_mutation=0.1, max_samples=0.9,
                               parsimony_coefficient=0.01, random_state=0)
    est_gp.fit(X_train, y_train)

    print(est_gp)