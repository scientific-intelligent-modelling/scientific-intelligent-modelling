import feyn

ql = feyn.QLattice()

from feyn.datasets import make_regression

train, test = feyn.datasets.make_regression(n_samples=500, n_features=3)
models = ql.auto_run(data=train, output_name='y', kind='regression', n_epochs=20)

for model in models:
    print(model.sympify())