"""RAG-SR wrapper backed by EvolutionaryForestRegressor.

RAG-SR 的公开仓库只是薄封装；真实实现位于 `evolutionary_forest` 包中。
当前包装器固定数值型 SR benchmark 口径，默认不启用分类特征编码。
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from ..base_wrapper import BaseWrapper
from scientific_intelligent_modelling.benchmarks.normalizers import normalize_ragsr_artifact


class RAGSRRegressor(BaseWrapper):
    """RAG-SR benchmark wrapper using EvolutionaryForestRegressor."""

    _DEFAULT_PARAMS = {
        "n_gen": 100,
        "n_pop": 200,
        "select": "AutomaticLexicase",
        "cross_pb": 0.9,
        "mutation_pb": 0.1,
        "max_height": 10,
        "ensemble_size": 1,
        "initial_tree_size": "0-6",
        "gene_num": 10,
        "basic_primitives": "Add,Sub,Mul,AQ,Sqrt,AbsLog,Abs,Square,RSin,RCos,Max,Min,Neg",
        "base_learner": "RidgeCV",
        "ridge_alphas": "Auto",
        "verbose": False,
        "boost_size": None,
        "normalize": "MinMax",
        "external_archive": 1,
        "max_trees": 10000,
        "library_clustering_mode": "Worst",
        "pool_addition_mode": "Smallest~Auto",
        "pool_hard_instance_interval": 10,
        "random_order_replacement": True,
        "pool_based_addition": True,
        "semantics_length": 50,
        "change_semantic_after_deletion": True,
        "include_subtree_to_lib": True,
        "library_updating_mode": "Recent",
        # 当前 benchmark 全是数值型特征；设为 None 可避开旧 category_encoders
        # 与新版 scikit-learn 的兼容问题。
        "categorical_encoding": None,
        "root_crossover": True,
        "scaling_before_replacement": False,
        "score_func": "R2",
        "number_of_invokes": 0,
        "mutation_scheme": "uniform-plus",
        "environmental_selection": None,
        "record_training_data": False,
        "complementary_replacement": False,
        "validation_size": 0,
        "constant_type": "Float",
        "full_scaling_after_replacement": False,
        "neural_pool": 0.1,
        "neural_pool_num_of_functions": 5,
        "weight_of_contrastive_learning": 0.05,
        "neural_pool_dropout": 0.1,
        "neural_pool_transformer_layer": 1,
        "neural_pool_hidden_size": 64,
        "neural_pool_mlp_layers": 3,
        "selective_retrain": True,
        "negative_data_augmentation": True,
        "negative_local_search": False,
    }
    _META_PARAMS = {
        "exp_name",
        "exp_path",
        "problem_name",
        "seed",
        "n_features",
        "feature_names",
        "target_name",
        "timeout_in_seconds",
        "progress_snapshot_interval_seconds",
        "task_label",
        "task_global_index",
        "expected_dataset_rel",
        "expected_dataset_dir",
    }
    _ALLOWED_PARAMS = set(_DEFAULT_PARAMS) | {"random_state", "categorical_features"}
    _INT_PARAMS = {
        "n_gen",
        "n_pop",
        "max_height",
        "ensemble_size",
        "gene_num",
        "external_archive",
        "max_trees",
        "pool_hard_instance_interval",
        "semantics_length",
        "number_of_invokes",
        "validation_size",
        "neural_pool_num_of_functions",
        "neural_pool_transformer_layer",
        "neural_pool_hidden_size",
        "neural_pool_mlp_layers",
        "random_state",
    }
    _FLOAT_PARAMS = {
        "cross_pb",
        "mutation_pb",
        "neural_pool",
        "weight_of_contrastive_learning",
        "neural_pool_dropout",
    }

    def __init__(self, **kwargs: Any):
        raw_kwargs = dict(kwargs or {})
        self._contract_n_features = raw_kwargs.get("n_features")
        self._contract_feature_names = raw_kwargs.get("feature_names")
        self._contract_target_name = raw_kwargs.get("target_name")
        self._seed = raw_kwargs.get("seed")
        self.params, self._fit_kwargs = self._validate_and_normalize_params(raw_kwargs)
        self.model = None
        self._equation = None

    @classmethod
    def _validate_and_normalize_params(cls, raw_params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        params = dict(raw_params or {})
        seed = params.get("seed")
        for key in cls._META_PARAMS:
            params.pop(key, None)

        if seed is not None and "random_state" not in params:
            params["random_state"] = int(seed)

        fit_kwargs = {}
        if "categorical_features" in params:
            fit_kwargs["categorical_features"] = params.pop("categorical_features")

        for key, value in cls._DEFAULT_PARAMS.items():
            params.setdefault(key, deepcopy(value))

        unknown = sorted(set(params) - cls._ALLOWED_PARAMS)
        if unknown:
            raise ValueError(
                "RAG-SR 参数不受支持: {}。当前允许的参数有: {}".format(
                    ", ".join(unknown),
                    ", ".join(sorted(cls._ALLOWED_PARAMS)),
                )
            )

        for key in cls._INT_PARAMS:
            if key in params and params[key] is not None:
                params[key] = int(params[key])
        for key in cls._FLOAT_PARAMS:
            if key in params and params[key] is not None:
                params[key] = float(params[key])
        return params, fit_kwargs

    def fit(self, X, y):
        self._validate_explicit_dataset_contract(
            X,
            n_features=self._contract_n_features,
            feature_names=self._contract_feature_names,
            target_name=self._contract_target_name,
            context="RAGSRRegressor.fit",
        )
        from evolutionary_forest.forest import EvolutionaryForestRegressor

        x_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        self.model = EvolutionaryForestRegressor(**self.params)
        self.model.fit(x_arr, y_arr, **self._fit_kwargs)
        self._equation = self._extract_model_expression()
        return self

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("RAG-SR 模型尚未训练")
        return np.asarray(self.model.predict(np.asarray(X, dtype=float))).reshape(-1)

    def _extract_model_expression(self) -> str:
        if self.model is None:
            raise RuntimeError("RAG-SR 模型尚未训练")
        if hasattr(self.model, "model"):
            value = self.model.model()
            if value is not None:
                return str(value)
        raise RuntimeError("RAG-SR 未能导出模型表达式")

    def get_optimal_equation(self):
        if self._equation is None:
            self._equation = self._extract_model_expression()
        return self._equation

    def get_total_equations(self):
        equation = self.get_optimal_equation()
        return [equation] if equation else []

    def export_canonical_symbolic_program(self):
        return normalize_ragsr_artifact(self.get_optimal_equation())
