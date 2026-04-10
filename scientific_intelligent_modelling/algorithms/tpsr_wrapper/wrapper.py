"""Wrapper for TPSR (Transformer-based Planning for Symbolic Regression)."""

import base64
import os
import pickle
import re
import tempfile
import sys
import torch
import numpy as np
import shutil

from ..base_wrapper import BaseWrapper
from scientific_intelligent_modelling.benchmarks.normalizers import normalize_tpsr_artifact


class TPSRRegressor(BaseWrapper):
    def __init__(self, **kwargs):
        # 延迟导入，避免环境问题
        self.params = kwargs
        self.model = None
        self.best_tree = None
        self.all_trees = []
        self._predict_fn = None
        self._backend_params = {}
        self._predict_variable_names = []
        self._n_features = None

        # 设置默认参数
        self.params.setdefault("backbone_model", "e2e")
        self.params.setdefault("max_input_points", 10000)
        self.params.setdefault("max_number_bags", -1)
        self.params.setdefault("stop_refinement_after", 1)
        self.params.setdefault("n_trees_to_refine", 1)
        self.params.setdefault("rescale", True)
        self.params.setdefault("beam_size", 10)
        self.params.setdefault("beam_type", "sampling")
        self.params.setdefault("no_seq_cache", False)
        self.params.setdefault("no_prefix_cache", False)
        self.params.setdefault("width", 3)  # Top-k in TPSR's expansion step
        self.params.setdefault("num_beams", 1)  # Beam size in TPSR's evaluation
        self.params.setdefault("rollout", 3)  # Number of rollouts in TPSR
        self.params.setdefault("horizon", 200)  # Horizon of lookahead planning
        self.params.setdefault("seed", 23)
        self.params.setdefault("cpu", False)
        self.params.setdefault("train_value", False)
        self.params.setdefault("lam", 0.1)

        # NeSymReS 配置
        self.params.setdefault("nesymres_eq_setting_path", os.path.join("nesymres", "jupyter", "100M", "eq_setting.json"))
        self.params.setdefault("nesymres_cfg_path", os.path.join("nesymres", "jupyter", "100M", "config.yaml"))
        self.params.setdefault("nesymres_model_path", None)
        self.params.setdefault("symbolicregression_model_path", os.path.join("symbolicregression", "weights", "model.pt"))
        self.params.setdefault("symbolicregression_model_url", "https://dl.fbaipublicfiles.com/symbolicregression/model1.pt")

    def _runtime_import_paths(self):
        root_dir = os.path.dirname(os.path.abspath(__file__))
        tpsr_dir = os.path.join(root_dir, "tpsr")
        return [
            tpsr_dir,
            os.path.join(tpsr_dir, "nesymres", "src"),
        ]

    def _set_parser_attr(self, args, name: str, value):
        if value is None:
            return
        try:
            setattr(args, name, value)
        except Exception:
            # 某些参数可能不存在于当前 parser 版本中，略过即可
            pass

    def _resolve_tpsr_path(self, rel_path: str):
        if not rel_path:
            return None

        if os.path.isabs(rel_path):
            return rel_path

        root_dir = os.path.dirname(os.path.abspath(__file__))
        tpsr_dir = os.path.join(root_dir, "tpsr")
        candidate = os.path.join(tpsr_dir, rel_path)
        if os.path.isfile(candidate):
            return candidate

        return os.path.abspath(rel_path)

    def _download_if_absent(self, url: str, target_path: str):
        if not url:
            return False
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        try:
            import urllib.request

            with urllib.request.urlopen(url) as response, open(target_path, "wb") as output:
                shutil.copyfileobj(response, output)
            return True
        except Exception as e:
            raise RuntimeError(f"TPSR 预训练权重下载失败: {url}, reason: {e}")

    def _ensure_e2e_model(self):
        requested_path = self._resolve_tpsr_path(self.params.get("symbolicregression_model_path"))
        target_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "tpsr",
            "symbolicregression",
            "weights",
            "model.pt",
        )
        if requested_path and os.path.isfile(requested_path):
            if os.path.abspath(requested_path) != os.path.abspath(target_path):
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                shutil.copy2(requested_path, target_path)
            return target_path

        download_url = self.params.get("symbolicregression_model_url")
        if not os.path.isfile(target_path):
            if not download_url:
                raise FileNotFoundError(
                    "未找到 e2e 预训练权重，且未配置 symbolicregression_model_url"
                )
            self._download_if_absent(download_url, target_path)
            return target_path

        return target_path

    def _resolve_nesymres_weights(self, cfg):
        model_path = self.params.get("nesymres_model_path")
        if not model_path:
            model_path = getattr(cfg, "model_path", None)

        candidates = []
        if model_path:
            candidates.append(model_path)
        candidates.extend(
            [
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "tpsr", "nesymres", "weights", "10MCompleted.ckpt"),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "tpsr", "nesymres", "weights", "10M.ckpt"),
            ]
        )

        for candidate in candidates:
            if not candidate:
                continue
            candidate = self._resolve_tpsr_path(candidate)
            if candidate and os.path.isfile(candidate):
                return candidate

        return None

    def _normalize_equation(self, expr):
        if expr is None:
            return ""

        text = str(expr)
        # 常见 token 到符号的替换（与 Symbolic SR 的表示统一）
        replacements = {
            "add": "+",
            "mul": "*",
            "sub": "-",
            "pow": "**",
            "inv": "1/",
        }
        for op, op_target in replacements.items():
            text = re.sub(rf"\b{op}\b", op_target, text)
        # 兼容某些形如 pow2/pow3 写法
        text = re.sub(r"\bpow2\b", "**2", text)
        text = re.sub(r"\bpow3\b", "**3", text)
        text = re.sub(r"\bpow4\b", "**4", text)
        return text

    def _normalize_equation_list(self, trees):
        equations = []
        for tree in trees:
            if tree is None:
                continue
            equations.append(self._normalize_equation(tree))
        return equations

    def __getstate__(self):
        state = self.__dict__.copy()
        state["model"] = None
        state["_predict_fn"] = None
        state["_backend_params"] = {
            "backend": getattr(self._backend_params, "get", lambda k, d=None: None)("backend"),
        }
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.model = None
        self._backend_params = dict(
            (k, v) for k, v in (self._backend_params or {}).items() if k == "backend"
        )
        if not self.all_trees:
            self.all_trees = []
        self.all_trees = self._normalize_equation_list(self.all_trees)
        if self.best_tree is None:
            self.best_tree = self.all_trees[0] if self.all_trees else None
        if self.best_tree is not None:
            self.best_tree = self._normalize_equation(self.best_tree)
        self.all_trees = list(self.all_trees or [])
        n_features = getattr(self, "_n_features", None)
        var_names = list(getattr(self, "_predict_variable_names", []) or [])
        if not var_names and n_features:
            var_names = [f"x_{i}" for i in range(int(n_features))]
        self._predict_fn = None
        if self.best_tree and var_names:
            self._predict_fn = self._build_predictor_from_expression(self.best_tree, var_names)

    def _build_predictor_from_expression(self, expr_text: str, variable_names):
        if not expr_text:
            return None

        expr_text = self._normalize_equation(expr_text)
        try:
            import sympy as sp

            expr = sp.sympify(expr_text)
            symbols = [sp.Symbol(v) for v in variable_names]
            fn = sp.lambdify(symbols, expr, modules="numpy")
            self._backend_params["predict_symbols"] = symbols
            self._backend_params["predict_expr"] = expr

            def _predict(X):
                X_arr = np.asarray(X)
                if X_arr.ndim == 1:
                    X_arr = X_arr.reshape(1, -1)
                feature_count = X_arr.shape[1]
                available_args = []
                for idx, sym in enumerate(symbols):
                    if idx < feature_count:
                        available_args.append(X_arr[:, idx])
                    else:
                        # 缺失的变量补 0，避免 NeSymReS 方程维度不匹配时直接报错
                        available_args.append(np.zeros(X_arr.shape[0], dtype=float))
                outputs = fn(*available_args)
                outputs_arr = np.asarray(outputs, dtype=float)
                if outputs_arr.ndim == 0:
                    outputs_arr = np.full((X_arr.shape[0],), float(outputs_arr))
                elif outputs_arr.ndim >= 1 and outputs_arr.size == 1:
                    outputs_arr = np.full((X_arr.shape[0],), float(outputs_arr.reshape(-1)[0]))
                outputs_arr = outputs_arr.reshape(-1)
                if outputs_arr.shape[0] != X_arr.shape[0]:
                    outputs_arr = np.full((X_arr.shape[0],), float(outputs_arr[0]))
                return outputs_arr

            return _predict

        except Exception as e:
            print(f"TPSR 解析表达式失败，无法生成在线预测函数: {str(e)}")
            return None

    def _fit_e2e(self, X, y, equation_env, args, samples):
        self._ensure_e2e_model()

        from symbolicregression.e2e_model import Transformer
        from symbolicregression.e2e_model import pred_for_sample_no_refine, refine_for_sample
        from dyna_gym.agents.uct import UCT
        from dyna_gym.agents.mcts import update_root
        from rl_env import RLEnv
        from default_pi import E2EHeuristic

        # 创建 TPSR 主体模型
        model = Transformer(params=args, env=equation_env, samples=samples)
        model.to(args.device)

        # 创建 RL 环境
        rl_env = RLEnv(
            params=args,
            samples=samples,
            equation_env=equation_env,
            model=model,
        )

        # 创建 TPSR planner
        dp = E2EHeuristic(
            equation_env=equation_env,
            rl_env=rl_env,
            model=model,
            k=args.width,
            num_beams=args.num_beams,
            horizon=args.horizon,
            device=args.device,
            use_seq_cache=not args.no_seq_cache,
            use_prefix_cache=not args.no_prefix_cache,
            length_penalty=args.beam_length_penalty if hasattr(args, "beam_length_penalty") else 1.0,
            train_value_mode=args.train_value if hasattr(args, "train_value") else False,
            debug=args.debug,
        )

        # 创建 UCT 代理
        agent = UCT(
            action_space=[],
            gamma=1.0,
            ucb_constant=args.ucb_constant if hasattr(args, "ucb_constant") else 1.0,
            horizon=args.horizon,
            rollouts=args.rollout,
            dp=dp,
            width=args.width,
            reuse_tree=True,
            alg=args.uct_alg if hasattr(args, "uct_alg") else "uct",
            ucb_base=args.ucb_base if hasattr(args, "ucb_base") else 4,
        )

        # 运行搜索
        done = False
        s = rl_env.state
        for _ in range(args.horizon):
            if done or len(s) >= args.horizon:
                break
            act = agent.act(rl_env, done)
            s, _, done, _ = rl_env.step(act)
            update_root(agent, act, s)
            dp.update_cache(s)

        # `s` 可能只是搜索到当前 horizon 的中间前缀，并不保证是完整程序。
        # 优先从 default policy 已经生成的完整候选中选最优者；只有在搜索确实完成时，才回退到 `s`。
        candidate_sequences = []
        for seq in getattr(dp, "candidate_programs", []) or []:
            if seq is None:
                continue
            candidate_sequences.append(seq)
        if done and s is not None:
            candidate_sequences.append(s)

        if not candidate_sequences:
            raise RuntimeError("TPSR 搜索阶段未生成任何候选程序")

        def _reward_of(seq):
            try:
                return float(rl_env.get_reward(seq, mode="test"))
            except Exception:
                return float("-inf")

        best_sequence = max(candidate_sequences, key=_reward_of)

        self.all_trees = []

        # TPSR 的核心结果来自搜索得到的完整候选程序，而不是重新跑一遍预训练 E2E 回归器。
        # 这里优先取 “MCTS + refinement” 的表达式；若 refinement 失败，则退回到 no-ref 结果。
        try:
            _, refined_expr, refined_trees = refine_for_sample(
                args,
                model,
                equation_env,
                best_sequence,
                samples["x_to_fit"],
                samples["y_to_fit"],
            )
        except Exception:
            refined_expr, refined_trees = None, []

        try:
            _, raw_expr, raw_trees = pred_for_sample_no_refine(
                model,
                equation_env,
                best_sequence,
                samples["x_to_fit"],
            )
        except Exception:
            raw_expr, raw_trees = None, []

        candidate_exprs = []
        for expr in (refined_expr, raw_expr):
            if isinstance(expr, str) and expr.strip():
                candidate_exprs.append(expr)

        for tree_group in (refined_trees, raw_trees):
            if not isinstance(tree_group, list):
                continue
            for tree in tree_group:
                if tree is None:
                    continue
                text = str(tree.infix() if hasattr(tree, "infix") else tree).strip()
                if text:
                    candidate_exprs.append(text)

        # 去重并保留顺序
        seen = set()
        self.all_trees = []
        for expr in candidate_exprs:
            normalized = self._normalize_equation(expr)
            if normalized not in seen:
                seen.add(normalized)
                self.all_trees.append(normalized)

        if not self.all_trees:
            raise RuntimeError("TPSR 未生成任何有效候选方程")

        self.best_tree = self.all_trees[0]
        self.model = model
        self._n_features = X.shape[1]
        self._predict_variable_names = [f"x_{i}" for i in range(self._n_features)]
        self._backend_params["backend"] = "e2e"
        self._backend_params["search_state"] = best_sequence
        self._predict_fn = self._build_predictor_from_expression(
            self.best_tree, self._predict_variable_names
        )

    def _resolve_nesymres_resources(self, cfg):
        eq_setting_path = self.params.get("nesymres_eq_setting_path", "")
        cfg_path = self.params.get("nesymres_cfg_path", "")
        if not eq_setting_path:
            raise FileNotFoundError("NeSymReS 方程配置文件未设置：nesymres_eq_setting_path")
        if not cfg_path:
            raise FileNotFoundError("NeSymReS 配置文件未设置：nesymres_cfg_path")

        eq_setting_path = self._resolve_tpsr_path(eq_setting_path)
        cfg_path = self._resolve_tpsr_path(cfg_path)

        if not os.path.isfile(eq_setting_path):
            raise FileNotFoundError(f"未找到 NeSymReS 方程配置文件: {eq_setting_path}")
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(f"未找到 NeSymReS 配置文件: {cfg_path}")

        model_path = self._resolve_nesymres_weights(cfg)
        if not model_path:
            raise FileNotFoundError(
                "未找到 NeSymReS 预训练权重。建议设置 nesymres_model_path，或在 `nesymres/weights/` 下放置 ckpt 文件"
            )

        return eq_setting_path, cfg_path, model_path

    def _fit_nesymres(self, X, y, args, samples):
        import json
        from functools import partial
        from reward import compute_reward_nesymres
        from nesymres.src.nesymres.architectures.model import Model
        from nesymres.dclasses import FitParams, BFGSParams
        from dyna_gym.agents.uct import UCT
        from dyna_gym.agents.mcts import update_root
        from rl_env import RLEnv
        from default_pi import NesymresHeuristic
        import omegaconf

        cfg_path = self._resolve_tpsr_path(self.params.get("nesymres_cfg_path"))
        if not cfg_path or not os.path.isfile(cfg_path):
            raise FileNotFoundError(f"NeSymReS 配置文件不存在: {cfg_path}")
        cfg = omegaconf.OmegaConf.load(cfg_path)
        eq_setting_path = self._resolve_tpsr_path(self.params.get("nesymres_eq_setting_path"))
        eq_setting_path, _, weight_path = self._resolve_nesymres_resources(cfg)
        with open(eq_setting_path, "r", encoding="utf-8") as f:
            eq_setting = json.load(f)

        cfg_obj = cfg
        try:
            if getattr(cfg, "model", None) is not None:
                cfg_obj = cfg.model
        except Exception:
            cfg_obj = cfg
        self._set_parser_attr(cfg_obj, "cfg", cfg)
        bfgs_cfg = BFGSParams(
            activated=cfg.inference.bfgs.activated,
            n_restarts=cfg.inference.bfgs.n_restarts,
            add_coefficients_if_not_existing=cfg.inference.bfgs.add_coefficients_if_not_existing,
            normalization_o=cfg.inference.bfgs.normalization_o,
            idx_remove=cfg.inference.bfgs.idx_remove,
            normalization_type=cfg.inference.bfgs.normalization_type,
            stop_time=cfg.inference.bfgs.stop_time,
        )

        fit_params = FitParams(
            word2id=eq_setting["word2id"],
            id2word={int(k): v for k, v in eq_setting["id2word"].items()},
            una_ops=eq_setting["una_ops"],
            bin_ops=eq_setting["bin_ops"],
            total_variables=list(eq_setting["total_variables"]),
            total_coefficients=list(eq_setting["total_coefficients"]),
            rewrite_functions=list(eq_setting["rewrite_functions"]),
            bfgs=bfgs_cfg,
            beam_size=cfg.inference.beam_size if hasattr(cfg, "inference") else 5,
        )

        cfg.model_path = weight_path
        try:
            from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
            import torch.serialization

            torch.serialization.add_safe_globals([ModelCheckpoint])
        except Exception:
            pass
        model = Model.load_from_checkpoint(weight_path, cfg=cfg.architecture)
        model.eval()
        if torch.cuda.is_available() and not args.cpu:
            model.cuda()

        fitfunc = partial(model.fitfunc, cfg_params=fit_params)
        y_flat = np.asarray(y).reshape(-1)
        baseline_output = fitfunc(X, y_flat)
        all_preds = baseline_output.get("all_bfgs_preds", [])
        best_preds = baseline_output.get("best_bfgs_preds", [])
        baseline_expr = best_preds[0] if best_preds else (all_preds[0] if all_preds else None)

        rl_env = RLEnv(
            params=args,
            samples=samples,
            model=model,
            cfg_params=fit_params,
        )
        model.to_encode(X, y_flat, cfg_params=fit_params)

        dp = NesymresHeuristic(
            rl_env=rl_env,
            model=model,
            k=args.width,
            num_beams=args.num_beams,
            horizon=args.horizon,
            device=args.device,
            use_seq_cache=not args.no_seq_cache,
            use_prefix_cache=not args.no_prefix_cache,
            length_penalty=args.beam_length_penalty if hasattr(args, "beam_length_penalty") else 1.0,
            cfg_params=fit_params,
            train_value_mode=args.train_value if hasattr(args, "train_value") else False,
            debug=args.debug,
        )

        agent = UCT(
            action_space=[],
            gamma=1.0,
            ucb_constant=args.ucb_constant if hasattr(args, "ucb_constant") else 1.0,
            horizon=args.horizon,
            rollouts=args.rollout,
            dp=dp,
            width=args.width,
            reuse_tree=True,
            alg=args.uct_alg if hasattr(args, "uct_alg") else "uct",
            ucb_base=args.ucb_base if hasattr(args, "ucb_base") else 4,
        )

        done = False
        s = rl_env.state
        for _ in range(200 if not getattr(args, "sample_only", False) else 1):
            if done or len(s) >= args.horizon:
                break
            act = agent.act(rl_env, done)
            s, _, done, _ = rl_env.step(act)
            update_root(agent, act, s)
            dp.update_cache(s)

        _, reward_mcts, pred_str = compute_reward_nesymres(model.X, model.y, s, fit_params)
        mcts_expr = pred_str or baseline_expr
        if reward_mcts is None:
            print("NeSymReS 奖励无法计算，回退到预训练候选表达式")

        self.best_tree = mcts_expr
        self.all_trees = self._normalize_equation_list(all_preds)
        if baseline_expr and baseline_expr not in self.all_trees:
            self.all_trees.append(self._normalize_equation(baseline_expr))

        if mcts_expr is None and self.all_trees:
            mcts_expr = self.all_trees[0]

        self.model = model
        self._n_features = X.shape[1]
        self._predict_variable_names = list(eq_setting["total_variables"])[: self._n_features]
        self._predict_fn = self._build_predictor_from_expression(
            mcts_expr, self._predict_variable_names
        )
        self._backend_params["backend"] = "nesymres"
        self._backend_params["search_state"] = s
        self._backend_params["best_expr"] = mcts_expr
        self._backend_params["eq_variables"] = eq_setting["total_variables"]

    def fit(self, X, y):
        """
        训练 TPSR 模型。

        参数:
            X: 特征矩阵，形状 (n_samples, n_features)
            y: 目标向量，形状 (n_samples,) 或 (n_samples, 1)
        """
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        temp_dir = None
        original_cwd = None
        tpsr_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tpsr")
        inserted_paths = []

        # 导入必要的模块
        try:
            for import_path in self._runtime_import_paths():
                if os.path.isdir(import_path) and import_path not in sys.path:
                    sys.path.insert(0, import_path)
                    inserted_paths.append(import_path)
            original_cwd = os.getcwd()
            os.chdir(tpsr_dir)

            temp_dir = tempfile.mkdtemp(prefix="tpsr_")

            from symbolicregression.envs import build_env
            from parsers import get_parser

            parser = get_parser()
            args = parser.parse_args([])

            # 注入用户参数（兼容 parser 版本差异）
            self._set_parser_attr(args, "backbone_model", self.params.get("backbone_model", "e2e"))
            self._set_parser_attr(args, "beam_size", int(self.params.get("beam_size", 10)))
            self._set_parser_attr(args, "beam_type", self.params.get("beam_type", "sampling"))
            self._set_parser_attr(args, "no_seq_cache", bool(self.params.get("no_seq_cache", False)))
            self._set_parser_attr(args, "no_prefix_cache", bool(self.params.get("no_prefix_cache", False)))
            self._set_parser_attr(args, "width", int(self.params.get("width", 3)))
            self._set_parser_attr(args, "num_beams", int(self.params.get("num_beams", 1)))
            self._set_parser_attr(args, "rollout", int(self.params.get("rollout", 3)))
            self._set_parser_attr(args, "horizon", int(self.params.get("horizon", 200)))
            self._set_parser_attr(args, "seed", int(self.params.get("seed", 23)))
            self._set_parser_attr(args, "debug", bool(self.params.get("debug", False)))
            self._set_parser_attr(args, "beam_length_penalty", float(self.params.get("beam_length_penalty", 1.0)))
            self._set_parser_attr(args, "train_value", bool(self.params.get("train_value", False)))
            self._set_parser_attr(args, "ucb_constant", float(self.params.get("ucb_constant", 1.0)))
            self._set_parser_attr(args, "uct_alg", self.params.get("uct_alg", "uct"))
            self._set_parser_attr(args, "ucb_base", float(self.params.get("ucb_base", 4.0)))
            self._set_parser_attr(args, "cpu", bool(self.params.get("cpu", False)))
            self._set_parser_attr(args, "lam", float(self.params.get("lam", 0.1)))
            self._set_parser_attr(args, "max_input_points", int(self.params.get("max_input_points", 10000)))
            self._set_parser_attr(args, "max_number_bags", int(self.params.get("max_number_bags", 10)))
            self._set_parser_attr(args, "rescale", bool(self.params.get("rescale", True)))
            self._set_parser_attr(args, "sample_only", bool(self.params.get("sample_only", False)))

            args.cpu = bool(self.params.get("cpu", False))
            args.device = torch.device("cpu") if args.cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")

            equation_env = build_env(args)

            samples = {
                "x_to_fit": [np.asarray(X)],
                "y_to_fit": [np.asarray(y)],
                "x_to_pred": [np.asarray(X)],
                "y_to_pred": [np.asarray(y)],
            }

            backbone_model = self.params.get("backbone_model", "e2e").lower()
            if backbone_model == "e2e":
                self._fit_e2e(X, y, equation_env, args, samples)
            elif backbone_model == "nesymres":
                self._fit_nesymres(np.asarray(X), np.asarray(y), args, samples)
            else:
                raise ValueError(f"不支持的 backbone_model: {backbone_model}")

            # 统一化所有树列表（若对象为 SymbolicTree，后续再做输出转换）
            self.all_trees = self._normalize_equation_list(self.all_trees)
            if self.best_tree is not None:
                self.best_tree = self._normalize_equation(self.best_tree)
            return self

        except Exception:
            import traceback

            traceback.print_exc()
            raise
        finally:
            if original_cwd is not None:
                os.chdir(original_cwd)
            for import_path in reversed(inserted_paths):
                if import_path in sys.path:
                    try:
                        sys.path.remove(import_path)
                    except ValueError:
                        pass
            if temp_dir is not None:
                try:
                    import shutil

                    shutil.rmtree(temp_dir)
                except Exception:
                    pass

    def predict(self, X):
        """使用模型进行预测"""
        if self._predict_fn is not None:
            return self._predict_fn(X)

        if hasattr(self.model, "predict"):
            return self.model.predict(X, refinement_type="BFGS")

        raise ValueError("当前后端不支持 predict，需要重新检查 fit 是否已完成并能解析表达式")

    def get_optimal_equation(self):
        """获取模型学习到的最优符号方程"""
        if self.best_tree is None:
            raise ValueError("未找到可用方程")
        return self._normalize_equation(self.best_tree)

    def get_total_equations(self):
        """获取模型学习到的所有候选符号方程"""
        if self.best_tree is None and not self.all_trees:
            raise ValueError("模型尚未训练，请先调用fit方法")
        return list(self.all_trees or [])

    def export_canonical_symbolic_program(self):
        if self.best_tree is None:
            raise ValueError("未找到可用方程")
        return normalize_tpsr_artifact(self.get_optimal_equation())


if __name__ == "__main__":
    import numpy as np

    X = np.random.rand(80, 2)
    y = 2.0 * X[:, 0] - 0.5 * X[:, 1] + 0.02 * np.random.randn(80)

    reg = TPSRRegressor(
        backbone_model="e2e",
        beam_size=6,
        width=3,
        num_beams=1,
        rollout=2,
    )
    reg.fit(X, y)
    print("eq:", reg.get_optimal_equation())
    print("pred:", reg.predict(X[:3]))
