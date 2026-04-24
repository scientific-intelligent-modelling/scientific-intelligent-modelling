# stressstrain 200-generation six-experiment bundle

This folder bundles the two available stressstrain 200-generation reruns across three tools:

- rerun200_llm_drsr_20260405_150543: drsr, llmsr, pysr
- rerun200_llm_drsr_20260406_111429: drsr, llmsr, pysr

Layout:

- summary.csv: compact metrics table for the six experiments.
- experiment_config.json: full bundle manifest with source paths and effective parameters.
- shared/dataset_config.json: common dataset contract.
- shared/llm.config.redacted.json: shared LLM config with credentials redacted.
- environment_and_best_formulas.md: human-readable environment, dependency, key-parameter, and best-formula record.
- environment_and_best_formulas.json: machine-readable environment, dependency, key-parameter, and best-formula record.
- runs/<batch>/<tool>/result.json: copied final result.
- runs/<batch>/<tool>/config.json: per-experiment reproducibility config extracted from result.json and wrapper defaults.
- runs/<batch>/<tool>/artifacts/: key available artifacts, such as specs, progress, top samples, and PySR hall_of_fame.
- runs/<batch>/<tool>/logs/: available stressstrain run logs for drsr/llmsr.

Notes:

- The original llm.config contained a credential, so api_key is intentionally REDACTED in this bundle.
- Full raw experiment directories are not fully duplicated here; their source paths are recorded in each config.json.
- PySR was rerun with current wrapper defaults and niterations=200.
