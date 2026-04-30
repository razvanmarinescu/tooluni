PYTHON ?= /Users/razvan/research/evals/tooluni/.venv/bin/python
RUN_ID ?= 00013
MODEL ?= claude-opus-4-7 claude-sonnet-4-6
WORKERS ?= 3
BENCHLING_DATASET ?= benchling/External Problems from Ruslan and Mihir.json
BENCHLING_TIERS ?= internal_only
BENCHLING_RUN_SUFFIX ?= _benchling
GENETIC_BENCHMARK_V1_DATASET ?= genetic_benchmark_v1/48-submissions-clean.json
GENETIC_BENCHMARK_V1_TIERS ?= internal_only

.PHONY: regenerate-summary regenerate-summary-00013 benchling-eval benchling-smoke-internal genetic-benchmark-v1-eval genetic-benchmark-v1-smoke-internal

regenerate-summary:
	$(PYTHON) scripts/regenerate_summary.py --run-id $(RUN_ID)

regenerate-summary-00013: RUN_ID := 00013
regenerate-summary-00013: regenerate-summary

benchling-eval:
	$(PYTHON) scripts/run_eval.py --dataset "$(BENCHLING_DATASET)" --run-name-suffix $(BENCHLING_RUN_SUFFIX) --tiers $(BENCHLING_TIERS) --models $(MODEL) --workers $(WORKERS)

benchling-smoke-internal:
	$(PYTHON) scripts/run_eval.py --dataset "$(BENCHLING_DATASET)" --run-name $(RUN_ID) --run-name-suffix $(BENCHLING_RUN_SUFFIX) --start-index 1 --end-index 1 --tiers internal_only --models $(MODEL) --workers $(WORKERS)

genetic-benchmark-v1-eval:
	$(PYTHON) scripts/run_eval.py --dataset "$(GENETIC_BENCHMARK_V1_DATASET)" --tiers $(GENETIC_BENCHMARK_V1_TIERS) --models $(MODEL) --workers $(WORKERS)

genetic-benchmark-v1-smoke-internal:
	$(PYTHON) scripts/run_eval.py --dataset "$(GENETIC_BENCHMARK_V1_DATASET)" --run-name $(RUN_ID) --start-index 1 --end-index 1 --tiers internal_only --models $(MODEL) --workers $(WORKERS)