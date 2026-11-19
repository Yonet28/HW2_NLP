# PatchBench FR/EN

Code for *Lost in the Middle, or Lost in Translation? Disentangling Language, Token Length, and
Interference in Long-Context Retrieval* (CSE4057 course project, Hanyang University).

The benchmark asks a language model for the new value of one statistic in a long series of
synthetic video game patch notes. Every item is rendered with **identical content** in English and
French, so that the two languages can be compared pair by pair. Values are random, so answers cannot
come from memory.

## Environment

* Python 3.10 or later, one NVIDIA GPU with 16 GB (tested setting: Google Colab, free T4).
* `pip install -r requirements.txt`
* Models are loaded from Hugging Face in float16. For Llama 3.2 we use `unsloth/Llama-3.2-3B-Instruct`,
  an ungated copy of the official weights (`meta-llama/Llama-3.2-3B-Instruct` works too after
  accepting its license and running `huggingface-cli login`).

## Layout

| Path | Content |
|---|---|
| `patchbench/data.py` | Item generator, English and French templates (native and arrow), padding sentences, answer parsing and scoring |
| `patchbench/experiments.py` | The four experiments and their presets |
| `patchbench/run.py` | Runs one model, appends one JSON line per prompt to `results/<model>.jsonl` (resumable) |
| `patchbench/analyze.py` | Builds every table and figure from the JSONL files |
| `notebooks/run_colab.ipynb` | Colab notebook that runs everything, with results saved to Google Drive |
| `notebooks/run_kaggle.ipynb` | Kaggle notebook running two models in parallel on two T4 GPUs |

## Experiments

| Name | What varies | Used for |
|---|---|---|
| `main` | language × target patch (1, 4, 7, 10, 14, 17, 20) × distractors (0, 4) × template (native, arrow) | Table 1, Figure 1, Table 3, H1, H2 |
| `oracle` | only the target patch as context (both templates), and a closed-book condition | Table 2 |
| `length` | content-matched EN and FR, token-matched EN-long and FR-short, English +50%, and the fixed-token-budget baseline used in prior work | Table 2, Figure 2, H3 |
| `crosslang` | context language × question language | Table 2 |
| `promptfmt` | same items as `main` (no distractor), with the pilot prompt (no answer format) and/or without padding | Table 2, comparison with the pilot |

Presets (`--preset`): `smoke` (a few items, to check the setup), `lean` (used in the paper, sized for a
free T4), `standard` and `full` (larger, if compute allows). Sizes are in `patchbench/experiments.py`.

## Reproducing the paper

```bash
# 1. Run each model (seed 0; the Qwen models use the standard preset, Llama the lean preset,
#    whose items are the first items of the standard preset; Llama runs main and oracle only)
python -m patchbench.run --model Qwen/Qwen2.5-3B-Instruct   --preset standard --batch_size 2 --out results/
python -m patchbench.run --model Qwen/Qwen2.5-1.5B-Instruct --preset standard --batch_size 4 --out results/
python -m patchbench.run --model unsloth/Llama-3.2-3B-Instruct --preset lean --experiments main,oracle --batch_size 2 --out results/

# 2. Build all tables and figures
python -m patchbench.analyze --results results/ --out results/analysis/
```

| Output | File |
|---|---|
| Table 1 (accuracy and EN minus FR gap) | `results/analysis/table1_main.csv` / `.tex` |
| Table 2 (oracle, template, token-length and baseline components) | `results/analysis/table2_decomposition.csv` / `.tex` |
| Table 3 (interference and previous-value errors) | `results/analysis/table3_interference.csv` |
| Parsing rates | `results/analysis/table_parsing.csv` |
| Figure 1 (accuracy by position) | `results/analysis/fig_position.pdf` |
| Figure 2 (decomposition of the gap) | `results/analysis/fig_decomposition.pdf` |
| Random failures | `results/analysis/examples_failures.md` |
| Hypothesis tests and verdicts | `results/analysis/summary.json` |

## Seeds and determinism

Items are generated from fixed seeds (`item_seed` in `experiments.py`), so every run sees the same
items. Decoding is greedy (`do_sample=False`), `--seed 0` sets the PyTorch seed, and the bootstrap in
`analyze.py` uses `numpy.random.default_rng(0)` with 2,000 resamples. Small numerical differences
across GPUs and library versions can still change a few greedy outputs.

## Offline test

`python -m patchbench.run --model fake --preset smoke --out /tmp/test` runs the whole pipeline with a
simulated model and a rough token counter, without a GPU or network access.
