# SDWPF paper protocol

This protocol keeps the final 20% of timestamps sealed until all model and
hyper-parameter choices are frozen.

## Log layout

SDWPF entry scripts automatically allocate one searchable directory per user
invocation:

```text
outputs/logs/SDWPF/
├── latest.txt
└── YYYYMMDD/
    ├── index.tsv
    ├── latest.txt
    └── 001_<task>_<key-parameters>/
        ├── log_meta.json
        ├── status.env
        ├── *.env
        ├── *.log
        ├── *summary.txt
        └── tb/
```

The three-digit prefix is the run number for that date. CV, final-training,
fold and ablation-suite scripts use one numbered parent directory and place
their fold/seed jobs below `runs/`. Existing callers may still override the
automatic layout with `SDWPF_LOG_DIR` or `SDWPF_LOG_FILE`.
Long visible parameter strings are shortened with a stable hash to stay below
common filesystem limits; `index.tsv`, `log_meta.json`, and the stage `.env`
files always retain the complete values.

Locate the newest run and its compact result with:

```bash
LATEST="$(cat outputs/logs/SDWPF/latest.txt)"
echo "outputs/logs/SDWPF/${LATEST}"
find "outputs/logs/SDWPF/${LATEST}" -maxdepth 3 -name '*summary.txt' -print
```

Browse every run created today with:

```bash
column -t -s $'\t' "outputs/logs/SDWPF/$(date +%Y%m%d)/index.tsv"
```

Before pushing or starting a server run, execute the fast regression suite:

```bash
python -m unittest discover -s tests -v
```

If the fixed CSV must be regenerated, keep the generated provenance JSON with
the dataset:

```bash
python scripts/fix_data.py \
  --input datasets/sdwpf_245days_v1.csv \
  --output datasets/sdwpf_fixed.csv
```

## 1. Validation-only cross-validation

```bash
NEW_MODULE_LEARNING_RATE=0.00001 \
bash scripts/train/SDWPF_paper_cv.sh
```

To confirm a selected setting only on the remaining predetermined seeds:

```bash
FOLDS='0 1 2' SEEDS='2025 2026' \
NEW_MODULE_LEARNING_RATE=0.00001 \
bash scripts/train/SDWPF_paper_cv.sh
```

`rolling_holdout` divides the 70%-80% interval into three disjoint expanding-
origin validation folds. It never evaluates the final 80%-100% holdout.

Run validation baselines with the same fold and horizon:

```bash
for fold in 0 1 2; do
  FOLD="$fold" SPLIT=rolling_holdout EVAL_SPLIT=val \
    bash scripts/eval/SDWPF_baselines.sh
done
```

For a matched pretraining checkpoint, run the controlled ablations with:

```bash
FOLD=0 SEED=2024 SPLIT=rolling_holdout \
PRETRAIN_RUN_ID='<matched_pretrain_run_id>' \
bash scripts/finetune/SDWPF_ablation_suite.sh
```

## 2. Freeze the configuration and train final seeds

Do not change architecture, loss, feature, optimizer, or cleaning settings
after this point.

```bash
NEW_MODULE_LEARNING_RATE=0.00001 \
bash scripts/train/SDWPF_paper_final.sh
```

This trains three predetermined seeds with `time_ratio` (70% train, 10%
validation, 20% sealed test) and prints the checkpoint-specific final command.

## 3. Evaluate the final test once per predetermined seed

Run each command printed by stage 2. The required shape is:

```bash
CONFIRM_FINAL_EVAL=1 SPLIT=time_ratio N_FOLDS=1 PRED_LEN=12 \
FINETUNE_CHECKPOINT='<checkpoint.pth>' \
bash scripts/eval/SDWPF_logged_final_eval.sh
```

Run the final baselines under the same split exactly once:

```bash
CONFIRM_FINAL_EVAL=1 SPLIT=time_ratio N_FOLDS=1 EVAL_SPLIT=test \
bash scripts/eval/SDWPF_baselines.sh
```
