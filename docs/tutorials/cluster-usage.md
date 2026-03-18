# Cluster Usage

For large-scale analyses (many perturbed TFs), the workflow is designed to run
as a SLURM job array where each array task processes one TF.

## Lookup File

Create a TSV with one row per TF. Only the first two columns are required:

```
# response_file              perturbed_tf    suffix (opt)    predictors_file (opt)
/data/response/GAT2.csv      GAT2
/data/response/pTF1.csv      pTF1            ypd1_run2       /data/predictors_v2.csv
```

- **`response_file`**: path to the response CSV for this TF
- **`perturbed_tf`**: the TF name (must match a column in both files)
- **`suffix`** (optional): appended to the output subdirectory name
- **`predictors_file`** (optional): overrides the global `--predictors_file`
  for this row

## Job Script

Save the following as `perturbation_binding_modeling.sh`. It is submitted as a
SLURM array; each task reads one line from the lookup file.

```bash
#!/bin/bash
#SBATCH --output=logs/%A_%a.out
#SBATCH --open-mode=append
#SBATCH --cpus-per-task=4
#SBATCH --mem=1G

set -euo pipefail

log() { echo -e "[INFO] $*"; }

print_usage() {
    cat <<EOF
Usage:
  sbatch --array=1-N perturbation_binding_modeling.sh \
      --venv=PATH_TO_VENV_ACTIVATE \
      --lookup=LOOKUP_FILE \
      --output_dir=OUTPUT_DIR \
      [--predictors=PREDICTORS_CSV] \
      [--all_data_ci_level=98.0] \
      [--topn_ci_level=90.0] \
      [--n_bootstraps=1000] \
      [--random_state=42] \
      [--stage3_lasso] \
      [--stage3_lasso_topn] \
      [--stage3_lassocv_bootstrap] \
      [--iterative_dropout] \
      [--scale_by_std] \
      [--bins=0,64,512,np.inf] \
      [--exclude_model_variables=VARNAME]

NOTE: --predictors is required unless provided in the lookup file.
EOF
    exit 1
}

parse_lookup_line() {
    local line="$1"
    local num_fields
    num_fields=$(echo "$line" | awk -F'\t' '{print NF}')
    RESPONSE_FILE=$(echo "$line" | awk -F'\t' '{print $1}')
    PERTURBED_TF=$(echo  "$line" | awk -F'\t' '{print $2}')
    SUFFIX=""
    PREDICTORS_FILE_OVERRIDE=""
    [[ "$num_fields" -ge 3 ]] && SUFFIX=$(echo "$line" | awk -F'\t' '{print $3}')
    [[ "$num_fields" -ge 4 ]] && \
        PREDICTORS_FILE_OVERRIDE=$(echo "$line" | awk -F'\t' '{print $4}')
}

[[ "$#" -eq 0 || "$1" == "--help" || "$1" == "-h" ]] && print_usage

# Defaults
MAX_ITER=10000
N_BOOTSTRAPS=1000
ALL_DATA_CI_LEVEL=98.0
TOPN_CI_LEVEL=90.0
VENV="" LOOKUP="" PREDICTORS_FILE="" OUTPUT_OUTER=""
RANDOM_STATE="" STAGE3_LASSO_FLAG="" STAGE3_LASSO_TOPN_FLAG=""
STAGE3_LASSOCV_BOOTSTRAP_FLAG="" BINS_FLAG="" EXCLUDE_MODEL_VARIABLES=""
ADD_MODEL_VARIABLES="" SCALE_BY_STD="" ITERATIVE_DROPOUT=""
STABILIZATION_CI_START=""

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --venv=*)              VENV="${1#*=}" ;;
        --lookup=*)            LOOKUP="${1#*=}" ;;
        --predictors=*)        PREDICTORS_FILE="${1#*=}" ;;
        --output_dir=*)        OUTPUT_OUTER="${1#*=}" ;;
        --all_data_ci_level=*) ALL_DATA_CI_LEVEL="${1#*=}" ;;
        --topn_ci_level=*)     TOPN_CI_LEVEL="${1#*=}" ;;
        --n_bootstraps=*)      N_BOOTSTRAPS="${1#*=}" ;;
        --random_state=*)      RANDOM_STATE="--random_state ${1#*=}" ;;
        --stage3_lasso)        STAGE3_LASSO_FLAG="--stage3_lasso" ;;
        --stage3_lasso_topn)   STAGE3_LASSO_TOPN_FLAG="--stage3_lasso_topn" ;;
        --stage3_lassocv_bootstrap) \
                               STAGE3_LASSOCV_BOOTSTRAP_FLAG="--stage3_lassocv_bootstrap" ;;
        --bins=*)              BINS_FLAG="--bins ${1#*=}" ;;
        --exclude_model_variables=*) \
                               EXCLUDE_MODEL_VARIABLES="${1#*=}" ;;
        --add_model_variables=*) \
                               ADD_MODEL_VARIABLES="${1#*=}" ;;
        --scale_by_std)        SCALE_BY_STD="--scale_by_std" ;;
        --iterative_dropout)   ITERATIVE_DROPOUT="--iterative_dropout" ;;
        --stabilization_ci_start=*) \
                               STABILIZATION_CI_START="--stabilization_ci_start ${1#*=}" ;;
        -h|--help) print_usage ;;
        *) echo "[ERROR] Unknown argument: $1"; print_usage ;;
    esac
    shift
done

log "Activating environment..."
source "$VENV"

LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LOOKUP")
parse_lookup_line "$LINE"

[[ -n "${PREDICTORS_FILE_OVERRIDE}" ]] && PREDICTORS_FILE="$PREDICTORS_FILE_OVERRIDE"

[[ -z "$VENV" || -z "$LOOKUP" || -z "$PREDICTORS_FILE" || -z "$OUTPUT_OUTER" ]] && {
    echo "[ERROR] Missing required argument."; print_usage
}

[[ -n "$SUFFIX" ]] && OUTPUT_SUFFIX_FLAG="--output_suffix $SUFFIX" \
                   || OUTPUT_SUFFIX_FLAG=""
[[ -n "$EXCLUDE_MODEL_VARIABLES" ]] \
    && EXCLUDE_FLAG="--exclude_model_variables=$EXCLUDE_MODEL_VARIABLES" \
    || EXCLUDE_FLAG=""
[[ -n "$ADD_MODEL_VARIABLES" ]] \
    && ADD_VAR_FLAG="--add_model_variables=$ADD_MODEL_VARIABLES" \
    || ADD_VAR_FLAG=""

CMD=(
    python -m tfbpmodeling
    --response_file  "$RESPONSE_FILE"
    --predictors_file "$PREDICTORS_FILE"
    --perturbed_tf   "$PERTURBED_TF"
    --n_bootstraps   "$N_BOOTSTRAPS"
    --output_dir     "$OUTPUT_OUTER"
    --max_iter       "$MAX_ITER"
    --all_data_ci_level "$ALL_DATA_CI_LEVEL"
    --topn_ci_level  "$TOPN_CI_LEVEL"
    $STAGE3_LASSO_FLAG
    $STAGE3_LASSO_TOPN_FLAG
    $STAGE3_LASSOCV_BOOTSTRAP_FLAG
    $RANDOM_STATE
    $OUTPUT_SUFFIX_FLAG
    $BINS_FLAG
    $EXCLUDE_FLAG
    $ADD_VAR_FLAG
    $ITERATIVE_DROPOUT
    $SCALE_BY_STD
    $STABILIZATION_CI_START
)

log "Launching:"
printf "%q " "${CMD[@]}"
echo
"${CMD[@]}"
```

## Submission Script

Save the following as `submit_jobs.sh` alongside the job script:

```bash
#!/bin/bash
# Usage:
#   ./submit_jobs.sh OUTPUT_DIR LOOKUP VENV_ACTIVATE [ARRAY_LENGTH]
#
# ARRAY_LENGTH defaults to the number of lines in LOOKUP.

OUTPUT_DIR=$1
LOOKUP=$2
ACTIVATE=$3
ARRAY_LENGTH=${4:-$(wc -l < "$LOOKUP")}

sbatch --array=1-${ARRAY_LENGTH} perturbation_binding_modeling.sh \
    --venv="$ACTIVATE" \
    --lookup="$LOOKUP" \
    --output_dir="$OUTPUT_DIR" \
    --bins="0,64,512,np.inf" \
    --scale_by_std \
    --stage3_lasso \
    --stage3_lasso_topn \
    --stage3_lassocv_bootstrap \
    --iterative_dropout \
    --random_state=42
```

Submit with:

```bash
./submit_jobs.sh /path/to/results lookup.tsv /path/to/venv/bin/activate
```
