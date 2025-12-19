#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --cpus-per-task=1
#SBATCH --mem=180G
#SBATCH --partition=ai
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0-18:30:00          # Set expected wall time
#SBATCH --job-name="pigskin_batch"
#SBATCH --output="pigskin_batch.out"

# Activate the desired Conda environment
source ~/.bashrc                  # Make sure Conda is initialized in your shell
conda activate tf-gpu-jk

module load cuda/11.8

# Update PYTHONPATH (same as your single-image script)
export PYTHONPATH=$PYTHONPATH:/cbica/home/dadashkj/pystrum
export PYTHONPATH=$PYTHONPATH:/cbica/home/dadashkj/neurite
export PYTHONPATH=$PYTHONPATH:/cbica/home/dadashkj/voxelmorph
export PYTHONPATH=$PYTHONPATH:/cbica/home/dadashkj/neurite-sandbox
export PYTHONPATH=$PYTHONPATH:/cbica/home/dadashkj/voxelmmorph-sandbox

# ---- Paths ----
INPUT_ROOT="/cbica/home/parkerwi/Pig/PROCEED/T1"
OUTPUT_ROOT="Drew/T1_masks"

mkdir -p "${OUTPUT_ROOT}"

# Optional: cd to the directory where pigskin.py lives
# cd /cbica/home/dadashkj/pigskin   # <--- update if needed

echo "Starting batch pigskin run..."
echo "Input root:  ${INPUT_ROOT}"
echo "Output root: ${OUTPUT_ROOT}"

# Loop over all .nii.gz files that are one level under INPUT_ROOT
for img in "${INPUT_ROOT}"/*/*.nii.gz; do
    # Skip if glob finds nothing
    [ -e "$img" ] || continue

    subj_dir="$(basename "$(dirname "$img")")"           # e.g. 2016-12_post
    img_name="$(basename "$img" .nii.gz)"               # e.g. image

    out_dir="${OUTPUT_ROOT}/${subj_dir}"
    mkdir -p "${out_dir}"

    out_file="${out_dir}/${img_name}_mask.nii.gz"

    echo "Running pigskin on:"
    echo "  INPUT : ${img}"
    echo "  OUTPUT: ${out_file}"

    python pigskin.py \
        --input  "${img}" \
        --output "${out_file}"
done

echo "All done."

