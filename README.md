
# PIGSKIN - PIG SKull stripping with synthetic Images and Neural network

---

## 📦 Requirements

- **Apptainer** (formerly Singularity) installed for running PIGSKIN using apptainer sif file
---

## 📥 Installation

### 1. Download the Apptainer image

Download the `.sif` file from the following link:

```bash 
[https://upenn.box.com/s/42bwvh1urtogkleubsy86034uczc76gt?download=1](https://upenn.box.com/s/o1h7kzlpslsnszyy7c2b88eiy65v7hgr)
```

---

### 2. Move the `.sif` file into the project directory

After downloading, move the file from your `Downloads` folder into the `pigskin` directory:

```bash
mkdir -p $PWD/pigskin
mv ~/Downloads/pigskin.sif $PWD/pigskin
```

---

## ▶️ Usage

Run the Apptainer image using the following command:

```bash
apptainer run --no-home \
  --bind {OUTPUT_PATH}:/output \
  --bind {INPUT_PATH}:/input \
  $PWD/pigskin-new.sif \
  -i /input/image.nii.gz \
  -o /output/output_image.nii.gz
```

### Notes

* Manual bind mounts are required for input and output.
* Replace `input.nii.gz` and `output.nii.gz` with your actual file paths.

---

## ❓ Troubleshooting

* Ensure Apptainer is installed and accessible from the command line.
* Make sure the input file exists and is in `.nii.gz` format.

