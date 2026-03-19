
# PIGSKIN - PIG SKull stripping with synthetic Images and Neural network

---

## 📦 Requirements

- **Apptainer** (formerly Singularity) installed for running PIGSKIN using apptainer sif file
---

## 📥 Installation

### 1. Download the Apptainer image

Download the `.sif` file from the following link:

```bash
https://upenn.box.com/s/3z226jfla9a39qkzdbs7nv4f1sawwfyx
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

cd $PWD/pigskin

apptainer run --no-home \
  --bind {OUTPUT_PATH}:/output \
  --bind {INPUT_PATH}:/input \
  $PWD/pigskin.sif \
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

