
# PIGSKIN - PIG SKull stripping with synthetic Images and Neural network

---

## 📦 Requirements

- Linux system
- **Apptainer** (formerly Singularity) installed
---

## 📥 Installation

### 1. Clone the repository

```bash
git clone https://github.com/dadashkarimi/pigskin/tree/main
cd pigskin
````

---

### 2. Download the Apptainer image

Download the `.sif` file from the following link:

```bash 
https://upenn.box.com/s/42bwvh1urtogkleubsy86034uczc76gt?download=1
```

---

### 3. Move the `.sif` file into the project directory

After downloading, move the file from your `Downloads` folder into the `pigskin` directory:

```bash
mv ~/Downloads/pigskin.sif $PWD/pigskin
```

---

## ▶️ Usage

Run the Apptainer image using the following command:

```bash
cd $PWD/pigskin

apptainer exec --nv pigskin.sif python pigskin.py \
  --input input.nii.gz \
  --output output.nii.gz
```

### Notes

* No manual bind mounts are required.
* Replace `input.nii.gz` and `output.nii.gz` with your actual file paths.
* The `--nv` flag enables GPU support.

---

## 📁 Example Directory Structure

```
pigskin/
│── pigskin.py
│── pigskin.sif
│── README.md
```

---

## ❓ Troubleshooting

* Ensure Apptainer is installed and accessible from the command line.
* Make sure the input file exists and is in `.nii.gz` format.
* Confirm GPU drivers are correctly installed if using `--nv`.

