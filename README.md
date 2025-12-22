# PIGSKIN Apptainer Container

This repository provides an Apptainer/Singularity container for running **PIGSKIN**.

---

## Build the container

From the directory containing `Singularity.def`, build the container image:

```bash
apptainer build pigskin.sif Singularity.def
``` 

This will create pigskin.sif in the current directory.

Repository location (CBICA example)

The PIGSKIN repository is located at:

/abc/home/user/pigskin


The container image is located at:

/abc/home/user/pigskin/pigskin.sif

Running the container
Step 1: Move to the cloned PIGSKIN repository

Before running the container, change to the cloned repository directory:

cd /abc/home/user/pigskin

Step 2: Execute the container

Example command:

apptainer exec --nv pigskin.sif python pigskin.py --input {file} --output {output_image}


Replace:

{file} with the input file path

{output_image} with the desired output image path

Example with real paths
apptainer exec --nv pigskin.sif \
  python pigskin.py \
  --input /path/to/input_file.nii.gz \
  --output /path/to/output_mask.nii.gz

Notes

No manual bind mounts are required. Apptainer automatically binds the current working directory.

The --nv flag enables NVIDIA GPU support.
If running on CPU only, omit this flag:

apptainer exec pigskin.sif python pigskin.py --input {file} --output {output_image}

Troubleshooting

Confirm Apptainer is available:

apptainer --version


If input files are not visible inside the container, ensure you are running the command from a directory that has access to those paths, or explicitly bind them:

apptainer exec --bind /path/to/data pigskin.sif ...
