export FSLOUTPUTTYPE=NIFTI_GZ 

for d in results/JAW*/; do
    rm "$d"/bet*.nii.gz "$d/tmp.nii.gz"
    bet2 "$d/image.nii" "$d/tmp.nii.gz" -m -f 0.2 -g 0
    mv "$d/tmp.nii.gz_mask.nii.gz" "$d/bet.nii.gz"
done

