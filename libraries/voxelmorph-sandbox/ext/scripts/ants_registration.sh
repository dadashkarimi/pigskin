# TODO: Malte, please complete this.

cmd = f''' \
    antsRegistration \
        --dimensionality 3 \
        --float 0 \
        --output [{d}/,{res},{d}/2-to-1.nii.gz] \
        --winsorize-image-intensities [0.005,0.995] \
        --use-histogram-matching {match_hist} \
        --initial-moving-transform [{fix},{mov},1] \
        \
        --transform Rigid[0.1] \
        --metric {metric} \
        --convergence [1000x500x250x100,1e-6,10] \
        --shrink-factors 8x4x2x1 \
        --smoothing-sigmas 3x2x1x0vox \
        \
        --transform Affine[0.1] \
        --metric {metric} \
        --convergence [1000x500x250x100,1e-6,10] \
        --shrink-factors 8x4x2x1 \
        --smoothing-sigmas 3x2x1x0vox ;

    mri_convert -odt uchar {res} {res} ;
    ConvertTransformFile 3 {d}/0GenericAffine.mat {lta} --hm --ras ;
    lta_convert --src {mov} --trg {fix} --inniftyreg {lta} \
        --outlta {lta} --ltavox2vox ;
    '''