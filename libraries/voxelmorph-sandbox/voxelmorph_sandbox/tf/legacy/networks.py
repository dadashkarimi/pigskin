"""
Unsupported Networks for VoxelMorph (sandbox)

Please:
- only include networks here (rather than network parts)
- name your network very descriptively (e.g.) "VxmAffineDecomposeWarp" instead of "VxmAffineNew"
"""

# internal python imports
import warnings

# third party imports
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as KL
import tensorflow.keras.initializers as KI

# local (our) imports
import neurite as ne
import neurite_sandbox as nes
import voxelmorph as vxm
import voxelmorph_sandbox as vxms


class VxmAffineWithCorrection(ne.tf.modelio.LoadableModel):
    """
    VoxelMorph network for linear (affine) registration between two images.
    including nonlinear B0 correction

    Author: brf2
    """
    @ne.modelio.store_config_args
    def __init__(
            self, 
            inshape, 
            labels_in, 
            labels_out, 
            affine_nf=64, 
            gen_args={}, 
            unet_features=64, 
            unet_levels=5, 
            fill_value=None, 
            name='VxmAffineWithCorrection', 
            max_trans=15, 
            rescale_affine=[1, .1],
            max_rot=10, 
            min_B0_blur=1,
            max_B0_blur=20,
            max_B0_std=1,
            reserve_capacity=None, 
            affine_conv_per_level=1, 
            **kwargs):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            name: Model name. Default is 'vxm_affine'.
        """

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # input vectors that give B0 distortion direction
        B0_dir1 = KL.Input(shape=(3, 1), name='B0_dir1')
        B0_dir2 = KL.Input(shape=(3, 1), name='B0_dir2')

        gen_model = ne.models.labels_to_image(
            inshape,
            labels_in,
            labels_out,
            zero_background=1,    # have air be black
            id=0,
            num_chan=2,
            return_def=False,
            one_hot=False,
            **gen_args
        )

        # input a label map and use it to synthesize an image. Remove air
        label_input = KL.Input(shape=inshape + (1,), name='label_in')
        synth_image, synth_labels = gen_model(label_input)
        skull_mask = tf.cast(synth_labels > 0, tf.float32)
        synth_image = KL.Multiply(name='synth_image_noair')([synth_image, skull_mask])

        # extract the separate images (disabled for now)
        synth_image1 = KL.Lambda(lambda x: x[..., 0:1])(synth_image)
        synth_image2 = KL.Lambda(lambda x: x[..., 0:1])(synth_image)  # was 1:2

        # sample a rigid transform  and apply it to the image
        im1_transformed, mats1 = vxms.layers.AffineAugment(
            name='im1_transformed', max_trans=max_trans, max_rot=max_rot, 
            return_mats=True, limit_to_fov=False)([synth_image1])
        im2_transformed, mats2 = vxms.layers.AffineAugment(
            name='im2_transformed', max_trans=max_trans, max_rot=max_rot, 
            return_mats=True, limit_to_fov=False)([synth_image2])

        # store the augmentation transforms for later use and apply to labels
        Id = tf.eye(ndims + 1)[0:3, :]  # have to remove identity
        Id = tf.eye(ndims + 1)[tf.newaxis, 0:3, :]  # have to remove identity
        mats1_noid = KL.Lambda(lambda x: tf.subtract(x, Id), name='subID1')(mats1)
        mats2_noid = KL.Lambda(lambda x: tf.subtract(x, Id), name='subID2')(mats2)
        labels1_transformed = vxm.layers.SpatialTransformer(
            name='labels1_transformed', interp_method='nearest', 
            fill_value=fill_value)([tf.cast(synth_labels, dtype=tf.float32), mats1_noid])
        labels2_transformed = vxm.layers.SpatialTransformer(
            name='labels2_transformed', interp_method='nearest',
            fill_value=fill_value)([tf.cast(synth_labels, dtype=tf.float32), mats2_noid])

        # sample B0 warps and apply them
        B0_map = vxms.layers.DrawB0Map(inshape, name='B0_map', max_std=max_B0_std, 
                                       min_blur=min_B0_blur, max_blur=max_B0_blur)(B0_dir1)
        B0_map_noair = KL.Multiply(name='B0_map_noair')([B0_map, skull_mask])

        B0_map1 = vxm.layers.SpatialTransformer(name='B0_map1', interp_method='linear',
                                                fill_value=fill_value)(
                                                    [B0_map_noair, mats1_noid])
        B0_map2 = vxm.layers.SpatialTransformer(name='B0_map2', interp_method='linear',
                                                fill_value=fill_value)(
                                                    [B0_map_noair, mats2_noid])

        # def_field[..., np.newaxis] * reshape(read_dir, [1, 1, 1, 3])
        B0_warp1 = KL.dot([B0_map1, tf.transpose(B0_dir1, (0, 2, 1))], axes=[-1, 1], 
                          name='B0_warp1')
        B0_warp2 = KL.dot([B0_map2, tf.transpose(B0_dir2, (0, 2, 1))], axes=[-1, 1], 
                          name='B0_warp2')

        im1_distorted = vxm.layers.SpatialTransformer(
            name='im1_distorted', interp_method='linear', 
            fill_value=fill_value)([im1_transformed, B0_warp1])
        im2_distorted = vxm.layers.SpatialTransformer(
            name='im2_distorted', interp_method='linear', 
            fill_value=fill_value)([im2_transformed, B0_warp2])

        # affine_inputs = [im1_B0_corrected, im2_B0_corrected] (disabled)
        model_inputs = [label_input, B0_dir1, B0_dir2]
        affine_inputs = [im1_distorted, im2_distorted]
        input_model = tf.keras.models.Model(inputs=model_inputs, outputs=affine_inputs)

        vxm_affine = VxmAffineEncoderThenDense(inshape, affine_nf, 
                                               input_model=input_model,
                                               fill_value=fill_value,
                                               rescale_affine=rescale_affine,
                                               reserve_capacity=reserve_capacity,
                                               **kwargs)
        fwd_affine = nes.utils.find_layers_with_substring(
            vxm_affine, 'full_affine')[0].get_output_at(-1)
        inv_affine = nes.utils.find_layers_with_substring(
            vxm_affine, 'invert_affine')[0].get_output_at(-1)

        im1_reg = vxm_affine.outputs[0]
        im2_reg = vxm_affine.outputs[1]

        # compose the augmentation matrices with the vxm-computed rigid transform
        comp_fwd = vxm.layers.ComposeTransform(name='comp_fwd')([mats1_noid, fwd_affine])
        comp_inv = vxm.layers.ComposeTransform(name='comp_inv')([mats2_noid, inv_affine])

        # a unet to predict B0 distortion magnitude
        # the inputs are the two images plus the two B0 vectors
        # connected with a dense layer and stacked
        B0_dir1_flat = KL.Flatten(name='dir1_flat')(B0_dir1)
        B0_image1_flat = KL.Dense(np.prod(inshape), name='B0_dir1_im')(B0_dir1_flat)
        B0_image1 = KL.Reshape(inshape + (1, ), name='B0_image1')(B0_image1_flat)

        B0_dir2_flat = KL.Flatten(name='dir2_flat')(B0_dir2)
        B0_image2_flat = KL.Dense(np.prod(inshape), name='B0_dir2_im')(B0_dir2_flat)
        B0_image2 = KL.Reshape(inshape + (1, ), name='B0_image2')(B0_image2_flat)

        # apply the composed transform to images and labels
        im1_reg_transformed = vxm.layers.SpatialTransformer(
            name='im1_reg_transformed', fill_value=fill_value)([synth_image1, comp_fwd])
        im2_reg_transformed = vxm.layers.SpatialTransformer(
            name='im2_reg_transformed', fill_value=fill_value)([synth_image2, comp_inv])
        labels_reg_fwd_transformed = vxm.layers.SpatialTransformer(
            name='labels_reg_fwd_transformed', interp_method='nearest',
            fill_value=fill_value)([tf.cast(synth_labels, dtype=tf.float32), comp_fwd])
        labels_reg_inv_transformed = vxm.layers.SpatialTransformer(
            name='labels_reg_inv_transformed', interp_method='nearest',
            fill_value=fill_value)([tf.cast(synth_labels, dtype=tf.float32), comp_inv])

        # cascade a 2nd affine that learns more slowly than the first
        input_model2 = tf.keras.models.Model(
            vxm_affine.inputs, [im1_reg_transformed, im2_transformed])
        rescale_affine2 = [rescale_affine[0] * 1e-2, rescale_affine[1] * 1e-3]
        vxm_affine2 = VxmAffineEncoderThenDense(inshape, affine_nf, 
                                                input_model=input_model2,
                                                rescale_affine=rescale_affine2,
                                                fill_value=fill_value,
                                                name='reserve',
                                                **kwargs)
        # nes.utils.scale_layer_weights(vxm_affine2, 1e-3)
        fwd_affine2 = nes.utils.find_layers_with_substring(
            vxm_affine2, 'full_affine')[-1].get_output_at(-1)
        inv_affine2 = nes.utils.find_layers_with_substring(
            vxm_affine2, 'invert_affine')[-1].get_output_at(-1)
        fwd_affine2 = vxm_affine2.references.affine
        inv_affine2 = vxm_affine2.references.inv_affine

        # transforms that go from before augmentation to in register
        comp_fwd2 = vxm.layers.ComposeTransform(name='comp_fwd2')([comp_fwd, fwd_affine2])
        comp_inv2 = vxm.layers.ComposeTransform(name='comp_inv2')([comp_inv, inv_affine2])

        # transforms that go from after augmentation to in register
        comp_aug1 = vxm.layers.ComposeTransform(name='comp_aug1')([fwd_affine, fwd_affine2])
        comp_aug2 = vxm.layers.ComposeTransform(name='comp_aug2')([inv_affine, inv_affine2])
        im1_reg2_transformed = vxm.layers.SpatialTransformer(
            name='im1_reg2_transformed', fill_value=fill_value)(
                [synth_image1, comp_fwd2])
        im2_reg2_transformed = vxm.layers.SpatialTransformer(
            name='im2_reg2_transformed', fill_value=fill_value)(
                [synth_image2, comp_inv2])
        im1_reg2_distorted = vxm.layers.SpatialTransformer(
            name='im1_reg2_distorted', fill_value=fill_value)(
                [im1_distorted, comp_aug1])
        im2_reg2_distorted = vxm.layers.SpatialTransformer(
            name='im2_reg2_distorted', fill_value=fill_value)(
                [im2_distorted, comp_aug2])
        labels1_reg2_transformed = vxm.layers.SpatialTransformer(
            name='labels1_reg2_transformed', interp_method='nearest',
            fill_value=fill_value)(
                [tf.cast(synth_labels, dtype=tf.float32), comp_fwd2])
        labels2_reg2_transformed = vxm.layers.SpatialTransformer(
            name='labels2_reg2_transformed', interp_method='nearest',
            fill_value=fill_value)(
                [tf.cast(synth_labels, dtype=tf.float32), comp_inv2])
        moving_outputs = KL.Concatenate(axis=-1, name='moving_out2')(
            [im2_transformed, im1_reg2_transformed, 
             labels2_transformed, labels1_reg2_transformed])
        fixed_outputs = KL.Concatenate(axis=-1, name='fixed_out2')(
            [im1_transformed, im2_reg2_transformed, 
             labels1_transformed, labels2_reg2_transformed])

        # now build a VxmDense that estimates the differential B0 distortion
        # inputs1 = [im1_reg, B0_image1]
        # inputs2 = [im2_reg, B0_image2]
        # unet_inputs1 = KL.Concatenate(axis=-1, name='unet_inputs1')(inputs1)
        # unet_inputs2 = KL.Concatenate(axis=-1, name='unet_inputs2')(inputs2)
        dense_inputs = [im1_reg2_distorted, im2_distorted]
        dense_input_model = tf.keras.Model(vxm_affine.inputs, dense_inputs)
        vxm_B0_correct1 = vxm.networks.VxmDense(
            inshape, 
            src_feats=dense_inputs[0].get_shape().as_list()[-1],
            trg_feats=dense_inputs[1].get_shape().as_list()[-1],
            fill_value=fill_value,
            bidir=True,
            nb_unet_features=unet_features,
            name='vxm_B0_correct1',
            input_model=dense_input_model)
        vxm_B0_correct2 = vxm.networks.VxmDense(
            inshape, 
            src_feats=dense_inputs[0].get_shape().as_list()[-1],
            trg_feats=dense_inputs[1].get_shape().as_list()[-1],
            fill_value=fill_value,
            bidir=True,
            nb_unet_features=unet_features,
            name='vxm_B0_correct2',
            input_model=dense_input_model)

        # comp_inv3 = vxm.layers.ComposeTransform(name='comp_inv3')(
        #    [comp_inv2, vxm_B0_correct.references.neg_flow])
        # comp_fwd3 = vxm.layers.ComposeTransform(name='comp_fwd3')(
        #    [comp_fwd2, vxm_B0_correct.references.pos_flow])
        # im1_B0_corrected = vxm.layers.SpatialTransformer(
        #    name='im1_B0_corrected_transformed', fill_value=fill_value)(
        #        [synth_image1, comp_fwd3])
        # im2_B0_corrected = vxm.layers.SpatialTransformer(
        #    name='im2_B0_corrected_transformed', fill_value=fill_value)(
        #        [synth_image2, comp_inv3])
        im1_B0_corrected = vxm.layers.SpatialTransformer(
            name='im1_B0_corrected_transformed', fill_value=fill_value)(
                [im1_distorted, vxm_B0_correct1.references.pos_flow])
        im2_B0_corrected = vxm.layers.SpatialTransformer(
            name='im2_B0_corrected_transformed', fill_value=fill_value)(
                [im2_distorted, vxm_B0_correct2.references.pos_flow])
        # B0_outputs1 = KL.Concatenate(axis=-1, name='B0_outputs1')(
        #    [im2_distorted, im1_B0_corrected])
        # B0_outputs2 = KL.Concatenate(axis=-1, name='B0_outputs2')(
        #    [im1_distorted, im2_B0_corrected]) 
        # im1_B0_corrected = vxm_B0_correct.outputs[0]
        # im2_B0_corrected = vxm_B0_correct.outputs[1]
        B0_outputs1 = KL.Concatenate(axis=-1, name='B0_outputs1')(
            [im1_transformed, im1_B0_corrected])
        B0_outputs2 = KL.Concatenate(axis=-1, name='B0_outputs2')(
            [im2_transformed, im2_B0_corrected]) 

        # return labels in fixed/moving space in case user wants them
        outputs = [moving_outputs, fixed_outputs, 
                   B0_outputs1, B0_outputs2, 
                   vxm_B0_correct1.references.pos_flow,
                   vxm_B0_correct2.references.pos_flow]

        self.references = vxm_affine.references
        self.references.vxm_affine = vxm_affine
        self.references.vxm_affine2 = vxm_affine2
        self.references.affine = comp_fwd2
        self.references.inv_affine = comp_inv2
        self.references.vxm_B0_correct1 = vxm_B0_correct1
        self.references.vxm_B0_correct2 = vxm_B0_correct2
        self.references.im_corr1 = im1_B0_corrected
        self.references.im_corr2 = im2_B0_corrected

        inputs = vxm_affine.inputs
        super().__init__(name=name, inputs=inputs, outputs=outputs)


class VxmAffineSymmetricWithCorrection(ne.tf.modelio.LoadableModel):
    """
    VoxelMorph network for linear (affine) registration between two images.
    including nonlinear B0 correction

    Author: brf2
    """
    @ne.modelio.store_config_args
    def __init__(
            self, 
            inshape, 
            labels_in, 
            labels_out, 
            affine_nf=64, 
            gen_args={}, 
            unet_features=64, 
            unet_levels=5, 
            fill_value=None, 
            name='VxmAffineWithCorrection', 
            max_trans=15, 
            rescale_affine=[1, .01],
            reserve_encoders=None,
            max_rot=10, 
            min_B0_blur=1,
            max_B0_blur=20,
            max_B0_std=1,
            reserve_capacity=None, 
            affine_conv_per_level=1, 
            **kwargs):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            name: Model name. Default is 'vxm_affine'.
        """

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # input vectors that give B0 distortion direction
        B0_dir1 = KL.Input(shape=(3, 1), name='B0_dir1')
        B0_dir2 = KL.Input(shape=(3, 1), name='B0_dir2')

        gen_model = ne.models.labels_to_image(
            inshape,
            labels_in,
            labels_out,
            zero_background=1,    # have air be black
            id=0,
            num_chan=2,
            return_def=False,
            one_hot=False,
            **gen_args
        )

        # input a label map and use it to synthesize an image. Remove air
        label_input = KL.Input(shape=inshape + (1,), name='label_in')
        synth_image, synth_labels = gen_model(label_input)
        skull_mask = tf.cast(synth_labels > 0, tf.float32)
        synth_image = KL.Multiply(name='synth_image_noair')([synth_image, skull_mask])

        # extract the separate images (disabled for now)
        synth_image1 = KL.Lambda(lambda x: x[..., 0:1])(synth_image)
        synth_image2 = KL.Lambda(lambda x: x[..., 0:1])(synth_image)  # was 1:2

        # sample a rigid transform  and apply it to the image
        im1_transformed, mats1 = vxms.layers.AffineAugment(
            name='im1_transformed', max_trans=max_trans, max_rot=max_rot, 
            return_mats=True, limit_to_fov=False)([synth_image1])
        im2_transformed, mats2 = vxms.layers.AffineAugment(
            name='im2_transformed', max_trans=max_trans, max_rot=max_rot, 
            return_mats=True, limit_to_fov=False)([synth_image2])

        # store the augmentation transforms for later use and apply to labels
        Id = tf.eye(ndims + 1)[0:3, :]  # have to remove identity
        Id = tf.eye(ndims + 1)[tf.newaxis, 0:3, :]  # have to remove identity
        mats1_noid = KL.Lambda(lambda x: tf.subtract(x, Id), name='subID1')(mats1)
        mats2_noid = KL.Lambda(lambda x: tf.subtract(x, Id), name='subID2')(mats2)
        labels1_transformed = vxm.layers.SpatialTransformer(
            name='labels1_transformed', interp_method='nearest', 
            fill_value=fill_value)([tf.cast(synth_labels, dtype=tf.float32), mats1_noid])
        labels2_transformed = vxm.layers.SpatialTransformer(
            name='labels2_transformed', interp_method='nearest',
            fill_value=fill_value)([tf.cast(synth_labels, dtype=tf.float32), mats2_noid])

        # sample B0 warps and apply them
        B0_map = vxms.layers.DrawB0Map(inshape, name='B0_map', max_std=max_B0_std, low_B0_std=0,
                                       min_blur=min_B0_blur, max_blur=max_B0_blur)(B0_dir1)
        B0_map_noair = KL.Multiply(name='B0_map_noair')([B0_map, skull_mask])

        B0_map1 = vxm.layers.SpatialTransformer(name='B0_map1', interp_method='linear',
                                                fill_value=fill_value)(
                                                    [B0_map_noair, mats1_noid])
        B0_map2 = vxm.layers.SpatialTransformer(name='B0_map2', interp_method='linear',
                                                fill_value=fill_value)(
                                                    [B0_map_noair, mats2_noid])

        # def_field[..., np.newaxis] * reshape(read_dir, [1, 1, 1, 3])
        B0_warp1 = KL.dot([B0_map1, tf.transpose(B0_dir1, (0, 2, 1))], axes=[-1, 1], 
                          name='B0_warp1')
        B0_warp2 = KL.dot([B0_map2, tf.transpose(B0_dir2, (0, 2, 1))], axes=[-1, 1], 
                          name='B0_warp2')

        im1_distorted = vxm.layers.SpatialTransformer(
            name='im1_distorted', interp_method='linear', 
            fill_value=fill_value)([im1_transformed, B0_warp1])
        im2_distorted = vxm.layers.SpatialTransformer(
            name='im2_distorted', interp_method='linear', 
            fill_value=fill_value)([im2_transformed, B0_warp2])

        # affine_inputs = [im1_B0_corrected, im2_B0_corrected] (disabled)
        model_inputs = [label_input, B0_dir1, B0_dir2]
        affine_inputs = [im1_distorted, im2_distorted]
        input_model = tf.keras.models.Model(inputs=model_inputs, outputs=affine_inputs)

        vxm_affine = VxmAffineEncoderThenDenseNew(inshape, affine_nf, 
                                                  input_model=input_model,
                                                  fill_value=fill_value,
                                                  rescale_affine=rescale_affine,
                                                  reserve_capacity=reserve_capacity,
                                                  reserve_encoders=reserve_encoders,
                                                  **kwargs)
        affine1 = vxm_affine.references.affine
        affine2 = vxm_affine.references.inv_affine

        half1 = vxms.layers.MidspaceTransform(add_identity=True, name='half1')(affine1)
        half2 = vxms.layers.MidspaceTransform(add_identity=True, name='half2')(affine2)

        # compose the augmentation matrices with the vxm-computed rigid transform
        comp1 = vxm.layers.ComposeTransform(name='comp1')([mats1_noid, affine1])
        comp2 = vxm.layers.ComposeTransform(name='comp2')([mats2_noid, affine2])
        comp1_half = vxm.layers.ComposeTransform(name='comp1_half')([mats1_noid, half1])
        comp2_half = vxm.layers.ComposeTransform(name='comp2_half')([mats2_noid, half2])

        # apply the composed half and full transforms to images and labels
        im1_halfreg = vxm.layers.SpatialTransformer(
            name='im1_halfreg', fill_value=fill_value)([synth_image1, comp1_half])
        im2_halfreg = vxm.layers.SpatialTransformer(
            name='im2_halfreg', fill_value=fill_value)([synth_image2, comp2_half])
        labels1_halfreg = vxm.layers.SpatialTransformer(
            name='labels1_halfreg', interp_method='nearest',
            fill_value=fill_value)([tf.cast(synth_labels, dtype=tf.float32), comp1_half])
        labels2_halfreg = vxm.layers.SpatialTransformer(
            name='labels2_halfreg', interp_method='nearest',
            fill_value=fill_value)([tf.cast(synth_labels, dtype=tf.float32), comp2_half])

        im1_reg = vxm.layers.SpatialTransformer(
            name='im1_reg', fill_value=fill_value)([synth_image1, comp1])
        im2_reg = vxm.layers.SpatialTransformer(
            name='im2_reg', fill_value=fill_value)([synth_image2, comp2])
        labels1_reg = vxm.layers.SpatialTransformer(
            name='labels1_reg', interp_method='nearest',
            fill_value=fill_value)([tf.cast(synth_labels, dtype=tf.float32), comp1])
        labels2_reg = vxm.layers.SpatialTransformer(
            name='labels2_reg', interp_method='nearest',
            fill_value=fill_value)([tf.cast(synth_labels, dtype=tf.float32), comp2])

        moving_outputs = KL.Concatenate(axis=-1, name='moving_out')(
            [im2_transformed, im1_reg, 
             labels2_transformed, labels1_reg])
        fixed_outputs = KL.Concatenate(axis=-1, name='fixed_out')(
            [im1_transformed, im2_reg, 
             labels1_transformed, labels2_reg])

        # now build a VxmDense that estimates the differential B0 distortion
        # inputs1 = [im1_reg, B0_image1]
        # inputs2 = [im2_reg, B0_image2]
        # unet_inputs1 = KL.Concatenate(axis=-1, name='unet_inputs1')(inputs1)
        # unet_inputs2 = KL.Concatenate(axis=-1, name='unet_inputs2')(inputs2)
        im1_distorted_halfreg = vxm.layers.SpatialTransformer(
            name='im1_distorted_halfreg', fill_value=fill_value)([synth_image1, comp1_half])
        im2_distorted_halfreg = vxm.layers.SpatialTransformer(
            name='im2_distorted_halfreg', fill_value=fill_value)([synth_image1, comp2_half])
        half1_inv = vxm.layers.InvertAffine(name='half1_inv')(half1)
        half2_inv = vxm.layers.InvertAffine(name='half2_inv')(half2)
        dense_inputs = [im1_distorted_halfreg, im2_distorted_halfreg]
        dense_input_model = tf.keras.Model(vxm_affine.inputs, dense_inputs)
        vxm_B0_correct1 = vxm.networks.VxmDense(
            inshape, 
            src_feats=dense_inputs[0].get_shape().as_list()[-1],
            trg_feats=dense_inputs[1].get_shape().as_list()[-1],
            fill_value=fill_value,
            bidir=True,
            nb_unet_features=unet_features,
            name='vxm_B0_correct1',
            input_model=dense_input_model)
        vxm_B0_correct2 = vxm.networks.VxmDense(
            inshape, 
            src_feats=dense_inputs[0].get_shape().as_list()[-1],
            trg_feats=dense_inputs[1].get_shape().as_list()[-1],
            fill_value=fill_value,
            bidir=True,
            nb_unet_features=unet_features,
            name='vxm_B0_correct2',
            input_model=dense_input_model)

        # transform the correction to the original space and apply it
        warp1 = vxm_B0_correct1.references.pos_flow
        warp2 = vxm_B0_correct2.references.pos_flow
        B0_unwarp1 = vxm.layers.ComposeTransform(name='B0_unwarp1')([half1, warp1])
        B0_unwarp2 = vxm.layers.ComposeTransform(name='B0_unwarp2')([half2, warp2])

        im1_B0_corrected = vxm.layers.SpatialTransformer(
            name='im1_B0_corrected', fill_value=fill_value)([im1_distorted, B0_unwarp1])
        im2_B0_corrected = vxm.layers.SpatialTransformer(
            name='im2_B0_corrected', fill_value=fill_value)([im2_distorted, B0_unwarp2])
        im1_B0_corrected = vxm_B0_correct1.outputs[0]
        im2_B0_corrected = vxm_B0_correct1.outputs[1]

        B0_outputs1 = KL.Concatenate(axis=-1, name='B0_outputs1')(
            [im2_distorted_halfreg, im1_B0_corrected])
        B0_outputs2 = KL.Concatenate(axis=-1, name='B0_outputs2')(
            [im1_distorted_halfreg, im2_B0_corrected]) 

        # return labels in fixed/moving space in case user wants them
        outputs = [moving_outputs, fixed_outputs, 
                   B0_outputs1, B0_outputs2, 
                   vxm_B0_correct1.references.pos_flow,
                   vxm_B0_correct2.references.pos_flow]

        self.references = vxm_affine.references
        self.references.vxm_affine = vxm_affine
        self.references.affine = comp1
        self.references.inv_affine = comp2
        self.references.vxm_B0_correct1 = vxm_B0_correct1
        self.references.vxm_B0_correct2 = vxm_B0_correct2
        self.references.im_corr1 = im1_B0_corrected
        self.references.im_corr2 = im2_B0_corrected

        inputs = vxm_affine.inputs
        super().__init__(name=name, inputs=inputs, outputs=outputs)


class VxmSynthCombo(ne.tf.modelio.LoadableModel):
    """
    VoxelMorph network for combo linear (affine)/dense registration between two images.
    including nonlinear B0 correction

    Author: brf2
    """
    @ne.modelio.store_config_args
    def __init__(
            self, 
            inshape, 
            labels_in, 
            labels_out, 
            affine_nf=64, 
            gen_args={}, 
            unet_features=64, 
            nb_unet_conv_per_level=2,
            unet_levels=5, 
            fill_value=None, 
            name='VxmAffineWithCorrection', 
            max_trans=15, 
            rescale_affine=[1, .01],
            reserve_encoders=[1e-3],
            max_rot=10, 
            min_B0_blur=1,
            max_B0_blur=20,
            max_B0_std=1,
            affine_conv_per_level=1, 
            structure_list={},
            Conv=None,
            subsample_atrophy=1,
            **kwargs):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            name: Model name. Default is 'vxm_affine'.
            subsample_atrophy - if not None, specifies what fraction of the total number of 
                                structures to simulate atrophy in for each sample
        """

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        seeds = dict(mean=1, std=2, warp=3, blur=4, bias=5, gamma=6)
        gen_model = ne.models.labels_to_image(
            inshape,
            labels_in,
            labels_out,
            zero_background=1,    # have air be black
            id=0,
            num_chan=2,
            return_def=False,
            one_hot=False,
            seeds=seeds,
            **gen_args
        )
        gen_model_a = ne.models.labels_to_image(
            inshape,
            labels_in,
            labels_out,
            zero_background=1,    # have air be black
            id=1,
            num_chan=2,
            return_def=False,
            one_hot=False,
            seeds=seeds,
            **gen_args
        )

        # input a label map and use it to synthesize an image. Remove air
        label_input = KL.Input(shape=inshape + (1,), name='label_in')

        # input vectors that give B0 distortion direction
        draw_dir = lambda _: tf.random.shuffle([1.0, 0.0, 0.0])[..., tf.newaxis]
        B0_dir1 = KL.Lambda(lambda x: tf.map_fn(draw_dir, x), name='B0_dir1')(label_input)
        B0_dir2 = KL.Lambda(lambda x: tf.map_fn(draw_dir, x), name='B0_dir2')(label_input)

        label_input_with_atrophy = vxms.layers.ResizeLabels(
            structure_list, name='resize_labels', subsample_atrophy=subsample_atrophy)(label_input)

        synth_image, synth_labels = gen_model(label_input)
        atrophy_image, atrophy_labels = gen_model_a(label_input_with_atrophy)

        # skull_mask = tf.cast(synth_labels > 0, tf.float32)

        # extract the separate images (disabled for now)
        synth_image1 = KL.Lambda(lambda x: x[..., 0:1], name='synth_image1')(synth_image)
        synth_image2 = KL.Lambda(lambda x: x[..., 0:1], name='synth_image2')(atrophy_image)  # 1:2
        synth_image2_no_atrophy = KL.Lambda(lambda x: x[..., 0:1], 
                                            name='synth_image2_no_atrophy')(synth_image)

        # sample a rigid transform  and apply it to the image
        im1_transformed, mats1 = vxms.layers.AffineAugment(
            name='im1_transformed', max_trans=max_trans, max_rot=max_rot, 
            return_mats=True, limit_to_fov=True)([synth_image1])
        im2_transformed, mats2 = vxms.layers.AffineAugment(
            name='im2_transformed', max_trans=max_trans, max_rot=max_rot, 
            return_mats=True, limit_to_fov=True)([synth_image2])

        # store the augmentation transforms for later use and apply to labels
        Id = tf.eye(ndims + 1)[0:3, :]  # have to remove identity
        Id = tf.eye(ndims + 1)[tf.newaxis, 0:3, :]  # have to remove identity
        mats1_noid = KL.Lambda(lambda x: tf.subtract(x, Id), name='aug1_noID')(mats1)
        mats2_noid = KL.Lambda(lambda x: tf.subtract(x, Id), name='aug2_noID')(mats2)
        labels1_transformed = vxm.layers.SpatialTransformer(
            name='labels1_transformed', interp_method='nearest', 
            fill_value=fill_value)([tf.cast(synth_labels, dtype=tf.float32), mats1_noid])
        labels2_transformed = vxm.layers.SpatialTransformer(
            name='labels2_transformed', interp_method='nearest',
            fill_value=fill_value)([tf.cast(atrophy_labels, dtype=tf.float32), mats2_noid])
        im2_transformed_no_atrophy = vxm.layers.SpatialTransformer(
            name='im2_transformed_no_atrophy', interp_method='linear', 
            fill_value=fill_value)([synth_image2_no_atrophy, mats2_noid])

        # sample B0 warps and apply them
        B0_map = vxms.layers.DrawB0Map(inshape, name='B0_map', max_std=max_B0_std, 
                                       min_blur=min_B0_blur, max_blur=max_B0_blur)(B0_dir1)
        # B0_map_noair = KL.Multiply(name='B0_map_noair')([B0_map, skull_mask])

        B0_map1 = vxm.layers.SpatialTransformer(name='B0_map1', interp_method='linear',
                                                fill_value=fill_value)(
                                                    [B0_map, mats1_noid])
        B0_map2 = vxm.layers.SpatialTransformer(name='B0_map2', interp_method='linear',
                                                fill_value=fill_value)(
                                                    [B0_map, mats2_noid])

        # compute B0 warp for each image (same B0 inhomogeneity, different direction)
        B0_warp1 = KL.dot([B0_map1, tf.transpose(B0_dir1, (0, 2, 1))], axes=[-1, 1], 
                          name='B0_warp1')
        B0_warp2 = KL.dot([B0_map2, tf.transpose(B0_dir2, (0, 2, 1))], axes=[-1, 1], 
                          name='B0_warp2')

        # transform from synth images to B0 warped images
        warp1_aug_B0 = vxm.layers.ComposeTransform(name='warp1_aug_B0')([mats1_noid, B0_warp1])
        warp2_aug_B0 = vxm.layers.ComposeTransform(name='warp2_aug_B0')([mats2_noid, B0_warp2])

        # apply the distortions to the images
        im1_distorted = vxm.layers.SpatialTransformer(
            name='im1_distorted', interp_method='linear', 
            fill_value=fill_value)([synth_image1, warp1_aug_B0])
        im2_distorted = vxm.layers.SpatialTransformer(
            name='im2_distorted', interp_method='linear', 
            fill_value=fill_value)([synth_image2, warp2_aug_B0])

        # affine_inputs = [im1_B0_corrected, im2_B0_corrected] (disabled)
        combo_inputs = [im1_distorted, im2_distorted]
        input_model = tf.keras.models.Model(inputs=label_input, outputs=combo_inputs)

        vxm_affine = VxmAffineEncoderThenDenseNew(
            inshape, 
            enc_nf=affine_nf, 
            input_model=input_model,
            fill_value=fill_value,
            rescale_affine=rescale_affine,
            reserve_encoders=reserve_encoders,
            Conv=Conv,
            store_midspace=True,
            **kwargs)

        half1 = vxm_affine.references.half_affine_fwd
        half2 = vxm_affine.references.half_affine_inv

        # compute transform that results in distorted images in common affine midspace
        comp1_nlin = vxm.layers.ComposeTransform(name='comp1_nl_input')([warp1_aug_B0, half1])
        comp2_nlin = vxm.layers.ComposeTransform(name='comp2_nl_input')([warp2_aug_B0, half2])

        # compose the augmentation/B0 warp with the vxm-computed affine warp transform
        affine1 = vxm_affine.references.affine
        affine2 = vxm_affine.references.inv_affine
        comp1 = vxm.layers.ComposeTransform(name='comp1')([warp1_aug_B0, affine1])
        comp2 = vxm.layers.ComposeTransform(name='comp2')([warp2_aug_B0, affine2])

        im1_to_im2 = vxm.layers.SpatialTransformer(
            name='im1_to_im2', interp_method='linear', 
            fill_value=fill_value)([synth_image1, comp1])
        im2_to_im1 = vxm.layers.SpatialTransformer(
            name='im2_to_im1', interp_method='linear', 
            fill_value=fill_value)([synth_image2, comp2])

        im1_dist_nlin = vxm.layers.SpatialTransformer(
            name='im1_dist_nlin', interp_method='linear', 
            fill_value=fill_value)([synth_image1, comp1_nlin])
        im2_dist_nlin = vxm.layers.SpatialTransformer(
            name='im2_dist_nlin', interp_method='linear', 
            fill_value=fill_value)([synth_image2, comp2_nlin])

        # unet_features[1] += [2]*nb_unet_conv_per_level  # we want 2 outputs of unet for B0 unwarps
        unet_model = ne.models.unet(
            unet_features,
            inshape + (4,),
            unet_levels,
            3,              # conv size
            2,              # nb_labels
            nb_conv_per_level=nb_unet_conv_per_level,
            name='B0_correction_unet',
            final_pred_activation='linear',
        )

        # unet_inputs = KL.Concatenate(name='unet_inputs', axis=-1)([im1_dist_nlin, im2_dist_nlin])
        unet_inputs = KL.Concatenate(name='unet_inputs', axis=-1)(
            [im1_to_im2, im2_distorted, im2_to_im1, im1_distorted])
        Conv = getattr(KL, 'Conv%dD' % ndims)
        flow_mean = Conv(ndims, kernel_size=3, padding='same',
                         kernel_initializer=KI.RandomNormal(mean=0.0, stddev=1e-5),
                         name='%s_flow' % name)(unet_model.output)
        unet_outputs = unet_model(unet_inputs)
        B0_unmap1 = KL.Lambda(lambda x: x[..., 0:1])(unet_outputs)
        B0_unmap2 = KL.Lambda(lambda x: x[..., 1:2])(unet_outputs)
        B0_unwarp1 = KL.Dot(axes=[-1, 1], name='B0_unwarp1')(
            [B0_unmap1, tf.transpose(B0_dir1, (0, 2, 1))])
        B0_unwarp2 = KL.Dot(axes=[-1, 1], name='B0_unwarp2')(
            [B0_unmap2, tf.transpose(B0_dir2, (0, 2, 1))])

        # create a composition that goes from im1_distorted --> im2_transformed and
        # im2_distorted --> im1_transformed 
        comp1_nl = vxm.layers.ComposeTransform(name='warp1_aug_B0_warp_B0_unwarp')(
            [mats1_noid, B0_warp1, B0_unwarp1, affine1])
        comp2_nl = vxm.layers.ComposeTransform(name='warp2_aug_B0_warp_B0_unwarp')(
            [mats2_noid, B0_warp2, B0_unwarp2, affine2])

        # apply the composed half and full transforms to images and labels
        # use the synth2 image without atrophy so that we learn a transform that
        # is invariant to atrophy
        im1_reg_nl = vxm.layers.SpatialTransformer(
            name='im1_reg_nl', interp_method='linear', 
            fill_value=fill_value)([synth_image1, comp1_nl])
        im2_reg_nl = vxm.layers.SpatialTransformer(
            name='im2_reg_nl', interp_method='linear', 
            fill_value=fill_value)([synth_image2_no_atrophy, comp2_nl])
        labels1_reg_nl = vxm.layers.SpatialTransformer(
            name='labels1_reg_nl', interp_method='nearest',
            fill_value=fill_value)([tf.cast(synth_labels, dtype=tf.float32), comp1_nl])
        labels2_reg_nl = vxm.layers.SpatialTransformer(
            name='labels2_reg_nl', interp_method='nearest',
            fill_value=fill_value)([tf.cast(atrophy_labels, dtype=tf.float32), comp2_nl])

        # apply the composed half and full transforms to images and labels
        im1_reg = vxm.layers.SpatialTransformer(
            name='im1_reg', fill_value=fill_value)([synth_image1, comp1])
        im2_reg = vxm.layers.SpatialTransformer(
            name='im2_reg', fill_value=fill_value)([synth_image2, comp2])
        labels1_reg = vxm.layers.SpatialTransformer(
            name='labels1_reg', interp_method='nearest',
            fill_value=fill_value)([tf.cast(synth_labels, dtype=tf.float32), comp1])
        labels2_reg = vxm.layers.SpatialTransformer(
            name='labels2_reg', interp_method='nearest',
            fill_value=fill_value)([tf.cast(atrophy_labels, dtype=tf.float32), comp2])

        moving_outputs = KL.Concatenate(axis=-1, name='moving_out')(
            [im2_distorted, im1_reg, labels2_transformed, labels1_reg])
        fixed_outputs = KL.Concatenate(axis=-1, name='fixed_out')(
            [im1_distorted, im2_reg, labels1_transformed, labels2_reg])

        moving_outputs_nl = KL.Concatenate(axis=-1, name='moving_out_nl')(
            [im2_transformed, im1_reg_nl, labels2_transformed, labels1_reg_nl])
        fixed_outputs_nl = KL.Concatenate(axis=-1, name='fixed_out_nl')(
            [im1_transformed, im2_reg_nl, labels1_transformed, labels2_reg_nl])

        # return labels in fixed/moving space in case user wants them
        outputs_affine = [moving_outputs, fixed_outputs, vxm_affine.outputs[-1]]
        outputs_nl = [moving_outputs_nl, fixed_outputs_nl, B0_unwarp1, B0_unwarp2]

        # make some things available to the caller for use in loss functions
        # including the undistorted affine-transformed volumes and the nonlinear
        # component of the combined warp

        affine1 = vxm_affine.references.affine
        affine2 = vxm_affine.references.inv_affine
        warp1 = vxm_affine.references.affine
        warp2 = vxm_affine.references.inv_affine

        affine1_comp = vxm.layers.ComposeTransform(name='affine1_comp')([mats1_noid, affine1])
        affine2_comp = vxm.layers.ComposeTransform(name='affine2_comp')([mats2_noid, affine2])
        im1_affine = vxm.layers.SpatialTransformer(name='im1_affine', fill_value=0)(
            [synth_image1, affine1_comp])
        im2_affine = vxm.layers.SpatialTransformer(name='im2_affine', fill_value=0)(
            [synth_image2_no_atrophy, affine2_comp])
        affine_out1 = KL.Concatenate(axis=-1, name='affine_out1')(
            [im2_transformed_no_atrophy, im1_affine])
        affine_out2 = KL.Concatenate(axis=-1, name='affine_out2')([im1_transformed, im2_affine])

        self.references = vxm_affine.references
        self.references.unet_model = unet_model
        self.references.im1 = synth_image1
        self.references.im2 = synth_image2
        self.references.affine1 = affine1_comp  # affine transform from im1 to im2_transformed
        self.references.affine2 = affine2_comp
        self.references.im1_affine = im1_affine
        self.references.im2_affine = im2_affine
        self.references.im1_reg_nl = im1_reg_nl
        self.references.im2_reg_nl = im2_reg_nl
        self.references.affine_out1 = affine_out1
        self.references.affine_out2 = affine_out2
        self.references.moving_outputs_nl = moving_outputs_nl
        self.references.fixed_outputs_nl = fixed_outputs_nl
        self.references.outputs_nl = outputs_nl
        self.references.outputs_affine = outputs_affine
        self.references.vxm_affine = vxm_affine
        self.references.composed1 = comp1
        self.references.composed2 = comp2

        super().__init__(name=name, inputs=label_input, outputs=outputs_nl)


class VxmAffineUnetDense(ne.tf.modelio.LoadableModel):
    """
    VoxelMorph network for linear (affine) registration between two images.
    Experimental idea: use a unet encoder/decoder with an optional breaking
    up into patches (if patch_size is not None), then tie the output to the affine
    parameters using a dense layer
    TODO: this init function is too long, we should have a separate model function
    """

    @ne.modelio.store_config_args
    def __init__(self, inshape,
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 src_feats=1,
                 trg_feats=1,
                 reserve_capacity=None,
                 bidir=False,
                 transform_type='affine',
                 blurs=[1],
                 rescale_affine=[1.0, 0.01],
                 name='vxm_affine_unet_dense',
                 fill_value=0,
                 patch_size=None,
                 strides=None,
                 pool_before_dense=None,
                 rates=None):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            enc_nf: List of encoder filters. e.g. [16, 32, 32, 32]
            bidir: Enable bidirectional cost function. Default is False.
            transform_type: 'affine' (default), 'rigid' or 'rigid+scale' currently
            blurs: List of gaussian blur kernel levels for inputs. Default is [1].
            rescale_affine: A scalar or array to rescale the output of the affine matrix layer.
                Default is 1.0.
                This improves stability by enabling different gradient flow to affect the affine
                parameters. Input array can match the transform matrix shape or it can be a 
                2-element list that represents individual [translation, linear] components.
            nchannels: Number of input channels. Default is 1.
            name: Model name. Default is 'vxm_affine_unet_dense'.
            patch_size,strides and rates are parameters to tf.image.extract_patches. Note
            that a singleton dimension is added to either end of these parameters for the
            batch and channels dimensions (so for a 2D images patches would be something
            like [32,32], which internally gets augmented to be [1,32,32,1]
        """

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure base encoder CNN
        Conv = getattr(KL, 'Conv%dD' % ndims)

        source = tf.keras.Input(shape=(*inshape, src_feats), name=name + '_source_input')
        target = tf.keras.Input(shape=(*inshape, trg_feats), name=name + '_target_input')

        if patch_size is not None:  # convert images into channels of overlapping patches
            extract_patches = tf.image.extract_patches if ndims == 2 else tf.extract_volume_patches
            if strides is None:
                strides = list(np.array(patch_size) / 2)
            if rates is None:
                rates = [1] * len(patch_size)
            patch_size = [1] + patch_size + [1]
            strides = [1] + strides + [1]
            rates = [1] + rates + [1]
            if ndims == 2:
                lbd_src = lambda x: extract_patches(x, patch_size,
                                                    strides=strides,
                                                    rates=rates,
                                                    padding='VALID',
                                                    name=name + '_source_patches')
                source_patches = KL.Lambda(lbd_src)(source[..., 0:1])
                lbd_tgt = lambda x: extract_patches(x, patch_size,
                                                    strides=strides,
                                                    rates=rates,
                                                    padding='VALID',
                                                    name=name + '_target_patches')
                target_patches = KL.Lambda(lbd_tgt)(target[..., 0:1])

                for fno in range(1, src_feats):
                    s2_patches = KL.Lambda(lbd_src)(source[..., fno:fno + 1])
                    concat_in = [source_patches, s2_patches]
                    source_patches = KL.Concatenate(axis=-1, name=name + '_s2_patches')(concat_in)

                for fno in range(1, trg_feats):
                    t2_patches = KL.Lambda(lbd_tgt)(target[..., fno:fno + 1])
                    concat_in = [target_patches, t2_patches]
                    target_patches = KL.Concatenate(axis=-1, name=name + '_t2_concat')(concat_in)

                pdims = (3, 1, 2)

            elif ndims == 3:
                lbd_src = lambda x: extract_patches(x, patch_size,
                                                    strides=strides,
                                                    padding='VALID',
                                                    name=name + '_source_patches')
                source_patches = KL.Lambda(lbd_src)(source)
                lbd_tgt = lambda x: extract_patches(x, patch_size,
                                                    strides=strides,
                                                    padding='VALID',
                                                    name=name + '_target_patches')
                target_patches = KL.Lambda(lbd_tgt)(target)
                pdims = (4, 1, 2, 3)

            source_perm = KL.Permute(pdims)(source_patches)
            target_perm = KL.Permute(pdims)(target_patches)
            source_shape = patch_size[1:ndims + 1] + \
                [np.prod(source_patches.shape[1:ndims + 1].as_list()) * src_feats]
            target_shape = patch_size[1:ndims + 1] + \
                [np.prod(target_patches.shape[1:ndims + 1].as_list()) * trg_feats]
            source_patches_reshaped = KL.Reshape(source_shape)(source_perm)
            target_patches_reshaped = KL.Reshape(target_shape)(target_perm)

            nchannels = source_patches_reshaped.shape.as_list()[-1]
            print('arranging patches into %d channels' % nchannels)

            input_model = tf.keras.Model(inputs=[source, target], outputs=[
                                         source_patches_reshaped, target_patches_reshaped])
        else:
            # configure default input layers if an input model is not provided
            input_model = tf.keras.Model(inputs=[source, target], outputs=[source, target])

        # build core unet model and grab inputs
        unet_model = vxm.networks.Unet(
            input_model=input_model,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            name=name + '_vxm_affine_unet',
        )

        latest_tensor = unet_model.output

        if transform_type == 'rigid':
            nb_params = 3 if ndims == 2 else 6
        elif transform_type == 'rigid+scale':
            nb_params = 4 if ndims == 2 else 7
        else:
            nb_params = ndims * (ndims + 1)

        latest_tensor = Conv(1, kernel_size=3, padding='same',
                             name=name + '_penultimate')(latest_tensor)
        if pool_before_dense is not None:
            MaxPooling = getattr(KL, 'MaxPooling%dD' % ndims)
            latest_tensor = MaxPooling(
                pool_size=[pool_before_dense] * ndims, name='pool_before_dense')(latest_tensor)

        flat = KL.Flatten(name=name + '_flat')(latest_tensor)
        init = KI.RandomNormal(mean=0.0, stddev=1e-7)
        full_affine = KL.Dense(
            nb_params, activation='linear', 
            name=name + '_DenseParams', kernel_initializer=init)(flat)

        rescale_np = np.ones((nb_params,))
        if hasattr(rescale_affine, '__len__') and len(rescale_affine) == 2:
            if transform_type.startswith('rigid'):
                rescale_np[:3] = rescale_affine[0]
                rescale_np[3:] = rescale_affine[1]
            else:
                scaling = np.ones((ndims, ndims + 1), dtype='float32')
                scaling[:, -1] = rescale_affine[0]  # translation
                scaling[:, :-1] = rescale_affine[1]  # linear (everything else)
                rescale_np = scaling[np.newaxis].flatten()
        else:
            rescale_np *= rescale_affine

        rescaled_affine = ne.layers.RescaleValues(
            rescale_np, name='affine_rescale_' + name)(full_affine)
        print(rescale_np)

        # turn affine parms into an affine matrix
        full_affine = vxm.layers.AffineTransformationsToMatrix(
            ndims, name='matrix_conversion_' + name)(rescaled_affine)

        # apply transform to moving image
        y_source = vxm.layers.SpatialTransformer(
            name=name + '_transformer', fill_value=fill_value)([source, full_affine])

        # invert affine for bidirectional training
        if bidir:
            inv_affine = vxm.layers.InvertAffine(name=name + '_invert_affine')(full_affine)
            y_target = vxm.layers.SpatialTransformer(
                name=name + '_neg_transformer', fill_value=fill_value)([target, inv_affine])
            outputs = [y_source, y_target]
        else:
            outputs = [y_source]

        # initialize the keras model
        super().__init__(name=name, inputs=[source, target], outputs=outputs)

        # cache affines
        self.references = ne.tf.modelio.LoadableModel.ReferenceContainer()
        self.references.affine = full_affine
        self.references.pos_flow = full_affine
        self.references.transform_type = transform_type
        self.references.bidir = bidir
        if bidir:
            self.references.inv_affine = inv_affine
            self.references.neg_flow = inv_affine

    def get_registration_model(self):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        return tf.keras.Model(self.inputs, self.references.affine)

    def register(self, src, trg):
        """
        Predicts the transform from src to trg tensors.
        """
        return self.get_registration_model().predict([src, trg])

    def apply_transform(self, src, trg, img, interp_method='linear', **kwargs):
        """
        Predicts the transform from src to trg and applies it to the img tensor.
        """
        warp_model = self.get_registration_model()
        img_input = tf.keras.Input(shape=img.shape[1:])
        stin = [img_input, warp_model.output]
        y_img = vxm.layers.SpatialTransformer(interp_method=interp_method)(stin)
        pred = [src, trg, img]
        return tf.keras.Model(warp_model.inputs + [img_input], y_img).predict(pred, **kwargs)

    def rescale_model(self, zoom_factor, interp_method='linear', fill_value=0):
        """
        Build and return a new model that computes the transform at the
        scale that was learned by the model, then rescales it so it can be
        applied to a different sized image (e.g. learning at 2x downsampling but
        applying at full res)

        Author: brf2
        """
        warnings.warn('brf: create rescale_model subclass of ne.tf.modelio.LoadableModel')

        # build new inputs that are scaled down and put the through old net
        shape = self.inputs[0].shape[1:].as_list()
        shape[0:-1] = list(np.array(shape[0:-1]) * zoom_factor)
        source_input = KL.Input(shape=shape, name='source_rescaled%d' % zoom_factor)
        target_input = KL.Input(shape=shape, name='target_rescaled%d' % zoom_factor)
        source_rescaled = ne.layers.Resize(1. / zoom_factor, interp_method)(source_input)
        target_rescaled = ne.layers.Resize(1. / zoom_factor, interp_method)(target_input)

        # put the scaled down inputs through the old net, then scale the affine
        # for full-scale images
        zoomed_inputs = [source_rescaled, target_rescaled]
        affine = self.get_registration_model()(zoomed_inputs)
        affine_rescaled = vxm.layers.RescaleTransform(
            zoom_factor,
            interp_method,
            name='affine')(affine)

        # apply the full-scale tranform to the full-scale images
        stin = [source_input, affine_rescaled]
        source_transformed = vxm.layers.SpatialTransformer(interp_method='linear',
                                                           indexing='ij',
                                                           name='transformer',
                                                           fill_value=fill_value)(stin)
        outputs = [source_transformed]
        if self.references.bidir:
            inv_affine = vxm.layers.InvertAffine(name='invert_affine')(affine_rescaled)
            target_transformed = vxm.layers.SpatialTransformer(
                name='inv_transformer', fill_value=fill_value)([target_input, inv_affine])

        # build the new model
        outputs += [target_transformed]
        inputs = [source_input, target_input]
        new_model = tf.keras.models.Model(inputs, outputs)

        # propagate variables from old model to new one
        new_model.references = ne.tf.modelio.LoadableModel.ReferenceContainer()
        new_model.references.affine = affine_rescaled
        # new_model.references.scale_affines = self.references.scale_affines
        new_model.references.transform_type = self.references.transform_type
        new_model.references.bidir = self.references.bidir
        if self.references.bidir:
            new_model.references.inv_affine = inv_affine

        return new_model


class VxmAffineEncoderThenDensePatches(ne.tf.modelio.LoadableModel):
    """
    VoxelMorph network for linear (affine) registration between two images. Internally
    this function reshapes the image into a set of overlapping patches and the conv
    layers operate across the patches as different channels

    Author: brf2
    """

    @ne.modelio.store_config_args
    def __init__(self, inshape, enc_nf,
                 bidir=False,
                 transform_type='affine',
                 blurs=[1],
                 rescale_affine=1.0,
                 nchannels=1,
                 name='vxm_affine',
                 fill_value=None,
                 input_model=None,
                 patch_size=[32, 32],
                 strides=[16, 16],
                 rates=[1, 1],
                 add_dim=False,
                 dropout=None,
                 downsize_input=None):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            enc_nf: List of encoder filters. e.g. [16, 32, 32, 32]
            bidir: Enable bidirectional cost function. Default is False.
            transform_type: 'affine' (default), 'rigid' or 'rigid+scale' currently
            blurs: List of gaussian blur kernel levels for inputs. Default is [1].
            rescale_affine: A scalar or array to rescale the output of the affine matrix layer.
                Default is 1.0.
                This improves stability by enabling different gradient flow to affect the affine
                parameters. Input array can match the transform matrix shape or it can be a 
                2-element list that represents individual [translation, linear] components.
            nchannels: Number of input channels. Default is 1.
            name: Model name. Default is 'vxm_affine'.
            patch_size,strides and rates are parameters to tf.image.extract_patches. Note
            that a singleton dimension is added to either end of these parameters for the
            batch and channels dimensions (so for a 2D images patches would be something
            like [32,32], which internally gets augmented to be [1,32,32,1]
        """

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        if transform_type == 'rigid':
            nb_params = 3 if ndims == 2 else 6
        elif transform_type == 'rigid+scale':
            nb_params = 4 if ndims == 2 else 7
        else:
            nb_params = ndims * (ndims + 1)

        # prepare rescaling matrix
        rescale_np = np.ones((nb_params,))
        if hasattr(rescale_affine, '__len__') and len(rescale_affine) == 2:
            if transform_type.startswith('rigid'):
                if ndims == 3:
                    rescale_np[:3] = rescale_affine[0]
                    rescale_np[3:] = rescale_affine[1]
                else:
                    rescale_np[:1] = rescale_affine[0]
                    rescale_np[1:] = rescale_affine[1]
            else:  # full affine
                rescale_np[0:ndims] = rescale_affine[0]
                rescale_np[ndims:] = rescale_affine[1]
        else:
            rescale_np *= rescale_affine 

        rescale_tensor = tf.cast(tf.convert_to_tensor(rescale_np), dtype=tf.float32)
        extract_patches = tf.image.extract_patches if ndims == 2 else tf.extract_volume_patches
        # configure base encoder CNN
        Conv = getattr(KL, 'Conv%dD' % (ndims + int(add_dim)))
        basenet = tf.keras.Sequential(name='core_model' + name)
        for nf in enc_nf:
            # if any entry in enc_nf is a list, then create a level with
            # multiple layers/level. The last one will always have strides=2 for downsampling
            if isinstance(nf, list):
                for nf1 in nf[0:-1]:
                    basenet.add(Conv(nf1, kernel_size=3, padding='same',
                                     kernel_initializer='he_normal', strides=1))
                    if dropout is not None:
                        basenet.add(KL.Dropout(dropout))
                    basenet.add(KL.LeakyReLU(0.2))
                nf = nf[-1]  # last layer at this level is added below with stride=2

            basenet.add(Conv(nf, kernel_size=3, padding='same',
                             kernel_initializer='he_normal', strides=2))
            if dropout is not None:
                basenet.add(KL.Dropout(dropout))
            basenet.add(KL.LeakyReLU(0.2))

        # dense layer to affine matrix scaled to make parameter estimation well-conditioned
        basenet.add(KL.Flatten())
        basenet.add(KL.Dense(nb_params, name='dense'))
        basenet.add(ne.layers.RescaleValues(rescale_tensor, name='rescale_affine'))
        basenet.add(vxm.layers.AffineTransformationsToMatrix(ndims, name='matrix_conversion'))

        # inputs
        if input_model is None:
            # configure default input layers if an input model is not provided
            source = tf.keras.Input(shape=[*inshape, nchannels], name='source_input' + name)
            target = tf.keras.Input(shape=[*inshape, nchannels], name='target_input' + name)
            input_model = tf.keras.Model(inputs=[source, target], outputs=[source, target])
        else:
            source, target = input_model.outputs[:2]

        if downsize_input is not None:
            source_res = Conv(1, kernel_size=3, padding='same',
                              kernel_initializer='he_normal', strides=downsize_input)(source)
            target_res = Conv(1, kernel_size=3, padding='same',
                              kernel_initializer='he_normal', strides=downsize_input)(target)
        else:
            source_res = source
            target_res = target

        # reformat the image into a set of overlapping patches and give these
        # to the conv layers instead of the original images (but still apply transform
        # to the images for use in the loss function)
        patch_size = [1] + patch_size + [1]
        strides = [1] + strides + [1]
        rates = [1] + rates + [1]
        if ndims == 2:
            lbd = lambda x: extract_patches(x, patch_size,
                                            strides=strides,
                                            rates=rates,
                                            padding='VALID',
                                            name='source_patches' + name)
            source_patches = KL.Lambda(lbd)(source_res)
            lbd = lambda x: extract_patches(x, patch_size,
                                            strides=strides,
                                            rates=rates,
                                            padding='VALID',
                                            name='target_patches' + name)
            target_patches = KL.Lambda(lbd)(target_res)
            pdims = (3, 1, 2)
        elif ndims == 3:
            lbd = lambda x: extract_patches(x, patch_size,
                                            strides=strides,
                                            padding='VALID',
                                            name='source_patches' + name)
            source_patches = KL.Lambda(lbd)(source_res)
            lbd = lambda x: extract_patches(x, patch_size,
                                            strides=strides,
                                            padding='VALID',
                                            name='target_patches' + name)
            target_patches = KL.Lambda(lbd)(target_res)
            pdims = (4, 1, 2, 3)

        # permute the patches that come from tf.image.extract_patches so that the
        # channels dimension is different patches
        source_perm = KL.Permute(pdims)(source_patches)
        target_perm = KL.Permute(pdims)(target_patches)
        new_shape = [patch_size[1], patch_size[2], np.prod(source_patches.shape[1:3].as_list())]
        new_shape = patch_size[1:ndims + 1] + [np.prod(source_patches.shape[1:ndims + 1].as_list())]
        new_shape[-1] *= nchannels
        if add_dim:
            new_shape += [1]
        source_patches_reshaped = KL.Reshape(new_shape)(source_perm)
        target_patches_reshaped = KL.Reshape(new_shape)(target_perm)

        nchannels = source_patches_reshaped.shape.as_list()[-(1 + int(add_dim))]
        print('arranging patches into %d channels' % nchannels)

        scale_affines = []
        full_affine = None
        y_source_patches = source_patches_reshaped

        # build net with multi-scales
        for blur_num, blur in enumerate(blurs):
            # get layer name prefix
            prefix = name + '_blur_%d_' % blur_num

            # set input and blur using gaussian kernel
            source_blur = ne.layers.GaussianBlur(
                blur, name=prefix + '_source_blur' + name)(y_source_patches)
            target_blur = ne.layers.GaussianBlur(
                blur, name=prefix + '_target_blur' + name)(target_patches_reshaped)

            # per-scale affine encoder
            curr_affine = basenet(KL.concatenate(
                [source_blur, target_blur], name=prefix + 'concat' + name))
            scale_affines.append(curr_affine)

            # compose affine at this scale
            if full_affine is None:
                full_affine = curr_affine
            else:
                full_affine = vxm.layers.ComposeTransform(
                    name=prefix + 'compose' + name)([full_affine, curr_affine])

            # provide new input for next scale
            y_source_res = vxm.layers.SpatialTransformer(
                name=prefix + 'transformer' + name,
                fill_value=fill_value)([source_res, full_affine])
            if ndims == 2:
                lbd = lambda x: extract_patches(x, patch_size,
                                                strides=strides,
                                                rates=rates,
                                                padding='VALID',
                                                name='source_patches_blur%d' % blur + name)
                source_patches = KL.Lambda(lbd)(y_source_res)
            else:  # ndims == 3, don't include rates
                lbd = lambda x: extract_patches(x, patch_size,
                                                strides=strides,
                                                padding='VALID',
                                                name='source_patches_blur%d' % blur + name)
                source_patches = KL.Lambda(lbd)(y_source_res)
            source_perm = KL.Permute(pdims)(source_patches)
            y_source_patches = KL.Reshape(new_shape)(source_perm)

        # scale affine to account for image rescaling
        if downsize_input is not None:
            scale = 1.0 / downsize_input
            scale_mat = lambda _: scale * np.eye(ndims, ndims + 1, dtype='float32')
            scale_affine = KL.Lambda(lambda x: tf.map_fn(
                scale_mat, x, fn_output_signature='float32'))(full_affine)

            full_affine = vxm.layers.ComposeTransform(
                name=prefix + 'compose' + name)([full_affine, scale_affine])

        # apply final transform to input image
        y_source = vxm.layers.SpatialTransformer(
            name=prefix + 'transformer' + name, fill_value=fill_value)([source, full_affine])

        # invert affine for bidirectional training
        if bidir:
            inv_affine = vxm.layers.InvertAffine(name='invert_affine' + name)(full_affine)
            y_target = vxm.layers.SpatialTransformer(
                name='neg_transformer' + name, fill_value=fill_value)([target, inv_affine])
            outputs = [y_source, y_target]
        else:
            outputs = [y_source]

        # initialize the keras model
        super().__init__(name=name, inputs=input_model.inputs, outputs=outputs)
        #        super().__init__(name=name, inputs=[source, target], outputs=outputs)

        # cache affines
        self.references = ne.tf.modelio.LoadableModel.ReferenceContainer()
        self.references.affine = full_affine
        self.references.scale_affines = scale_affines
        self.references.transform_type = transform_type
        self.references.bidir = bidir
        if bidir:
            self.references.inv_affine = inv_affine

    def get_registration_model(self):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        return tf.keras.Model(self.inputs, self.references.affine)

    def get_inv_registration_model(self):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        return tf.keras.Model(self.inputs, self.references.inv_affine)

    def register(self, src, trg):
        """
        Predicts the transform from src to trg tensors.
        """
        return self.get_registration_model().predict([src, trg])

    def apply_transform(self, src, trg, img, interp_method='linear'):
        """
        Predicts the transform from src to trg and applies it to the img tensor.
        """
        warp_model = self.get_registration_model()
        img_input = tf.keras.Input(shape=img.shape[1:])
        y_img = vxm.layers.SpatialTransformer(interp_method=interp_method)([
            img_input, warp_model.output])
        return tf.keras.Model(warp_model.inputs + [img_input], y_img).predict([src, trg, img])

    def apply_inv_transform(self, src, trg, img, interp_method='linear'):
        """
        Predicts the transform from trg to src and applies it to the img tensor.
        """
        warp_model = self.get_inv_registration_model()
        img_input = tf.keras.Input(shape=img.shape[1:])
        stin = [img_input, warp_model.output]
        y_img = vxm.layers.SpatialTransformer(interp_method=interp_method)(stin)
        return tf.keras.Model(warp_model.inputs + [img_input], y_img).predict([src, trg, img])

    def rescale_model(self, zoom_factor, interp_method='linear', fill_value=0):
        """
        Build and return a new model that computes the transform at the
        scale that was learned by the model, then rescales it so it can be
        applied to a different sized image (e.g. learning at 2x downsampling but
        applying at full res)

        Author: brf2
        """
        warnings.warn('brf: rescale_model will be moved to a utility from a method')
        # build new inputs that are scaled down and put the through old net
        shape = self.inputs[0].shape[1:].as_list()
        shape[0:-1] = list(np.array(shape[0:-1]) * zoom_factor)
        source_input = KL.Input(shape=shape, name='source_rescaled%d' % zoom_factor)
        target_input = KL.Input(shape=shape, name='target_rescaled%d' % zoom_factor)
        source_rescaled = ne.layers.Resize(1. / zoom_factor, interp_method)(source_input)
        target_rescaled = ne.layers.Resize(1. / zoom_factor, interp_method)(target_input)

        # put the scaled down inputs through the old net, then scale the affine
        # for full-scale images
        zoomed_inputs = [source_rescaled, target_rescaled]
        affine = self.get_registration_model()(zoomed_inputs)
        affine_rescaled = vxm.layers.RescaleTransform(zoom_factor, interp_method)(affine)

        # apply the full-scale tranform to the full-scale images
        stin = [source_input, affine_rescaled]
        source_transformed = vxm.layers.SpatialTransformer(interp_method='linear',
                                                           indexing='ij',
                                                           name='transformer',
                                                           fill_value=fill_value)(stin)
        outputs = [source_transformed]
        if self.references.bidir:
            inv_affine = vxm.layers.InvertAffine(name='invert_affine')(affine_rescaled)
            target_transformed = vxm.layers.SpatialTransformer(
                name='inv_transformer', fill_value=fill_value)([target_input, inv_affine])
            outputs += [target_transformed]

        # build the new model
        inputs = [source_input, target_input]
        new_model = tf.keras.models.Model(inputs, outputs)

        # propagate variables from old model to new one
        new_model.references = ne.tf.modelio.LoadableModel.ReferenceContainer()
        new_model.references.affine = affine_rescaled
        new_model.references.scale_affines = self.references.scale_affines
        new_model.references.transform_type = self.references.transform_type
        new_model.references.bidir = self.references.bidir
        if self.references.bidir:
            new_model.references.inv_affine = inv_affine

        return new_model


class VxmAffineEncoderThenDenseRegSegAtlasWithSeparateUnets(ne.tf.modelio.LoadableModel):
    """
    VoxelMorph network to perform combined affine registration and segmentation.
    inputs  are image, atlas image, atlas seg (one-hot)
    outputs = [moving, fixed, moving_seg_warped, fixed_seg_warped, both_segs]
       moving image xformed to fixed space
       atlas image xformed to moving space
       moving seg xformed to fixed space
       atlas seg xformed to moving space
       stack of xformed moving seg and (non xformed) atlas seg for overlap calc

    usual unet/vxm parameters (see e.g. VxmAffine or ne.models.unet)
    in addition seg_dilations specifies a number of dilations to apply
    to the outputs segmentations that make up the stacked output only,
    NOT to the individual segmentation outputs

    Author: brf2
    """
    @ne.modelio.store_config_args
    def __init__(self,
                 inshape,
                 transform_type='affine',
                 seg_channels=2,
                 im_channels=1,
                 model_affine=None,
                 unet=None,
                 enc_nf_affine=None,
                 affine_blurs=[1],
                 bidir=True,
                 rescale_affine=.1,
                 name='vxm_regseg_atlas_affine',
                 unet_features=None,
                 unet_levels=4,
                 unet_feat_mult=1,
                 unet_conv_per_level=2,
                 seg_dilations=0,
                 fill_value=0,
                 use_patches=True,
                 **kwargs):

        assert bidir, 'bidir = false not supported yet'

        if model_affine is None:
            if use_patches:
                vxm_class = vxms.networks.VxmAffineUnetDense
                model_affine = vxm_class(inshape,
                                         enc_nf_affine,
                                         bidir=bidir,
                                         transform_type=transform_type,
                                         rescale_affine=rescale_affine,
                                         name=name + '.affine_model',
                                         fill_value=fill_value,
                                         src_feats=seg_channels + im_channels,
                                         trg_feats=seg_channels + im_channels)
            else:
                vxm_class = vxms.networks.VxmAffineEncoderThenDense
                model_affine = vxm_class(inshape,
                                         enc_nf_affine,
                                         nchannels=seg_channels + im_channels,
                                         bidir=bidir,
                                         transform_type=transform_type,
                                         rescale_affine=rescale_affine,
                                         name='affine_model',
                                         blurs=affine_blurs)

        if unet is None:
            unet = ne.models.unet(
                unet_features,
                inshape + (im_channels,),
                unet_levels,
                len(inshape) * (5,),
                seg_channels,
                feat_mult=unet_feat_mult,
                nb_conv_per_level=unet_conv_per_level,
                name='regseg_unet'
            )

        # allocate the three inputs - moving,atlas_im, attlas_seg
        moving_input = KL.Input((*inshape, im_channels), name='unet_input_moving')
        atlas_im_input = KL.Input((*inshape, im_channels), name='atlas_im_input')
        atlas_seg_input = KL.Input((*inshape, seg_channels), name='atlas_seg_input')

        # create the segmentations by running them each through the unet
        moving_seg = unet(moving_input)

        # reshape the inputs to have an extra channel, then stack them
        # with the segmentations so they can be inputs to the affine
        moving_both = KL.Concatenate(axis=-1, name='moving_stack')([moving_input, moving_seg])
        atlas_both = KL.Concatenate(axis=-1, name='atlas_stack')([atlas_im_input, atlas_seg_input])
        affine_outputs = model_affine([moving_both, atlas_both])

        moving = KL.Lambda(lambda x: x[..., 0], name='moving')(affine_outputs[0])
        moving_seg_warped = KL.Lambda(lambda x: x[..., 1:], name='moving_seg')(affine_outputs[0])
        atlas = KL.Lambda(lambda x: x[..., 0], name='atlas')(affine_outputs[1])
        atlas_seg_warped = KL.Lambda(
            lambda x: x[..., 1:], name='atlas_seg_warped')(affine_outputs[1])
        new_shape = tuple(atlas_seg_input.shape.as_list()[1:] + [1])  # exclude batch
        ndims = len(inshape)
        if ndims == 2:
            pool = tf.keras.layers.MaxPool2D
        else:
            pool = tf.keras.layers.MaxPool3D

        # if dilations are reqeusted, do so now (essentially dilating labels)
        # dilate only frames 1: to avoid dilating background and eroding labels
        if seg_dilations >= 1:
            lbd = lambda x: KL.Concatenate(
                axis=-1)([x[..., 0:1], pool((seg_dilations,) * ndims,
                                            padding='same', strides=1)(x[..., 1:])])
            reshape_moving_input = KL.Lambda(
                lbd, name='moving_seg_pool_%d' % seg_dilations)(moving_seg_warped)
            lbd = lambda x: KL.Concatenate(
                axis=-1)([x[..., 0:1], pool((seg_dilations,) * ndims,
                                            padding='same', strides=1)(x[..., 1:])])
            reshape_atlas_input = KL.Lambda(lbd, name='atlas_seg_pool_%d' %
                                            seg_dilations)(atlas_seg_warped)
        else:
            reshape_moving_input = moving_seg_warped
            reshape_atlas_input = atlas_seg_input

        # create a new dimensions and stack the segs
        # so that a single loss function gets both of them
        if 0:
            reshape_moving = KL.Reshape(new_shape, name='moving_reshaped')(reshape_moving_input)
            reshape_atlas = KL.Reshape(new_shape, name='atlas_reshaped')(reshape_atlas_input)
            both_segs = KL.Concatenate(axis=-1, name='seg_stack')([reshape_moving, reshape_atlas])

        # don't output the atlas seg as the generator won't have the moving seg for training
        #        outputs = [moving, atlas, moving_seg_warped, atlas_seg_warped, both_segs]
        outputs = [moving, atlas, moving_seg_warped]
        inputs = [moving_input, atlas_im_input, atlas_seg_input]
        super().__init__(name=name, inputs=inputs, outputs=outputs)
        # cache pointers to layers and tensors for future reference
        self.references = ne.tf.modelio.LoadableModel.ReferenceContainer()
        self.references.unet_model = unet
        self.references.moving_seg = moving_seg
        self.references.moving_seg_warped = moving_seg_warped
        self.references.affine_model = model_affine
        self.references.affine = model_affine.references.affine
        #        self.references.scale_affines = model_affine.references.scale_affines
        self.references.transform_type = transform_type

    def get_moving_seg(self, inb):
        """
        Predicts the transform from src to trg and applies it to the img tensor.
        """
        outputs = [self.references.moving_seg]
        return tf.keras.models.Model(inputs=self.inputs, outputs=outputs).predict(inb)

    def get_moving_seg_warped(self, inb):
        """
        Predicts the transform from src to trg and applies it to the img tensor.
        """
        outputs = [self.references.moving_seg_warped]
        return tf.keras.models.Model(inputs=self.inputs, outputs=outputs).predict(inb)

    def get_registration_model(self):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        return self.references.affine_model.get_registration_model()

    def register(self, src, trg):
        """
        Predicts the transform from src to trg tensors.
        """
        return self.references.affine_model.register(src, trg)

    def apply_transform(self, src, trg, img, interp_method='linear'):
        """
        Predicts the transform from src to trg and applies it to the img tensor.
        """
        return self.references.affine_model.apply_transform(src, trg, img, interp_method)

    def apply_inv_transform(self, src, trg, img, interp_method='linear'):
        """
        Predicts the transform from src to trg and applies it to the img tensor.
        """
        return self.references.affine_model.apply_inv_transform(src, trg, img, interp_method)


class VxmAffineEncoderThenDenseComboDensePatches(ne.tf.modelio.LoadableModel):
    """
    VoxelMorph network to perform combined affine (with patches) and nonlinear registration.

    Author: brf2
    """

    @ne.modelio.store_config_args
    def __init__(self,
                 inshape,
                 nb_unet_features=None,
                 enc_nf_affine=None,
                 transform_type='affine',
                 affine_bidir=False,
                 affine_blurs=[1],
                 bidir=False,
                 affine_model=None,
                 rescale_affine=1.0,
                 fill_value=None,
                 nfeats=1,
                 patch_size=None,
                 strides=None,
                 nb_unet_conv_per_level=1,
                 **kwargs):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. 
                See VxmDense documentation for more information.
            enc_nf_affine: List of affine encoder filters.
                Default is None (uses unet encoder features).
            transform_type:  See VxmAffine for types. Default is 'affine'.
            affine_bidir: Enable bidirectional affine training. Default is False.
            affine_blurs: List of blurring levels for affine transform. Default is [1].
            bidir: Enable bidirectional cost function. Default is False.
            affine_model: Provide alternative affine model as inputs. Default is None.
            kwargs: Forwarded to the internal VxmDense model.
        """

        # default encoder and decoder layer features if nothing provided
        if nb_unet_features is None:
            nb_unet_features = vxm.networks.default_unet_features()

        # use dense-net encoder features if affine-net features aren't provided
        if enc_nf_affine is None:
            if isinstance(nb_unet_features, int):
                raise ValueError(
                    'enc_nf_affine list must be provided when nb_unet_features is an integer')
            enc_nf_affine = nb_unet_features[0]

        # affine component
        if affine_model is None:
            if patch_size is not None:
                affine_model = VxmAffineEncoderThenDensePatches(
                    inshape, enc_nf_affine,
                    bidir=bidir,
                    transform_type=transform_type,
                    rescale_affine=rescale_affine,
                    name='affine_model',
                    fill_value=fill_value,
                    patch_size=patch_size,
                    strides=strides,
                    nchannels=nfeats)
            else:
                affine_model = vxms.networks.VxmAffineUnetDense(
                    inshape, enc_nf_affine,
                    bidir=bidir,
                    transform_type=transform_type,
                    rescale_affine=rescale_affine,
                    name='affine_model',
                    fill_value=fill_value,
                    nb_unet_conv_per_level=nb_unet_conv_per_level,
                    src_feats=nfeats,
                    trg_feats=nfeats)

        source = affine_model.inputs[0]
        target = affine_model.inputs[1]
        affine = affine_model.references.affine

        # build a dense model that takes the affine transformed src as input
        dense_input_model = tf.keras.Model(affine_model.inputs, (affine_model.outputs[0], target))
        dense_model = vxm.networks.VxmDense(inshape, nb_unet_features=nb_unet_features,
                                            bidir=bidir,
                                            input_model=dense_input_model,
                                            nb_unet_conv_per_level=nb_unet_conv_per_level,
                                            src_feats=nfeats,
                                            trg_feats=nfeats, **kwargs)
        flow_params = dense_model.outputs[-1]
        pos_flow = dense_model.references.pos_flow

        # build a single transform that applies both affine and dense to src
        # and apply it to the input (src) volume so that there is only 1 interpolation
        # and output it as the combined model output (plus the dense warp)
        composed = vxm.layers.ComposeTransform()([affine, pos_flow])
        y_source = vxm.layers.SpatialTransformer(fill_value=fill_value)([source, composed])

        # invert and transform for bidirectional training
        if bidir:
            neg_flow = dense_model.references.neg_flow
            inv_affine = vxm.layers.InvertAffine()(affine)
            inv_composed = vxm.layers.ComposeTransform()([inv_affine, neg_flow])
            y_target = vxm.layers.SpatialTransformer(fill_value=fill_value)([target, inv_composed])
            outputs = [y_source, y_target, flow_params]
            outputs = [y_source, y_target, composed]
        else:
            outputs = [y_source, composed]

        # initialize the keras model
        super().__init__(inputs=affine_model.inputs, outputs=outputs)

        # cache pointers to layers and tensors for future reference
        self.references = ne.tf.modelio.LoadableModel.ReferenceContainer()
        self.references.affine = affine
        self.references.affine_model = affine_model
        self.references.dense_model = dense_model
        self.references.pos_flow = pos_flow
        self.references.composed = composed
        self.references.neg_flow = neg_flow if bidir else None
        self.references.inv_composed = inv_composed if bidir else None

    def get_split_registration_model(self):
        """
        Returns a reconfigured model to predict only the final affine and dense transforms.
        """
        return tf.keras.Model(self.inputs, [self.references.affine, self.references.pos_flow])

    def get_registration_model(self):
        """
        Returns a reconfigured model to predict only the final composed transform.
        """
        return tf.keras.Model(self.inputs, self.references.composed)

    def get_inv_registration_model(self):
        """
        Returns a reconfigured model to predict only the final composed transform.
        """
        return tf.keras.Model(self.inputs, self.references.inv_composed)

    def register(self, src, trg):
        """
        Predicts the transform from src to trg tensors.
        """
        return self.get_registration_model().predict([src, trg])

    def apply_transform(self, src, trg, img, interp_method='linear', fill_value=None):
        """
        Predicts the transform from src to trg and applies it to the img tensor.
        """
        warp_model = self.get_registration_model()
        img_input = tf.keras.Input(shape=img.shape[1:])
        y_img = vxm.layers.SpatialTransformer(interp_method=interp_method,
                                              fill_value=fill_value)([img_input, warp_model.output])
        return tf.keras.Model(warp_model.inputs + [img_input], y_img).predict([src, trg, img])

    def apply_inv_transform(self, src, trg, img, interp_method='linear', fill_value=None):
        """
        Predicts the transform from src to trg and applies it to the img tensor.
        """
        warp_model = self.get_inv_registration_model()
        img_input = tf.keras.Input(shape=img.shape[1:])
        y_img = vxm.layers.SpatialTransformer(interp_method=interp_method,
                                              fill_value=fill_value)([img_input, warp_model.output])
        return tf.keras.Model(warp_model.inputs + [img_input], y_img).predict([src, trg, img])


class VxmAffineEncoderThenDenseOld(AbstractVxmModel):
    """
    Original Voxelorph network for linear (affine) registration between two images.
    This is an old network using keras Sequential models. VxmAffineEncoderThenDenseNew should
    be used instead.

    Author: brf2
    """

    @ne.modelio.store_config_args
    def __init__(self, inshape, enc_nf,
                 bidir=False,
                 transform_type='affine',
                 blurs=[1],
                 rescale_affine=[1.0, .01],
                 nchannels=1,
                 name='vxm_affine',
                 fill_value=None,
                 reserve_capacity=None,
                 input_model=None):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            enc_nf: List of encoder filters. e.g. [16, 32, 32, 32]
            bidir: Enable bidirectional cost function. Default is False.
            transform_type: 'affine' (default), 'rigid' or 'rigid+scale' currently
            blurs: List of gaussian blur kernel levels for inputs. Default is [1].
            rescale_affine: A scalar or array to rescale the output of the affine matrix layer.
                Default is 1.0.
                This improves stability by enabling different gradient flow to affect the affine 
                parameters.
                Input array can match the transform matrix shape or it can be a 2-element list that
                represents individual [translation, linear] components.
            nchannels: Number of input channels. Default is 1.
            name: Model name. Default is 'vxm_affine'.
        """

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        if transform_type == 'rigid':
            nb_params = 2 * ndims
        elif transform_type == 'rigid+scale':
            nb_params = 2 * ndims + 1
        else:
            nb_params = ndims * (ndims + 1)

        # prepare rescaling matrix
        rescale_np = np.ones((nb_params,))
        if hasattr(rescale_affine, '__len__') and len(rescale_affine) == 2:
            if transform_type.startswith('rigid'):
                rescale_np[:3] = rescale_affine[0]
                rescale_np[3:] = rescale_affine[1]
            else:
                scaling = np.ones((ndims, ndims + 1), dtype='float32')
                scaling[:, -1] = rescale_affine[0]  # translation
                scaling[:, :-1] = rescale_affine[1]  # linear (everything else)
                rescale_np = scaling[np.newaxis].flatten()
        else:
            rescale_np *= rescale_affine 

        print(rescale_np)
        rescale_tensor = tf.cast(tf.convert_to_tensor(rescale_np), dtype=tf.float32)

        # configure base encoder CNN
        Conv = getattr(KL, 'Conv%dD' % ndims)
        basenet = tf.keras.Sequential(name=f'core_model_{name}')
        for nf in enc_nf:
            if isinstance(nf, list):
                for nf1 in nf[0:-1]:
                    basenet.add(Conv(nf1, kernel_size=3, padding='same',
                                     kernel_initializer='he_normal', strides=1))
                    basenet.add(KL.LeakyReLU(0.2))
                nf = nf[-1]  # last layer at this level is added below with stride=2

            basenet.add(Conv(nf, kernel_size=3, padding='same',
                             kernel_initializer='he_normal', strides=2))
            basenet.add(KL.LeakyReLU(0.2))

        # dense layer to affine matrix scaled to make parameter estimation well-conditioned
        basenet.add(KL.Flatten())

        if reserve_capacity is not None:
            basenet.add(vxms.layers.Mask(reserve_capacity, name='reserve'))

        init = KI.RandomUniform(minval=-1e-5, maxval=1e-5) 
        basenet.add(KL.Dense(nb_params, name='dense', kernel_initializer=init))
        basenet.add(ne.layers.RescaleValues(rescale_tensor, name='rescale_affine'))
        basenet.add(vxm.layers.AffineTransformationsToMatrix(ndims, name='matrix_conversion'))

        # inputs
        if input_model is None:
            # configure default input layers if an input model is not provided
            source = tf.keras.Input(shape=[*inshape, nchannels], name='source_input')
            target = tf.keras.Input(shape=[*inshape, nchannels], name='target_input')
            input_model = tf.keras.Model(inputs=[source, target], outputs=[source, target])
        else:
            source, target = input_model.outputs[:2]

        scale_affines = []
        full_affine = None
        y_source = source

        # build net with multi-scales
        for blur_num, blur in enumerate(blurs):
            # get layer name prefix
            prefix = 'blur_%d_%s' % (blur_num, name)

            # set input and blur using gaussian kernel
            source_blur = ne.layers.GaussianBlur(blur, name=prefix + '_source_blur')(y_source)
            target_blur = ne.layers.GaussianBlur(blur, name=prefix + '_target_blur')(target)

            # per-scale affine encoder
            curr_affine_prescaled = basenet(KL.concatenate(
                [source_blur, target_blur], name=prefix + 'concat'))
            #            curr_affine = ne.layers.RescaleValues(
            #                rescale_affine, name=prefix + 'rescale')(curr_affine_prescaled)
            curr_affine = curr_affine_prescaled
            scale_affines.append(curr_affine)

            # compose affine at this scale
            if full_affine is None:
                full_affine = curr_affine
            else:
                full_affine = vxm.layers.ComposeTransform(
                    name=prefix + 'compose')([full_affine, curr_affine])

            # provide new input for next scale
            y_source = vxm.layers.SpatialTransformer(
                name=prefix + 'transformer', fill_value=fill_value)([source, full_affine])

        # hack so that full_affine has a consistent and findable name
        full_affine = KL.Reshape(full_affine.get_shape().as_list()[1:],
                                 name=f'full_affine_{name}')(full_affine)

        # invert affine for bidirectional training
        if bidir:
            inv_affine = vxm.layers.InvertAffine(name=f'invert_affine_{name}')(full_affine)
            y_target = vxm.layers.SpatialTransformer(
                name=f'neg_transforme_r_{name}', fill_value=fill_value)([target, inv_affine])
            outputs = [y_source, y_target]
        else:
            outputs = [y_source]

        # initialize the keras model
        super().__init__(name=name, inputs=input_model.inputs, outputs=outputs)
        # super().__init__(name=name, inputs=[source, target], outputs=outputs)

        # cache affines
        self.references = ne.tf.modelio.LoadableModel.ReferenceContainer()
        self.references.affine = full_affine
        self.references.transform = full_affine
        self.references.pos_flow = full_affine
        self.references.scale_affines = scale_affines
        self.references.transform_type = transform_type
        self.references.bidir = bidir
        if bidir:
            self.references.neg_flow = inv_affine
            self.references.inv_affine = inv_affine
            self.references.inv_transform = inv_affine
