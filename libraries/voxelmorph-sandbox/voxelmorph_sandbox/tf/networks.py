"""
Experimental Networks for VoxelMorph (sandbox)

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
import pdb as gdb
from voxelmorph.py.utils import default_unet_features

# local (our) imports
import neurite as ne
import neurite_sandbox as nes
import voxelmorph as vxm
import voxelmorph_sandbox as vxms
from . import utils
from . import layers


###############################################################################
# Instance Affine models
###############################################################################

class InstanceAffine(ne.tf.modelio.LoadableModel):
    """
    VoxelMorph Affine network to perform instance-specific optimization.

    Tested a bit, but not super thoroughly.

    Example usecase:
        see utils.py

    Author: adalca
    """

    @ne.modelio.store_config_args
    def __init__(self, inshape, original_affine=None, affine_rescale=None, nb_feats=1):
        """
        # TODO: change inshape to volume shap e(without the final [1])
        Parameters:
            inshape: the keras input shape, usually volshape + [1]
        """
        self.inshape = inshape
        source = tf.keras.Input(shape=inshape + [nb_feats])

        # get some dimensions ready
        ndims = len(inshape)
        affine_shape = [ndims, ndims + 1]

        flow_layer = ne.layers.LocalParamWithInput(shape=affine_shape, name='affine_params')
        flow = flow_layer(source)
        if affine_rescale is not None:
            assert affine_rescale.shape == affine_shape, 'affine_rescale has bad shape'
            flow = KL.Lambda(lambda x: x * affine_rescale[np.newaxis, ...])(flow)

        # warp initial image
        y = vxm.layers.SpatialTransformer(name='warp')([source, flow])

        # initialize the keras model
        super().__init__(name='instance_net', inputs=source, outputs=[y])

        # initialize weights with original predicted warp
        if original_affine:
            flow_layer.set_weights([original_affine])

    def get_affine_model(self, return_dense=False):
        tensor = self.get_layer('warp').input[1]
        if return_dense:
            affine_to_dense_layer = vxm.layers.AffineToDense(self.inputs[0].shape[1:-1])
            tensor = affine_to_dense_layer(tensor)

        return tf.keras.Model(self.inputs, tensor)


class InstanceAffineWithEmbeddedData(ne.tf.modelio.LoadableModel):
    """
    VoxelMorph Affine network to perform instance-specific optimization,
    with the data to be registered embedded in the model.
    The original idea was that this might be faster (because of CPU-GPU moving of data),
    but it didn't make a huge different in a couple of quick runs.

    Tested a bit, but not super thoroughly.

    Author: adalca
    """

    @ne.modelio.store_config_args
    def __init__(self, moving, fixed, loss,
                 upsample_warp=None, original_affine=None, affine_rescale=None):
        """


        # TODO: change inshape to volume shap e(without the final [1])
        Parameters:
            indata: moving volume
            outdata: fixed volume
        """

        inp = tf.keras.Input(shape=[1], name='fake_input')
        self.moving = moving
        self.fixed = fixed
        lambda_fcn = lambda x: x * 0 + tf.convert_to_tensor(moving)[tf.newaxis, ..., tf.newaxis]
        source = KL.Lambda(lambda_fcn, name='source')(inp)
        self.source = source

        # get some dimensions ready
        ndims = len(moving.shape)
        affine_shape = [ndims, ndims + 1]

        flow_layer = ne.layers.LocalParamWithInput(shape=affine_shape, name='affine_params')
        flow = flow_layer(inp)
        if affine_rescale is not None:
            assert affine_rescale.shape == affine_shape, 'affine_rescale has bad shape'
            flow = KL.Lambda(lambda x: x * affine_rescale[np.newaxis, ...])(flow)

        if upsample_warp is not None:
            sz = [f // upsample_warp for f in moving.shape]
            dense_warp_small = vxm.layers.AffineToDense(sz)(flow)
            flow = vxm.layers.RescaleTransform(upsample_warp)(dense_warp_small)

        # warp initial image
        moved = vxm.layers.SpatialTransformer(name='warp')([source, flow])
        self.moved = moved

        lossfn = lambda x: K.mean(loss(tf.convert_to_tensor(fixed)[tf.newaxis, ..., tf.newaxis], x))
        y = KL.Lambda(lossfn)(moved)

        # initialize the keras model
        super().__init__(name='instance_net', inputs=inp, outputs=[y])

        # initialize weights with original predicted warp
        if original_affine:
            flow_layer.set_weights([original_affine])

    def get_affine_model(self, return_dense=False, upsample_warp=None):
        tensor = self.get_layer('warp').input[1]

        if return_dense:
            affine_to_dense_layer = vxm.layers.AffineToDense(self.source.shape[1:-1])
            tensor = affine_to_dense_layer(tensor)

            if upsample_warp is not None:
                tensor = vxm.layers.RescaleTransform(upsample_warp)(tensor)

        return tf.keras.Model(self.inputs, tensor)

    def get_warped_model(self):
        return tf.keras.Model(self.inputs, self.moved)


class InstanceAffineDenseIndependent(ne.tf.modelio.LoadableModel):
    """
    VoxelMorph network to perform instance-specific optimization, estimating both affine and
    deformable deformations.

    This implementaion is single-scale.

    TODO: To have a multi-scale implementation, could loop this model, or (more integrated)
    create a cascade. If you do the latter, could ask for both images as 'input' and could apply
    the blur/downscaling to both.
    - could also use optimization layers
    - could flow through the entire cascade network, but is probably wasteful in terms of timing.
    - alternatively, could use masking trick, but again this is probably slower.

    Author: adalca
    """

    @ne.modelio.store_config_args
    def __init__(self, inshape, feats=1, affmult=100, flowmult=1000):

        source = tf.keras.Input(shape=(*inshape, feats))

        # get some dimensions ready
        ndims = len(inshape)
        affine_shape = [ndims, ndims + 1]

        # learn affine component
        affine_layer = ne.layers.LocalParamWithInput(shape=affine_shape,
                                                     name='affine_params',
                                                     mult=affmult)
        affine = affine_layer(source)

        # deformable flow (on top of the affine)
        flow_layer = ne.layers.LocalParamWithInput(shape=(*inshape, len(inshape)),
                                                   name='phi',
                                                   mult=flowmult)
        flow = flow_layer(source)

        # joint flow
        joint_flow = vxm.layers.ComposeTransform(name='joint')([affine, flow])

        # prepare outputs.
        y_affine = vxm.layers.SpatialTransformer()([source, affine])
        y = vxm.layers.SpatialTransformer()([source, joint_flow])

        # initialize the keras model.
        # Final output is "flow" (deformable) only, to penalize spatial gradient
        super().__init__(name='instance_dense', inputs=[source], outputs=[y_affine, y, flow])

        # cache pointers to important layers and tensors for future reference
        self.references = ne.tf.modelio.LoadableModel.ReferenceContainer()
        self.references.flow_layer = flow_layer

    def set_flow(self, warp):
        '''
        Sets the networks flow field weights.
        '''
        self.references.flow_layer.set_weights(warp)


class InstanceAffineDenseDecompose(ne.tf.modelio.LoadableModel):
    """
    VoxelMorph network to perform instance-specific optimization.

    Author: adalca
    """

    @ne.modelio.store_config_args
    def __init__(self, inshape, feats=1, weighted=False):

        source = tf.keras.Input(shape=(*inshape, feats))
        flow_layer = ne.layers.LocalParamWithInput(shape=(*inshape, len(inshape)), name='phi')
        flow = flow_layer(source)

        if not weighted:
            # extract dense warp matching affine component of current dense
            # The current utility function decompose_warp would also return the reconstruction
            # itself but it's better to work with the affine so we can invert it.
            dec_aff = lambda x: utils.decompose_warp(x, return_recon=False, include_identity=False)
            decompose_warp = lambda phi: tf.map_fn(
                dec_aff,
                phi,
                fn_output_signature=(tf.float32, tf.float32))
            flow_affine, flow_dense_diff = KL.Lambda(decompose_warp, name='decompose_warp')(flow)

        else:
            # weights are *only* used for computing affines.
            weights = ne.layers.LocalParamWithInput(shape=(*inshape, 1), name='weights')(source)

            dec_aff = lambda x: utils.decompose_warp(x[0], return_recon=False,
                                                     include_identity=False, weights=x[1])
            decomposed_affine = lambda x: tf.map_fn(
                dec_aff, x, fn_output_signature=(tf.float32, tf.float32))
            flow_affine, flow_dense_diff = KL.Lambda(decomposed_affine,
                                                     name='decompose_warp')([flow, weights])

        y = vxm.layers.SpatialTransformer()([source, flow])
        y_affine = vxm.layers.SpatialTransformer()([source, flow_affine])

        # initialize the keras model
        super().__init__(name='instance_dense',
                         inputs=[source],
                         outputs=[y_affine, y, flow_dense_diff])

        # cache pointers to important layers and tensors for future reference
        self.references = ne.tf.modelio.LoadableModel.ReferenceContainer()
        self.references.flow_layer = flow_layer

    def set_flow(self, warp):
        '''
        Sets the networks flow field weights.
        '''
        self.references.flow_layer.set_weights(warp)


class InstanceRotate(ne.tf.modelio.LoadableModel):
    """
    VoxelMorph Rotate network to perform instance-specific optimization.

    Tested a bit, but not super thoroughly.

    Example usecase:
        see utils.py

    Author: adalca
    """

    @ne.modelio.store_config_args
    def __init__(self, inshape, original_affine=None, affine_rescale=None):
        """
        # TODO: change inshape to volume shap e(without the final [1])
        Parameters:
            inshape: the keras input shape, usually volshape + [1]
        """
        self.inshape = list(inshape)
        source = tf.keras.Input(shape=inshape + [1])

        # get some dimensions ready
        ndims = len(inshape)
        affine_shape = [1]

        flow_layer = ne.layers.LocalParamWithInput(shape=affine_shape, name='rotation')
        rot = flow_layer(source)

        def angle_to_aff(x):
            a = vxms.utils.angle_to_affine_matrix(x, is_2d=ndims == 2)
            aff = tf.concat([a - tf.eye(ndims), tf.zeros((ndims, 1))], 1)
            return aff

        flow = KL.Lambda(lambda x: tf.map_fn(angle_to_aff, [x], fn_output_signature=tf.float32),
                         name='affine')(rot)
        print(rot, flow)

        if affine_rescale is not None:
            assert affine_rescale.shape == affine_shape, 'affine_rescale has bad shape'
            flow = KL.Lambda(lambda x: x * affine_rescale[np.newaxis, ...])(flow)

        # warp initial image
        y = vxm.layers.SpatialTransformer(name='warp')([source, flow])

        # initialize the keras model
        super().__init__(name='instance_net', inputs=source, outputs=[y])

        # initialize weights with original predicted warp
        if original_affine:
            flow_layer.set_weights([original_affine])

    def get_affine_model(self, return_dense=False):
        tensor = self.get_layer('warp').input[1]
        if return_dense:
            affine_to_dense_layer = vxm.layers.AffineToDense(self.inputs[0].shape[1:-1])
            tensor = affine_to_dense_layer(tensor)

        return tf.keras.Model(self.inputs, tensor)


###############################################################################
# Affine models
###############################################################################


class AbstractVxmModel(ne.tf.modelio.LoadableModel):
    """
    An abstract base class for vxm models with predefined utilities. All
    inheriting models are expected to at least have the reference variable
    `references.transform` pointing to the final transform tensor (whether
    it be an affine matrix or dense warp).
    """

    def get_registration_model(self):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        return tf.keras.Model(self.inputs, self.references.transform)

    def get_inv_registration_model(self):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        return tf.keras.Model(self.inputs, self.references.inv_transform)

    def register(self, src, trg):
        """
        Predicts the transform from src to trg tensors.
        """
        return self.get_registration_model().predict([src, trg])

    def apply_transform(self, src, trg, img, interp_method='linear'):
        """
        Predicts the transform from src to trg and applies it to the img tensor.
        """
        model = self.get_registration_model()
        moving = tf.keras.Input(shape=img.shape[1:])
        mvd = vxm.layers.SpatialTransformer(interp_method=interp_method)([moving, model.output])
        return tf.keras.Model([*model.inputs, moving], mvd).predict([src, trg, img])

    def apply_inv_transform(self, src, trg, img, interp_method='linear'):
        """
        Predicts the inverse transform from src to trg and applies it to the img tensor.
        """
        model = self.get_inv_registration_model()
        moving = tf.keras.Input(shape=img.shape[1:])
        mvd = vxm.layers.SpatialTransformer(interp_method=interp_method)([moving, model.output])
        return tf.keras.Model([*model.inputs, moving], mvd).predict([src, trg, img])

    def rescale_model(self, *args, **kwargs):
        """
        alias to get_rescaled_registration_model()
        """
        return self.get_rescaled_registration_model(*args, **kwargs)

    def get_rescaled_registration_model(self,
                                        zoom_factor,
                                        interp_method='linear',
                                        resize_interp='linear',
                                        fill_value=0):
        """
        Builds a new model that computes the transform at the scale that was learned
        by the model and rescales it to be applied to a different sized image. For
        example: learning at 2x downsampling, but applying at full resolution.

        ATH: right now this is somewhat specific to affine models, but it can definitely
        be generalized
        a bit. See comments below - I just don't want to mess up anything BRF has right now.
        """

        # ATH: I'm not so sure, I think it could make sense here?
        warnings.warn('brf: rescale_model will be moved to a utility from a method')

        # determine scaled shape
        shape = self.inputs[0].shape[1:].as_list()
        shape[0:-1] = list(np.array(shape[0:-1]) * zoom_factor)

        # build new input tensors
        source_input = KL.Input(shape=shape, name='input_source_rescaled_%dx' % zoom_factor)
        target_input = KL.Input(shape=shape, name='input_target_rescaled_%dx' % zoom_factor)

        # scale the inputs
        source_rescaled = ne.layers.Resize(1. / zoom_factor, resize_interp)(source_input)
        target_rescaled = ne.layers.Resize(1. / zoom_factor, resize_interp)(target_input)

        # run through the original network, then scale the resulting transform
        affine = self.get_registration_model()([source_rescaled, target_rescaled])
        # todo this has an interpolation?
        affine_rescaled = vxm.layers.RescaleTransform(zoom_factor, resize_interp)(affine)

        # apply the target-scale transform to the target-scale images
        stin = [source_input, affine_rescaled]
        source_transformed = vxm.layers.SpatialTransformer(
            interp_method=interp_method,
            indexing='ij',
            name='transformer',
            fill_value=fill_value)(stin)

        # ATH: I'm not sure what the common use case is, but I imagine this should be an option.
        outputs = [source_transformed]
        if self.references.bidir:
            inv_affine = vxm.layers.InvertAffine(name='invert_affine')(affine_rescaled)
            target_transformed = vxm.layers.SpatialTransformer(
                name='inv_transformer', fill_value=fill_value)([target_input, inv_affine])
            outputs += [target_transformed]

        # build the new model
        model = tf.keras.models.Model([source_input, target_input], outputs)

        # propagate variables from old model to new one
        # ATH: this should be generalized, or dealt with a bit differently
        model.references = ne.tf.modelio.LoadableModel.ReferenceContainer()
        model.references.affine = affine_rescaled
        model.references.scale_affines = self.references.scale_affines
        model.references.transform_type = self.references.transform_type
        model.references.bidir = self.references.bidir
        if self.references.bidir:
            model.references.inv_affine = inv_affine

        return model


class VxmAffineWeightedUnet(ne.tf.modelio.LoadableModel):
    """
    VoxelMorph network for linear (affine) registration between two images.

    Experimental idea: weighted averaging of UNet output

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
                 bidir=False,
                 transform_type='affine',
                 blurs=[1],
                 rescale_affine=1.0, nchannels=1, name='vxm_affine'):
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
        """

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure base encoder CNN
        Conv = getattr(KL, 'Conv%dD' % ndims)

        # configure default input layers if an input model is not provided
        source = tf.keras.Input(shape=(*inshape, src_feats), name='source_input')
        target = tf.keras.Input(shape=(*inshape, trg_feats), name='target_input')
        input_model = tf.keras.Model(inputs=[source, target], outputs=[source, target])

        # build core unet model and grab inputs
        unet_model = vxm.networks.Unet(
            input_model=input_model,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level
        )

        latest_tensor = unet_model.output

        if transform_type == 'rigid':
            nb_params = 3 if ndims == 2 else 6
        elif transform_type == 'rigid+scale':
            nb_params = 4 if ndims == 2 else 7
        else:
            nb_params = ndims * (ndims + 1)

        params = Conv(nb_params,
                      kernel_size=3,
                      padding='same',
                      kernel_initializer=KI.RandomNormal(mean=0.0, stddev=1e-5),
                      strides=1,
                      name='params_out')(latest_tensor)
        weights = Conv(nb_params,
                       kernel_size=3,
                       padding='same',
                       kernel_initializer=KI.RandomNormal(mean=1.0, stddev=1e-5),
                       activation='relu',
                       strides=1,
                       name='weights_out')(latest_tensor)

        sum_axes = list(range(1, ndims + 1))
        weighted_vol_avg_fn = lambda x: tf.math.divide_no_nan(
            tf.reduce_sum(x[0] * x[1], axis=sum_axes),
            tf.math.reduce_sum(x[1], axis=sum_axes))
        final_affine = KL.Lambda(weighted_vol_avg_fn)([params, weights])

        # prepare rescaling matrix
        rescale_np = np.ones((nb_params,))
        if hasattr(rescale_affine, '__len__') and len(rescale_affine) == 2:
            if transform_type.startswith('rigid'):
                rescale_np[:ndims] = rescale_affine[0]
                rescale_np[ndims:] = rescale_affine[1]
            else:
                scaling = np.ones((ndims, ndims + 1), dtype='float32')
                scaling[:, -1] = rescale_affine[0]  # translation
                scaling[:, :-1] = rescale_affine[1]  # linear (everything else)
                rescale_np = scaling[np.newaxis].flatten()
        else:   # a single scalar applied to all parameters
            rescale_np *= rescale_affine

        rescaled_affine = ne.layers.RescaleValues(
            rescale_np, name='affine_rescale_' + name)(full_affine)

        if transform_type in ['rigid', 'rigid+scale']:
            full_affine = vxm.layers.ParamsToAffineMatrix(
                ndims, name='matrix_conversion')(final_affine)
        else:
            full_affine = final_affine

        y_source = vxm.layers.SpatialTransformer(name='transformer')([source, full_affine])

        # invert affine for bidirectional training
        if bidir:
            inv_affine = vxm.layers.InvertAffine(name='invert_affine')(full_affine)
            y_target = vxm.layers.SpatialTransformer(name='neg_transformer')([target, inv_affine])
            outputs = [y_source, y_target]
        else:
            outputs = [y_source]

        # initialize the keras model
        super().__init__(name=name, inputs=[source, target], outputs=outputs)

        # cache affines
        self.references = ne.tf.modelio.LoadableModel.ReferenceContainer()
        self.references.affine = full_affine
        self.references.transform_type = transform_type

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
        prin = [src, trg, img]
        return tf.keras.Model(warp_model.inputs + [img_input], y_img).predict(prin, **kwargs)


class VxmAffineDecoder(ne.tf.modelio.LoadableModel):
    """
    Experimental affine model based on a UNet and a Dense (FC) Layer with Dropout
    """

    @ne.modelio.store_config_args
    def __init__(self,
                 inshape,
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 int_steps=0,
                 int_downsize=2,
                 bidir=False,
                 use_probs=False,
                 src_feats=1,
                 trg_feats=1,
                 max_pool=2,
                 unet_half_res=False,
                 input_model=None,
                 dropout=0.99,
                 **kwargs):
        """
        Parameters:
                      inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer.
                If None (default), the unet features are defined by the default config described in
                the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_unet_features is an integer.
                Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_unet_features is an
                integer. Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when
                this value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration.
                The flow field is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            input_model: Model to replace default input layer before concatenation. Default is None.
        """

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        if input_model is None:
            # configure default input layers if an input model is not provided
            source = tf.keras.Input(shape=(*inshape, src_feats),
                                    name='source_input')
            target = tf.keras.Input(shape=(*inshape, trg_feats),
                                    name='target_input')
            input_model = tf.keras.Model(inputs=[source, target],
                                         outputs=[source, target])
        else:
            source, target = input_model.outputs[:2]

        # build core unet model and grab inputs
        unet_model = vxm.networks.Unet(
            input_model=input_model,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            max_pool=max_pool,
            **kwargs,
        )

        deepest_layer_name = [
            f.name for f in unet_model.layers if f.name.startswith('unet_enc_pooling')][-1]

        # transform unet output into a flow field
        uo = KL.Flatten()(unet_model.get_layer(deepest_layer_name).output)
        uo = KL.Dropout(dropout)(uo)
        flow_affine = KL.Dense(ndims * (ndims + 1), name='affine', activation=None,
                               kernel_initializer=KI.RandomNormal(mean=0.0, stddev=1e-7),
                               bias_initializer=KI.RandomNormal(mean=0.0, stddev=1e-7))(uo)
        flow_affine = KL.Reshape((ndims, ndims + 1))(flow_affine)

        # warp image with flow field
        st_layer = vxm.layers.SpatialTransformer(
            interp_method='linear', indexing='ij', name='transformer', fill_value=0)
        y_source = st_layer([source, flow_affine])
        outputs = [y_source]

        # if bidirection is required
        if bidir:
            inv_affine = vxm.layers.InvertAffine(name='invert_affine')(flow_affine)
            y_target = vxm.layers.SpatialTransformer(
                name='neg_transformer', fill_value=0)([target, inv_affine])
            outputs = [y_source, y_target]

        # initialize the keras model
        super().__init__(name='vxm_dense', inputs=input_model.inputs, outputs=outputs)

        # cache pointers to layers and tensors for future reference
        self.references = ne.tf.modelio.LoadableModel.ReferenceContainer()
        self.references.unet_model = unet_model
        self.references.y_source = y_source
        self.references.y_target = y_target if bidir else None
        self.references.pos_flow = flow_affine
        self.references.neg_flow = inv_affine if bidir else None

    def get_registration_model(self):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        assert 0, 'bug'
        return tf.keras.Model(self.inputs, self.references.pos_flow)

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


class VxmAffineFCDropout(ne.tf.modelio.LoadableModel):
    """
    Experimental affine model based on a UNet and a Dense (FC) Layer with Dropout
    """

    @ne.modelio.store_config_args
    def __init__(self,
                 inshape,
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 int_steps=0,
                 int_downsize=2,
                 bidir=False,
                 use_probs=False,
                 src_feats=1,
                 trg_feats=1,
                 max_pool=2,
                 unet_half_res=False,
                 input_model=None,
                 dropout=0.99,
                 **kwargs):
        """
        Parameters:
                      inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer.
                If None (default), the unet features are defined by the default config described in
                the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_unet_features is an integer.
                Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_unet_features is an
                integer. Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when
                this value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration.
                The flow field is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            input_model: Model to replace default input layer before concatenation. Default is None.
        """

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        if input_model is None:
            # configure default input layers if an input model is not provided
            source = tf.keras.Input(shape=(*inshape, src_feats), name='source_input')
            target = tf.keras.Input(shape=(*inshape, trg_feats), name='target_input')
            input_model = tf.keras.Model(inputs=[source, target], outputs=[source, target])
        else:
            source, target = input_model.outputs[:2]

        # build core unet model and grab inputs
        print(vxm.networks)
        unet_model = vxm.networks.Unet(
            input_model=input_model,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            max_pool=max_pool,
            **kwargs,
        )

        # transform unet output into a flow field
        uo = KL.Flatten()(unet_model.output)
        uo = KL.Dropout(dropout)(uo)
        flow_affine = KL.Dense(ndims * (ndims + 1), name='affine', activation=None,
                               kernel_initializer=KI.RandomNormal(mean=0.0, stddev=1e-7),
                               bias_initializer=KI.RandomNormal(mean=0.0, stddev=1e-7))(uo)
        flow_affine = KL.Reshape((ndims, ndims + 1))(flow_affine)

        # warp image with flow field
        st_layer = vxm.layers.SpatialTransformer(
            interp_method='linear', indexing='ij', name='transformer', fill_value=0)
        y_source = st_layer([source, flow_affine])
        outputs = [y_source]

        # if bidirection is required
        if bidir:
            inv_affine = vxm.layers.InvertAffine(name='invert_affine')(flow_affine)
            y_target = vxm.layers.SpatialTransformer(
                name='neg_transformer', fill_value=0)([target, inv_affine])
            outputs = [y_source, y_target]

        # initialize the keras model
        super().__init__(name='vxm_dense', inputs=input_model.inputs, outputs=outputs)

        # cache pointers to layers and tensors for future reference
        self.references = ne.tf.modelio.LoadableModel.ReferenceContainer()
        self.references.unet_model = unet_model
        self.references.y_source = y_source
        self.references.y_target = y_target if bidir else None
        self.references.pos_flow = flow_affine
        self.references.neg_flow = inv_affine if bidir else None

    def get_registration_model(self):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        assert 0, 'bug'
        return tf.keras.Model(self.inputs, self.references.pos_flow)

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


class VxmAffineDecomposeWarp(ne.tf.modelio.LoadableModel):
    """
    Experimental affine model based on decomposing dense warp into affine and (left-over) dense
    """

    @ne.modelio.store_config_args
    def __init__(self,
                 inshape,
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 int_steps=0,
                 int_downsize=2,
                 bidir=False,
                 use_probs=False,
                 src_feats=1,
                 trg_feats=1,
                 max_pool=2,
                 unet_half_res=False,
                 fill_value=0,
                 input_model=None,
                 weighted=False,
                 use_midspace=False,
                 do_inv=False,
                 hackmask=True,
                 **kwargs):
        """
        Parameters:
                      inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer.
                If None (default), the unet features are defined by the default config described in
                the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_unet_features is an integer.
                Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_unet_features is an
                integer. Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when
                this value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration.
                The flow field is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            use_midspace: compute midspace (matrix sqrt) affine transforms
            and output a stacked tensor of source and target mapped to this space
            input_model: Model to replace default input layer before concatenation. Default is None.
        """

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        if input_model is None:
            # configure default input layers if an input model is not provided
            source = tf.keras.Input(shape=(*inshape, src_feats), name='source_input')
            target = tf.keras.Input(shape=(*inshape, trg_feats), name='target_input')
            input_model = tf.keras.Model(inputs=[source, target], outputs=[source, target])
        else:
            source, target = input_model.outputs[:2]

        # build core unet model and grab inputs
        print(vxm.networks)
        unet_model = vxm.networks.Unet(
            input_model=input_model,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            max_pool=max_pool,
            **kwargs,
        )

        # transform unet output into a flow field
        Conv = getattr(KL, 'Conv%dD' % ndims)
        flow = Conv(ndims,
                    kernel_size=3,
                    padding='same',
                    kernel_initializer=KI.RandomNormal(mean=0.0, stddev=1e-5),
                    name='flow')(unet_model.output)

        if int_steps > 0:
            flow = vxm.layers.VecInt(int_steps=5, name='flow_int')(flow)

        # extract dense warp matching affine component of current dense
        # The current utility function decompose_warp would also return the reconstruction itself
        # but it's better to work with the affine so we can invert it.
        if not weighted:
            dec_aff_fn = lambda x: utils.decompose_warp(
                x, return_recon=False, include_identity=True)[0]
            decomposed_affine = lambda phi: tf.map_fn(
                dec_aff_fn, phi, fn_output_signature=tf.float32)
            flow_affine = KL.Lambda(decomposed_affine, name='decompose_warp')(flow)

        else:
            if not hackmask:
                weights = Conv(1,
                               kernel_size=3,
                               padding='same',
                               kernel_initializer=KI.RandomNormal(mean=0.0, stddev=1e-5),
                               activation='relu',
                               name='weights')(unet_model.output)
            else:
                if do_inv:
                    weights = KL.Lambda(lambda x: tf.cast(x > 0.001, tf.float32),
                                        name='weights_hack')(source)
                else:
                    weights = KL.Lambda(lambda x: tf.cast(x > 0.001, tf.float32),
                                        name='weights_hack')(target)

            dec_aff_fn = lambda x: utils.decompose_warp(
                x[0], return_recon=False, include_identity=True, weights=x[1])[0]
            decomposed_affine = lambda x: tf.map_fn(dec_aff_fn, x, fn_output_signature=tf.float32)
            flow_affine = KL.Lambda(decomposed_affine, name='decompose_warp')([flow, weights])

        if do_inv:
            # this option assumes that the affine fitted in the LSQ was the inverse affine,
            # which is a subtle but important aspect
            flow_affine = vxm.layers.InvertAffine(name='fix_affine')(flow_affine)

        # warp image with flow field
        st_layer = vxm.layers.SpatialTransformer(
            interp_method='linear', indexing='ij', name='transformer', fill_value=fill_value)
        y_source = st_layer([source, flow_affine])
        outputs = [y_source]

        # if bidirection is required
        if bidir:
            inv_affine = vxm.layers.InvertAffine(name='invert_affine')(flow_affine)
            y_target = vxm.layers.SpatialTransformer(
                name='neg_transformer', fill_value=fill_value)([target, inv_affine])
            outputs = [y_source, y_target]

            # need to move this to a function that can be called by other affines
            if use_midspace:
                affine_fwd = KL.Reshape((ndims, ndims + 1), name='fwd_reshape')(flow_affine)
                affine_inv = KL.Reshape((ndims, ndims + 1), name='inv_reshape')(inv_affine)
                Id = tf.eye(ndims + 1)  # have to add identity back in before sqrt
                row_aug = tf.zeros((1, ndims + 1))
                aug_and_sqrt = lambda x: tf.linalg.sqrtm(
                    tf.add(Id, tf.concat([x, row_aug], axis=0)))[0:3, :]
                half_fwd = KL.Lambda(lambda x: tf.map_fn(
                    aug_and_sqrt, x), name='half_fwd')(affine_fwd)
                half_inv = KL.Lambda(lambda x: tf.map_fn(
                    aug_and_sqrt, x), name='half_inv')(affine_inv)
                st_layer_id = vxm.layers.SpatialTransformer(
                    interp_method='linear',
                    indexing='ij',
                    name='transformer_id',
                    fill_value=fill_value,
                    add_identity=False)
                half_mov = st_layer_id([source, half_fwd])
                half_fix = st_layer_id([target, half_inv])
                half_out = KL.Concatenate(name='midspace', axis=-1)([half_mov, half_fix])
                outputs += [half_out]

        # initialize the keras model
        super().__init__(name='vxm_dense', inputs=input_model.inputs, outputs=outputs)

        # cache pointers to layers and tensors for future reference
        self.references = ne.tf.modelio.LoadableModel.ReferenceContainer()
        self.references.unet_model = unet_model
        self.references.y_source = y_source
        self.references.y_target = y_target if bidir else None
        self.references.pos_flow = flow_affine
        self.references.neg_flow = inv_affine if bidir else None
        self.references.bidir = bidir
        self.references.affine = flow_affine
        self.references.inv_affine = inv_affine if bidir else None
        self.references.half_fwd = half_fwd if use_midspace else None
        self.references.half_inv = half_inv if use_midspace else None

    def get_registration_model(self):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        # assert 0, 'bug'
        return tf.keras.Model(self.inputs, self.references.pos_flow)

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

    def apply_midspace_transform(self, src, trg, img_mov, img_fix,
                                 interp_method='linear', fill_value=0):
        """
        Predicts the transform from src to trg and applies the mispace
        fwd and inverse ones to the two image
        """
        fwd_warp_model = tf.keras.Model(self.inputs, self.references.half_fwd)
        inv_warp_model = tf.keras.Model(self.inputs, self.references.half_inv)
        fwd_img_input = tf.keras.Input(shape=img_mov.shape[1:])
        inv_img_input = tf.keras.Input(shape=img_fix.shape[1:])
        fwd_img = vxm.layers.SpatialTransformer(
            interp_method=interp_method, fill_value=fill_value, add_identity=False)(
                [fwd_img_input, fwd_warp_model.output])
        inv_img = vxm.layers.SpatialTransformer(
            interp_method=interp_method, fill_value=fill_value, add_identity=False)([
                inv_img_input, inv_warp_model.output])
        fwd_model = tf.keras.Model(fwd_warp_model.inputs + [fwd_img_input], fwd_img)
        inv_model = tf.keras.Model(inv_warp_model.inputs + [inv_img_input], inv_img)
        fwd = fwd_model.predict([src, trg, img_mov])
        inv = inv_model.predict([src, trg, img_fix])
        return fwd, inv

    def rescale_model(self, zoom_factor, interp_method='linear', fill_value=0):
        """
        Build and return a new model that computes the transform at the
        scale that was learned by the model, then rescales it so it can be
        applied to a different sized image (e.g. learning at 2x downsampling but
        applying at full res)

        Author: brf2
        """
        # build new inputs that are scaled down and put the through old net
        warnings.warn('brf: rescale_model will be moved to a utility from a method')
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

        # build the new model
        outputs += [target_transformed]
        inputs = [source_input, target_input]
        new_model = tf.keras.models.Model(inputs, outputs)

        # propagate variables from old model to new one
        new_model.references = ne.tf.modelio.LoadableModel.ReferenceContainer()
        new_model.references.affine = affine_rescaled
        new_model.references.pos_flow = affine_rescaled
        # new_model.references.scale_affines = self.references.scale_affines
        new_model.references.bidir = self.references.bidir
        if self.references.bidir:
            new_model.references.inv_affine = inv_affine
            new_model.references.neg_flow = inv_affine

        return new_model


###############################################################################
# Joint Decompose and Affine models
###############################################################################

class VxmDecomposeWarp(ne.tf.modelio.LoadableModel):
    """
    Experimental model based on decomposing dense warp into affine and (left-over) dense
    """

    @ne.modelio.store_config_args
    def __init__(self,
                 inshape,
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 int_steps=7,
                 int_downsize=2,
                 bidir=False,
                 use_probs=False,
                 src_feats=1,
                 trg_feats=1,
                 unet_half_res=False,
                 weighted=False,
                 do_inv=True,
                 hackmask=False,
                 input_model=None):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer.
                If None (default), the unet features are defined by the default config described in
                the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_unet_features is an integer.
                Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_unet_features is an
                integer. Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when
                this value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration.
                The flow field is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
            input_model: Model to replace default input layer before concatenation. Default is None.
        """

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        if input_model is None:
            # configure default input layers if an input model is not provided
            source = tf.keras.Input(shape=(*inshape, src_feats), name='source_input')
            target = tf.keras.Input(shape=(*inshape, trg_feats), name='target_input')
            input_model = tf.keras.Model(inputs=[source, target], outputs=[source, target])
        else:
            source, target = input_model.outputs[:2]

        # build core unet model and grab inputs
        print(vxm.networks)
        unet_model = vxm.networks.Unet(
            input_model=input_model,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
        )

        # transform unet output into a flow field
        Conv = getattr(KL, 'Conv%dD' % ndims)
        pre_flow = Conv(ndims,
                        kernel_size=3,
                        padding='same',
                        kernel_initializer=KI.RandomNormal(mean=0.0, stddev=1e-5),
                        name='flow')(unet_model.output)

        if do_inv:
            orig_svf = pre_flow
            orig_pre_flow = vxm.layers.VecInt(int_steps=5)(pre_flow)  # flow
            neg_orig_svf = ne.layers.Negate()(orig_svf)
            pre_flow = vxm.layers.VecInt(int_steps=5)(neg_orig_svf)  # negative flow

        if not weighted:
            # extract dense warp matching affine component of current dense
            # The current utility function decompose_warp would also return the reconstruction
            # itself but it's better to work with the affine so we can invert it.
            dec_aff_fn = lambda x: utils.decompose_warp(
                x, return_recon=False, include_identity=False)
            decompose_warp = lambda phi: tf.map_fn(
                dec_aff_fn, phi, fn_output_signature=(tf.float32, tf.float32))
            flow_affine, flow_dense_diff = KL.Lambda(
                decompose_warp, name='decompose_warp')(pre_flow)
            flow = pre_flow

        else:
            # weights are *only* used for computing affines.
            if not hackmask:
                weights = Conv(1,
                               kernel_size=3,
                               padding='same',
                               kernel_initializer=KI.RandomNormal(mean=0.0, stddev=1e-5),
                               activation='relu',
                               name='weights')(unet_model.output)
            else:
                if do_inv:
                    weights = KL.Lambda(lambda x: tf.cast(x > 0.001, tf.float32),
                                        name='weights_hack')(source)
                else:
                    weights = KL.Lambda(lambda x: tf.cast(x > 0.001, tf.float32),
                                        name='weights_hack')(target)

            dec_aff_fn = lambda x: utils.decompose_warp(
                x[0], return_recon=True, include_identity=False, weights=x[1])
            decomposed_affine = lambda x: tf.map_fn(
                dec_aff_fn, x, fn_output_signature=(tf.float32, tf.float32, tf.float32))
            flow_affine, flow_dense_diff, flow = KL.Lambda(
                decomposed_affine, name='decompose_warp')([pre_flow, weights])

        if do_inv:
            # computed affine is actually the inverse because it is computed
            # from the inverted flow field
            inv_affine = flow_affine
            flow_affine = vxm.layers.InvertAffine(name='fix_affine')(inv_affine)
            flow_dense_diff = vxm.layers.ComposeTransform()([inv_affine, orig_pre_flow])

            flow = orig_pre_flow
            if weighted:
                flow = vxm.layers.ComposeTransform()([flow_affine, flow_dense_diff])

        # warp image with flow field
        st_layer = vxm.layers.SpatialTransformer(
            interp_method='linear', indexing='ij', name='transformer', fill_value=0)
        y_source_warped_full = st_layer([source, flow])
        outputs = [y_source_warped_full, flow_dense_diff]

        # if bidirection is required
        if bidir:
            assert 0, \
                'bidir not currently implemented since flow is not SVF.''Should be easy to achieve'
            inv_affine = vxm.layers.InvertAffine(name='invert_affine')(flow_affine)
            y_target = vxm.layers.SpatialTransformer(
                name='neg_transformer', fill_value=0)([target, inv_affine])
            outputs = [y_source_warped_full, y_target, flow_dense_diff]

        # initialize the keras model
        super().__init__(name='vxm_decompose', inputs=input_model.inputs, outputs=outputs)

        # cache pointers to layers and tensors for future reference
        self.references = ne.tf.modelio.LoadableModel.ReferenceContainer()
        self.references.unet_model = unet_model
        self.references.y_source = y_source_warped_full
        self.references.y_target = y_target if bidir else None
        self.references.pos_flow_affine = flow_affine
        self.references.pos_flow = flow
        self.references.neg_flow = inv_affine if bidir else None

    def get_registration_model(self, affine_only=False):
        """
        Returns a reconfigured model to predict only the final transform.
        """

        if affine_only:
            flow = self.references.pos_flow_affine
        else:
            flow = self.references.pos_flow

        st_layer = vxm.layers.SpatialTransformer(
            interp_method='linear', indexing='ij', name='reg_trf', fill_value=0)

        return tf.keras.Model(self.inputs, st_layer([self.inputs[0], flow]))

    def register(self, src, trg, affine_only=False):
        """
        Predicts the transform from src to trg tensors.
        """

        return self.get_registration_model(affine_only=affine_only).predict([src, trg])

    def apply_transform(self, src, trg, img, interp_method='linear'):
        """
        Predicts the transform from src to trg and applies it to the img tensor.
        """
        warp_model = self.get_registration_model()
        img_input = tf.keras.Input(shape=img.shape[1:])
        y_img = vxm.layers.SpatialTransformer(interp_method=interp_method)([
            img_input, warp_model.output])
        return tf.keras.Model(warp_model.inputs + [img_input], y_img).predict([src, trg, img])


class VxmAffineDenseRegSegAtlas(ne.tf.modelio.LoadableModel):
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
    @ ne.modelio.store_config_args
    def __init__(self,
                 inshape,
                 transform_type='affine',
                 seg_channels=2,
                 im_channels=1,
                 model_input=None,
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
                 vxm_dense=None,
                 **kwargs):

        assert bidir, 'bidir = false not supported yet'

        if model_input is None:
            vxm_class = VxmAffineEncoderThenDenseComboDensePatches
            model_input = vxm_class(inshape,
                                    enc_nf_affine=enc_nf_affine,
                                    nfeats=seg_channels + im_channels,
                                    bidir=bidir,
                                    transform_type=transform_type,
                                    rescale_affine=rescale_affine,
                                    name='affine_model',
                                    blurs=affine_blurs)

        if unet is None:
            unet = ne.models.unet(
                unet_features,
                tuple(inshape) + (im_channels,),
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
        atlas_both = KL.Concatenate(
            axis=-1, name='atlas_stack')([atlas_im_input, atlas_seg_input])
        model_outputs = model_input([moving_both, atlas_both])

        moving = KL.Lambda(lambda x: x[..., 0], name='moving')(model_outputs[0])
        moving_seg_warped = KL.Lambda(lambda x: x[..., 1:], name='moving_seg')(model_outputs[0])
        atlas = KL.Lambda(lambda x: x[..., 0], name='atlas')(model_outputs[1])
        atlas_seg_warped = KL.Lambda(
            lambda x: x[..., 1:], name='atlas_seg_warped')(model_outputs[1])
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
        # outputs = [moving, atlas, moving_seg_warped, atlas_seg_warped, warp]
        outputs = [moving, atlas, moving_seg_warped, model_outputs[-1]]
        inputs = [moving_input, atlas_im_input, atlas_seg_input]
        super().__init__(name=name, inputs=inputs, outputs=outputs)

        # cache pointers to layers and tensors for future reference
        self.references = ne.tf.modelio.LoadableModel.ReferenceContainer()
        self.references.affine = affine
        self.references.unet_model = unet
        self.references.moving_seg = moving_seg
        self.references.moving_seg_warped = moving_seg_warped
        self.references.affine_model = model_input
        self.references.affine = model_input.references.affine
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


class VxmAffineThenDenseRegSegWithSeparateUnets(ne.tf.modelio.LoadableModel):
    """
    VoxelMorph network to perform combined affine registration and segmentation.
    inputs  are 2 images
    outputs = [moving, fixed, moving_seg_warped, fixed_seg_warped, both_segs]
       moving image xformed to fixed space
       fixed image xformed to moving space
       moving seg xformed to fixed space
       fixed seg xformed to moving space
       stack of xformed moving seg and (non xformed) fixed seg for overlap calc

    usual unet/vxm parameters (see e.g. VxmAffine or ne.models.unet)
    in addition seg_dilations specifies a number of dilations to apply
    to the outputs segmentations that make up the stacked output only,
    NOT to the individual segmentation outputs

    Author: brf2
    """
    @ ne.modelio.store_config_args
    def __init__(self,
                 inshape,
                 transform_type='affine',
                 seg_channels=2,
                 im_channels=1,
                 model_affine=None,
                 unet=None,
                 enc_nf_affine=None,
                 affine_blurs=[1],
                 bidir=False,
                 rescale_affine=.1,
                 name='vxm_regseg_affine',
                 unet_features=None,
                 unet_levels=4,
                 unet_feat_mult=1,
                 unet_conv_per_level=2,
                 seg_dilations=0,
                 vxm_dense=None,
                 fill_value=0,
                 **kwargs):

        if model_affine is None:
            #            model_affine = vxms.networks.VxmAffine(inshape, enc_nf_affine,
            # nchannels=seg_channels+im_channels, bidir=bidir, transform_type=transform_type,
            # rescale_affine=rescale_affine, name='affine_model', blurs=affine_blurs)
            vxm_model_class = vxms.networks.VxmAffineEncoderThenDensePatches
            model_affine = vxm_model_class(inshape, enc_nf_affine,
                                           nchannels=seg_channels + im_channels,
                                           bidir=bidir,
                                           transform_type=transform_type,
                                           rescale_affine=rescale_affine,
                                           name='affine_model',
                                           fill_value=fill_value,
                                           **kwargs)

        if unet is None:
            unet = ne.models.unet(
                unet_features,
                tuple(inshape) + (im_channels,),
                unet_levels,
                len(inshape) * (5,),
                seg_channels,
                feat_mult=unet_feat_mult,
                nb_conv_per_level=unet_conv_per_level,
                name='regseg_unet'
            )

        # allocate the two inputs - moving,fixed
        moving_input = KL.Input((*inshape, im_channels), name='unet_input_moving')
        fixed_input = KL.Input((*inshape, im_channels), name='unet_input_fixed')

        # create the segmentations by running them each through the unet
        moving_seg = unet(moving_input)
        fixed_seg = unet(fixed_input)

        # reshape the inputs to have an extra channel, then stack them
        # with the segmentations so they can be inputs to the affine
        moving_both = KL.Concatenate(axis=-1, name='moving_stack')([moving_input, moving_seg])
        fixed_both = KL.Concatenate(axis=-1, name='fixed_stack')([fixed_input, fixed_seg])
        affine_outputs = model_affine([moving_both, fixed_both])

        moving = KL.Lambda(lambda x: x[..., 0], name='moving')(affine_outputs[0])
        moving_seg_warped = KL.Lambda(lambda x: x[..., 1:], name='moving_seg')(affine_outputs[0])
        fixed = KL.Lambda(lambda x: x[..., 0], name='fixed')(affine_outputs[1])
        fixed_seg_warped = KL.Lambda(lambda x: x[..., 1:], name='fixed_seg')(affine_outputs[1])
        # create a new dimensions and stack the segs
        # so that a single loss function gets both of them
        new_shape = tuple(fixed_seg.shape.as_list()[1:] + [1])  # exclude batch
        ndims = len(inshape)
        if ndims == 2:
            pool = tf.keras.layers.MaxPool2D
        else:
            pool = tf.keras.layers.MaxPool3D

        # if dilations are reqeusted, do so now (essentially dilating labels)
        # dilate only frames 1: to avoid dilating background and eroding labels
        if seg_dilations >= 1:
            lbd = lambda x: KL.Concatenate(axis=-1)([x[..., 0:1], pool((seg_dilations,) * ndims,
                                                                       padding='same',
                                                                       strides=1)(x[..., 1:])])
            reshape_moving_input = KL.Lambda(lbd, name='moving_seg_pool_%d' %
                                             seg_dilations)(moving_seg_warped)
            lbd = lambda x: KL.Concatenate(axis=-1)([x[..., 0:1], pool((seg_dilations,) * ndims,
                                                                       padding='same',
                                                                       strides=1)(x[..., 1:])])
            reshape_fixed_input = KL.Lambda(lbd, name='fixed_seg_pool_%d' %
                                            seg_dilations)(fixed_seg_warped)
        else:
            reshape_moving_input = moving_seg_warped
            reshape_fixed_input = fixed_seg

        # reshape and stack them so they can go to the same loss function
        reshape_moving = KL.Reshape(new_shape, name='moving_reshaped')(reshape_moving_input)
        reshape_fixed = KL.Reshape(new_shape, name='fixed_reshaped')(reshape_fixed_input)
        both_segs = KL.Concatenate(axis=-1, name='seg_stack')([reshape_moving, reshape_fixed])

        outputs = [moving, fixed, moving_seg_warped, fixed_seg_warped, both_segs]
        inputs = [moving_input, fixed_input]
        super().__init__(name=name, inputs=inputs, outputs=outputs)
        # cache pointers to layers and tensors for future reference
        self.references = ne.tf.modelio.LoadableModel.ReferenceContainer()
        self.references.unet_model = unet
        self.references.moving_seg = moving_seg
        self.references.affine_model = model_affine
        self.references.affine = model_affine.references.affine
        self.references.scale_affines = model_affine.references.scale_affines
        self.references.transform_type = transform_type

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

    def get_moving_seg(self, inb):
        """
        Predicts the transform from src to trg and applies it to the img tensor.
        """
        outputs = [self.references.moving_seg]
        return tf.keras.models.Model(inputs=self.inputs, outputs=outputs).predict(inb)

    def apply_inv_transform(self, src, trg, img, interp_method='linear'):
        """
        Predicts the transform from src to trg and applies it to the img tensor.
        """
        return self.references.affine_model.apply_inv_transform(src, trg, img, interp_method)


class VxmAffineEncoderThenDenseComboDenseAtlas(ne.tf.modelio.LoadableModel):
    """
    VoxelMorph network to perform combined affine and nonlinear registration.

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
                 nfeats=1,
                 fill_value=0,
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
            # append variance so nfeats+1
            affine_model = vxms.networks.VxmAffineEncoderThenDense(inshape,
                                                                   enc_nf_affine,
                                                                   transform_type=transform_type,
                                                                   bidir=affine_bidir,
                                                                   blurs=affine_blurs,
                                                                   rescale_affine=rescale_affine,
                                                                   nchannels=nfeats + 1)

        # build model inputs - image, atlas mean and atlas var
        image_input = KL.Input(inshape, name='image_input')
        atlas_mean_input = KL.Input(inshape, name='atlas_mean_input')
        atlas_var_input = KL.Input(inshape, name='atlas_var_input')

        # form two stacks - one for the image and one for the atlas mean/var
        new_shape = image_input.shape.as_list()[1:] + [1]
        image_input_re = KL.Reshape(new_shape, name='image_input_reshaped')(image_input)
        atlas_mean_re = KL.Reshape(new_shape, name='atlas_mean_reshaped')(atlas_mean_input)
        atlas_var_re = KL.Reshape(new_shape, name='atlas_var_reshaped')(atlas_var_input)
        # just duplicate it so it is the same shape
        image_stack = KL.Concatenate(
            axis=-1, name='source_concat')([image_input_re, image_input_re])
        atlas_stack = KL.Concatenate(axis=-1, name='atlas_concat')([atlas_mean_re, atlas_var_re])

        # run the image and the atlas through the affine model
        affine_outputs = affine_model([image_stack, atlas_stack])
        source = affine_model.inputs[0]
        atlas = affine_model.inputs[1]
        affine = affine_model.references.affine

        # build a dense model that takes the affine transformed src as input
        dense_input_model = tf.keras.Model(affine_model.inputs, (affine_outputs[0], atlas_stack))
        #        dense_model = vxm.networks.VxmDense(inshape, nb_unet_features=nb_unet_features,
        #                                            bidir=bidir, input_model=dense_input_model,
        #                                            src_feats=nfeats+1, trg_feats=nfeats+1,
        #                                            **kwargs)
        dense_outputs = dense_model([affine_model.outputs[0], atlas_stack])
        flow_params = dense_outputs[-1]
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
            y_atlas = vxm.layers.SpatialTransformer(fill_value=fill_value)([atlas, inv_composed])
            outputs = [y_source, y_atlas, flow_params]
            outputs = [y_source, y_atlas, composed]
        else:
            outputs = [y_source, composed]

        # initialize the keras model
        super().__init__(inputs=affine_model.inputs, outputs=outputs)

        # cache pointers to layers and tensors for future reference
        self.references = ne.tf.modelio.LoadableModel.ReferenceContainer()
        self.references.affine = affine
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
        img_input = tf.keras.Input(shape=img[1:])
        y_img = vxm.layers.SpatialTransformer(interp_method=interp_method)([
            img_input, warp_model.output])
        return tf.keras.Model(warp_model.inputs + [img_input], y_img).predict([src, trg, img])


class VxmAffineEncoderThenDenseComboDense(ne.tf.modelio.LoadableModel):
    """
    VoxelMorph network to perform combined affine and nonlinear registration.

    Author: brf2
    """

    @ne.modelio.store_config_args
    def __init__(self,
                 inshape,
                 nb_unet_features=None,
                 enc_nf_affine=None,
                 transform_type='affine',
                 affine_bidir=True,
                 affine_blurs=[1],
                 input_model=None,
                 bidir=True,
                 affine_model=None,
                 rescale_affine=[1.0, 0.01],
                 reserve_encoders=None,
                 nb_unet_conv_per_level=1,
                 fill_value=None,
                 nfeats=1,
                 Conv=None,
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
            nb_unet_features = default_unet_features()

        # use dense-net encoder features if affine-net features aren't provided
        if enc_nf_affine is None:
            if isinstance(nb_unet_features, int):
                raise ValueError(
                    'enc_nf_affine list must be provided when nb_unet_features is an integer')
            enc_nf_affine = nb_unet_features[0]

        # affine component
        if affine_model is None:
            affine_model = VxmAffineEncoderThenDense(
                inshape, enc_nf_affine,
                transform_type=transform_type,
                input_model=input_model,
                bidir=affine_bidir,
                blurs=affine_blurs,
                rescale_translations=1.0,
                rescale_non_translations=0.01,
                reserve_encoders=reserve_encoders,
                nchannels=nfeats,
                Conv=Conv,
                fill_value=fill_value)

        if input_model:
            source = input_model.outputs[0]
            target = input_model.outputs[1]
        else:
            source = affine_model.inputs[0]
            target = affine_model.inputs[1]

        affine = affine_model.references.affine
        inv_affine = affine_model.references.inv_affine
        half_src = vxms.layers.MidspaceTransform(name='half_src')(affine)
        half_trg = vxms.layers.MidspaceTransform(name='half_trg')(inv_affine)
        half_src_inv = vxm.layers.InvertAffine(name='half_src_inv')(half_src)
        half_trg_inv = vxm.layers.InvertAffine(name='half_trg_inv')(half_trg)
        source_half = vxm.layers.SpatialTransformer(fill_value=fill_value, name='source_half')(
            [source, half_src])
        target_half = vxm.layers.SpatialTransformer(fill_value=fill_value, name='target_half')(
            [target, half_trg])

        # build a dense model that takes the affine transformed src as input
        dense_input_model = tf.keras.Model(affine_model.inputs, [source_half, target_half])
        dense_model = vxm.networks.VxmDense(inshape,
                                            nb_unet_features=nb_unet_features,
                                            bidir=bidir,
                                            input_model=dense_input_model,
                                            src_feats=nfeats,
                                            trg_feats=nfeats,
                                            fill_value=fill_value,
                                            nb_unet_conv_per_level=nb_unet_conv_per_level,
                                            **kwargs)

        # build a single transform that applies both affine and dense to src
        # and apply it to the input (src) volume so that there is only 1 interpolation
        # and output it as the combined model output (plus the dense warp)
        # compute transform that takes source to target
        flow_params = dense_model.outputs[-1]
        pos_flow = dense_model.references.pos_flow
        src_composed = vxm.layers.ComposeTransform()([half_src, pos_flow, half_trg_inv])
        y_source = vxm.layers.SpatialTransformer(fill_value=fill_value)([source, src_composed])

        # invert and transform for bidirectional training
        if bidir:
            neg_flow = dense_model.references.neg_flow
            inv_affine = vxm.layers.InvertAffine()(affine)
            trg_composed = vxm.layers.ComposeTransform()([half_trg, neg_flow, half_src_inv])
            y_target = vxm.layers.SpatialTransformer(fill_value=fill_value)([target, trg_composed])
            outputs = [y_source, y_target, src_composed]
        else:
            outputs = [y_source, src_composed]

        # initialize the keras model
        super().__init__(inputs=affine_model.inputs, outputs=outputs)

        # cache pointers to layers and tensors for future reference
        self.references = ne.tf.modelio.LoadableModel.ReferenceContainer()
        self.references.input_model = input_model
        self.references.affine = affine
        self.references.affine_model = affine_model
        self.references.dense_model = dense_model
        self.references.pos_flow = pos_flow
        self.references.composed = src_composed
        self.references.neg_flow = neg_flow if bidir else None
        self.references.inv_composed = trg_composed if bidir else None
        self.references.inv_affine = inv_affine if bidir else None

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
        y_img = vxm.layers.SpatialTransformer(interp_method=interp_method, fill_value=fill_value)([
            img_input, warp_model.output])
        return tf.keras.Model(warp_model.inputs + [img_input], y_img).predict([src, trg, img])

    def apply_inv_transform(self, src, trg, img, interp_method='linear', fill_value=None):
        """
        Predicts the transform from src to trg and applies it to the img tensor.
        """
        warp_model = self.get_inv_registration_model()
        img_input = tf.keras.Input(shape=img.shape[1:])
        y_img = vxm.layers.SpatialTransformer(interp_method=interp_method, fill_value=fill_value)(
            [img_input, warp_model.output])
        return tf.keras.Model(warp_model.inputs + [img_input], y_img).predict([src, trg, img])


class VxmAffineEncoderThenDense(AbstractVxmModel):
    """
    Voxelorph network for linear (affine) registration between two images.
    Authors: brf2
    """

    @ne.modelio.store_config_args
    def __init__(self,
                 inshape,
                 enc_nf,
                 bidir=False,
                 transform_type='affine',
                 predict_matrix_params=False,
                 rescale_translations=.01,
                 rescale_non_translations=1,
                 blurs=[1],
                 nchannels=1,
                 name='vxm_affine',
                 fill_value=None,
                 reserve_capacity=None,
                 reserve_encoders=None,
                 Conv=None,
                 store_midspace=True,
                 trans_enc_first=None,
                 input_model=None):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            enc_nf: List of encoder filters. e.g. [16, 32, 32, 32]
            bidir: Enable bidirectional cost function. Default is False.
            transform_type: 'affine', 'rigid', or 'shearless'.
            predict_matrix_params: Directly predict matrix, not affine parameters.
            rescale_translations: Rescale predicted translation parameters.
            rescale_non_translations: Rescale predicted non-translation parameters.
            blurs: List of gaussian blur kernel levels for inputs. Default is [1].
            nchannels: Number of input channels. Default is 1.
            name: model name. Default is 'vxm_affine'.
            trans_enc_first: have first encoder only output translations.
                             If a list, use it for features
        """

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        if Conv is None:
            Conv = getattr(KL, 'Conv%dD' % ndims)

        # configure base encoder CNN
        cnn_layers = []
        if type(trans_enc_first) is list:   # use a different architecture for first encoder
            first_nf = trans_enc_first
            trans_enc_first = True
        else:
            if trans_enc_first is None:
                trans_enc_first = False
            first_nf = enc_nf   # same as the other encoders

        for nf in first_nf:
            if isinstance(nf, list):
                for nf1 in nf[0:-1]:
                    cnn_layers.append(Conv(nf1, kernel_size=3, padding='same',
                                           kernel_initializer='he_normal', strides=1))
                    cnn_layers.append(KL.LeakyReLU(0.2))
                nf = nf[-1]  # last layer at this level is added below with stride=2

            cnn_layers.append(Conv(nf, kernel_size=3, padding='same',
                                   kernel_initializer='he_normal', strides=2))
            cnn_layers.append(KL.LeakyReLU(0.2))

        # flatten features
        cnn_layers.append(KL.Flatten())
        if reserve_capacity is not None:
            cnn_layers.append(vxms.layers.ask(reserve_capacity, name='reserve'))

        # predict affine matrix
        cnn_layers.append(vxms.layers.TensorToAffineMatrix(
            ndims=ndims,
            transform_type='trans' if trans_enc_first else transform_type,
            predict_matrix_params=predict_matrix_params,
            rescale_translations=1 if trans_enc_first else rescale_translations,
            rescale_non_translations=1 if trans_enc_first else rescale_non_translations,
            name='affine_matrix'))

        # general caller for base CNN
        def cnn(x):
            for layer in cnn_layers:
                x = layer(x)
            return x

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
            blur_concat = KL.concatenate([source_blur, target_blur], name=prefix + 'concat')
            curr_affine = cnn(blur_concat)
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

        # for each item in reserve_encoders create a new encoder with a different
        # affine scale
        if reserve_encoders is not None:
            # hack so that full_affine has a consistent and findable name
            full_affine = KL.Reshape(full_affine.get_shape().as_list()[1:],
                                     name=f'full_affine_preres')(full_affine)
            full_affines_fwd = [full_affine]
            full_affines_inv = [vxm.layers.InvertAffine()(full_affine)]
            image1_midspace = []
            image2_midspace = []
            if type(reserve_encoders) is not list:
                reserver_encoders = [reserve_encoders]

            res_enc_start_tensors = []
            for resno, encoder_scale in enumerate(reserve_encoders):

                print(f'attaching reserve encoder with wt {encoder_scale}')

                half_fwd = vxms.layers.MidspaceTransform(name=f'half_fwd_{resno+1}')(full_affine)
                res_enc_start_tensors.append(half_fwd)
                # check_batch_is_invertible(full_affine, ndims)
                # full_affine = KL.Lambda(lambda x: check_batch_is_invertible(x,ndims))(full_affine)
                inv_affine = vxm.layers.InvertAffine()(full_affine)
                half_inv = vxms.layers.MidspaceTransform()(inv_affine)
                current_source = vxm.layers.SpatialTransformer(fill_value=fill_value)(
                    [source, half_fwd])
                current_target = vxm.layers.SpatialTransformer(fill_value=fill_value)(
                    [target, half_inv])
                image1_midspace.append(current_source)
                image2_midspace.append(current_target)
                prev_tensor = KL.concatenate([current_source, current_target])
                for nf in enc_nf:
                    if isinstance(nf, list):
                        conv_out = prev_tensor
                        for nf1 in nf[0:-1]:
                            conv_out = Conv(nf1, kernel_size=3, padding='same',
                                            kernel_initializer='he_normal', strides=1)(conv_out)
                            conv_out = KL.LeakyReLU(0.2)(conv_out)
                        nf = nf[-1]  # last layer at this level is added below with stride=2
                    else:
                        conv_out = prev_tensor

                    conv_out = Conv(nf, kernel_size=3, padding='same',
                                    kernel_initializer='he_normal', strides=2)(conv_out)
                    prev_tensor = KL.LeakyReLU(0.2)(conv_out)

                # predict affine
                affine = vxms.layers.TensorToAffineMatrix(
                    ndims=ndims,
                    transform_type=transform_type,
                    predict_matrix_params=predict_matrix_params,
                    rescale_translations=encoder_scale * rescale_translations,
                    rescale_non_translations=encoder_scale * rescale_non_translations,
                    name=f'affine_matrix_res_{resno}')(prev_tensor)

                # compose predicted transform with last
                full_affine = vxm.layers.ComposeTransform(name=f'compose_res{resno}')(
                    [full_affine, affine])

                full_affines_fwd.append(full_affine)
                full_affines_inv.append(vxm.layers.InvertAffine()(full_affine))

            y_source = vxm.layers.SpatialTransformer(
                name=prefix + f'transformer_res{resno}', fill_value=fill_value)(
                    [source, full_affine])

        if reserve_capacity is not None:
            cnn_layers.append(vxms.layers.ask(reserve_capacity, name='reserve'))

        # hack so that full_affine has a consistent and findable name
        full_affine = KL.Lambda(lambda x: x, name=f'full_affine_{name}')(full_affine)

        # invert affine for bidirectional training
        if bidir:
            inv_affine = vxm.layers.InvertAffine(name=f'invert_affine_{name}')(full_affine)
            y_target = vxm.layers.SpatialTransformer(
                name=f'neg_transformer_{name}', fill_value=fill_value)([target, inv_affine])
            outputs = [y_source, y_target]
        else:
            outputs = [y_source]

        # initialize the keras model
        super().__init__(name=name, inputs=input_model.inputs, outputs=outputs)
        # super().__init__(name=name, inputs=[source, target], outputs=outputs)

        # cache affines
        self.references = ne.tf.modelio.LoadableModel.ReferenceContainer()
        if reserve_encoders is not None:
            self.references.affines_fwd = full_affines_fwd
            self.references.affines_inv = full_affines_inv
            self.references.res_enc_start_tensors = res_enc_start_tensors

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

        if reserve_encoders is not None:
            self.references.image1_midspace = image1_midspace
            self.references.image2_midspace = image2_midspace

        # compute midspace images for user if needed
        if store_midspace:
            half_fwd = vxms.layers.MidspaceTransform()(full_affine)
            inv_affine = vxm.layers.InvertAffine()(full_affine)
            half_inv = vxms.layers.MidspaceTransform()(inv_affine)
            source_half = vxm.layers.SpatialTransformer(fill_value=fill_value)([source, half_fwd])
            target_half = vxm.layers.SpatialTransformer(fill_value=fill_value)([target, half_inv])
            self.references.half_affine_fwd = half_fwd
            self.references.half_affine_inv = half_inv
            self.references.source_half = source_half
            self.references.target_half = target_half


class VxmAffineEncoderThenDenseNew(VxmAffineEncoderThenDense):
    def __init__(self, **kwargs):
        warnings.warn('use VxmAffineEncoderThenDense without the New')
        super().__init__(**kwargs)


class VxmSynthCombo(ne.tf.modelio.LoadableModel):
    """
    VoxelMorph network for combo linear (affine)/dense registration between two images.
    including nonlinear B0 correction.

    synth_image from label maps. Augment it with 2 random affines. Apply B0 distortion
    that is the same magnitude but different orientations (randomly drawn) to each of the
    two images. Then try to learn the inverse - a single scalar map applied in the two
    directions that should cancel the distortions.

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
            name='VxmSynthAffineWithCorrection',
            max_trans=15,
            reserve_encoders=[1e-3],
            max_rot=10,
            min_B0_blur=1,
            max_B0_blur=20,
            max_B0_std=1,
            affine_conv_per_level=1,
            structure_list={},
            Conv=None,
            subsample_atrophy=1,
            reserve_unets=None,
            debug=False,
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
            seeds=seeds,         # same seeds as gen_model do intensity distributions are same
            **gen_args
        )

        # input a label map and use it to synthesize an image. Remove air
        label_input = KL.Input(shape=tuple(inshape) + (1,), name='label_in')

        # input vectors that give B0 distortion direction
        draw_dir = lambda _: tf.random.shuffle([1.0, 0.0, 0.0])[tf.newaxis, ...]
        B0_dir1 = KL.Lambda(lambda x: tf.map_fn(draw_dir, x), name='B0_dir1')(label_input)
        B0_dir2 = KL.Lambda(lambda x: tf.map_fn(draw_dir, x), name='B0_dir2')(label_input)

        label_input_with_atrophy = vxms.layers.ResizeLabels(
            structure_list, name='resize_labels', subsample_atrophy=subsample_atrophy)(label_input)

        synth_image, synth_labels = gen_model(label_input)
        atrophy_image, atrophy_labels = gen_model_a(label_input_with_atrophy)

        # skull_mask = tf.cast(synth_labels > 0, tf.float32)

        # extract the separate image contrasts (disabled for now)
        synth_image1 = KL.Lambda(lambda x: x[..., 0:1], name='synth_image1')(synth_image)
        synth_image2 = KL.Lambda(lambda x: x[..., 0:1], name='synth_image2')(atrophy_image)  # 1:2
        synth_image2_no_atrophy = KL.Lambda(lambda x: x[..., 0:1],
                                            name='synth_image2_no_atrophy')(synth_image)

        # sample a rigid transform  and apply it to the image
        B0_map = vxms.layers.DrawB0Map(inshape, name='B0_map', max_std=max_B0_std,
                                       min_blur=min_B0_blur, max_blur=max_B0_blur)(B0_dir1)
        if 0:
            im1_transformed, mats1 = vxms.layers.AffineAugment(
                name='im1_transformed', max_trans=max_trans, max_rot=max_rot,
                return_mats=True, limit_to_fov=True)([synth_image1])
            im2_transformed, mats2 = vxms.layers.AffineAugment(
                name='im2_transformed', max_trans=max_trans, max_rot=max_rot,
                return_mats=True, limit_to_fov=True)([synth_image2])
            # store the augmentation transforms for later use and apply to labels
            Id = tf.eye(ndims + 1)[tf.newaxis, 0:3, :]  # have to remove identity
            mats1_noid = KL.Lambda(lambda x: tf.subtract(x, Id), name='aug1_noID')(mats1)
            mats2_noid = KL.Lambda(lambda x: tf.subtract(x, Id), name='aug2_noID')(mats2)
            im2_transformed_no_atrophy = vxm.layers.SpatialTransformer(
                name='im2_transformed_no_atrophy', interp_method='linear',
                fill_value=fill_value)([synth_image2_no_atrophy, mats2_noid])

            # sample B0 map (scalar) then transform it into 2 image spaces to create warps
            B0_map1 = vxm.layers.SpatialTransformer(
                name='B0_map1', interp_method='linear', fill_value=fill_value)([B0_map, mats1_noid])
            B0_map2 = vxm.layers.SpatialTransformer(
                name='B0_map2', interp_method='linear', fill_value=fill_value)([B0_map, mats2_noid])
        else:   # disable affine aug
            im1_transformed = synth_image1
            im2_transformed = synth_image2
            im2_transformed_no_atrophy = synth_image2_no_atrophy
            B0_map1 = B0_map
            B0_map2 = B0_map

        # compute B0 warp for each image (same B0 inhomogeneity, different direction)
        B0_warp1 = KL.Dot(axes=[-1, 1], name='B0_warp1')([B0_map1, B0_dir1])
        B0_warp2 = KL.Dot(axes=[-1, 1], name='B0_warp2')([B0_map2, B0_dir2])

        # composite warps that transform from synth images to B0 warped images
        if 0:
            warp1_aug_B0 = vxm.layers.ComposeTransform(name='warp1_aug_B0')([mats1_noid, B0_warp1])
            warp2_aug_B0 = vxm.layers.ComposeTransform(name='warp2_aug_B0')([mats2_noid, B0_warp2])
        else:
            warp1_aug_B0 = B0_warp1
            warp2_aug_B0 = B0_warp2

        # apply the distortions to the images
        im1_distorted = vxm.layers.SpatialTransformer(
            name='im1_distorted', interp_method='linear',
            fill_value=fill_value)([synth_image1, warp1_aug_B0])
        im2_distorted = vxm.layers.SpatialTransformer(
            name='im2_distorted', interp_method='linear',
            fill_value=fill_value)([synth_image2, warp2_aug_B0])

        # compute an affine that aligns the two distorted images (loss is based on
        # unseen at test time undistorted images)
        combo_inputs = [im1_distorted, im2_distorted]
        input_model = tf.keras.models.Model(inputs=label_input, outputs=combo_inputs)

        vxm_affine = VxmAffineEncoderThenDense(
            inshape,
            enc_nf=affine_nf,
            input_model=input_model,
            fill_value=fill_value,
            reserve_encoders=reserve_encoders,
            Conv=Conv,
            store_midspace=True,
            **kwargs)

        # compose the augmentation/B0 warp with the vxm-computed affine warp transform
        # this is really just the affine alignment of the distorted volumes, but avoids
        # the multiple resamplings (synth->aug->B0 dist->target im)
        if 0:
            affine1 = vxm_affine.references.affine
            affine2 = vxm_affine.references.inv_affine
        else:    # disable the affine
            inv_aug1 = vxm.layers.InvertAffine(name='inv_aug1')(mats1_noid)
            inv_aug2 = vxm.layers.InvertAffine(name='inv_aug2')(mats2_noid)
            affine1 = vxm.layers.ComposeTransform(name='fake_aff1')([inv_aug1, mats2_noid])
            affine2 = vxm.layers.ComposeTransform(name='fake_aff2')([inv_aug2, mats1_noid])

        half1 = vxms.layers.MidspaceTransform(name='half1')(affine1)
        half2 = vxms.layers.MidspaceTransform(name='half2')(affine2)

        warp1_synth_to_im2_aff = vxm.layers.ComposeTransform(name='comp1')([warp1_aug_B0, affine1])
        warp2_synth_to_im1_aff = vxm.layers.ComposeTransform(name='comp2')([warp2_aug_B0, affine2])
        warp1_synth_to_midspace = vxm.layers.ComposeTransform(name='warp1_synth_to_mid')(
            [warp1_aug_B0, half1])
        warp2_synth_to_midspace = vxm.layers.ComposeTransform(name='warp2_synth_to_mid')(
            [warp2_aug_B0, half2])

        # map images to midspace
        im1_midspace = vxm.layers.SpatialTransformer(
            name='im1_midspace', interp_method='linear',
            fill_value=fill_value)([synth_image1, warp1_synth_to_midspace])
        im2_midspace = vxm.layers.SpatialTransformer(
            name='im2_midspace', interp_method='linear',
            fill_value=fill_value)([synth_image2, warp2_synth_to_midspace])
        im2_midspace_no_atrophy = vxm.layers.SpatialTransformer(
            name='im2_midspace_no_atrophy', interp_method='linear',
            fill_value=fill_value)([synth_image2_no_atrophy, warp2_synth_to_midspace])

        inshape_vox = np.array(inshape).prod()
        B0_dir1_flat = KL.Flatten()(B0_dir1)
        B0_dir2_flat = KL.Flatten()(B0_dir2)
        B0_dir1_dense = KL.Dense(inshape_vox, name='B0_dir1_dense')(B0_dir1_flat)
        B0_dir2_dense = KL.Dense(inshape_vox, name='B0_dir2_dense')(B0_dir2_flat)
        B0_dir1_im = KL.Reshape(tuple(inshape) + (1,), name='B0_dir1_im')(B0_dir1_dense)
        B0_dir2_im = KL.Reshape(tuple(inshape) + (1,), name='B0_dir2_im')(B0_dir2_dense)
        input_list = [im1_midspace, im2_midspace, B0_dir1_im, B0_dir2_im]
        unet_inputs = KL.Concatenate(name='unet_inputs', axis=-1)(input_list)
        unet_model = ne.models.unet(
            unet_features,
            tuple(inshape) + (len(input_list),),
            unet_levels,
            3,              # conv size
            1,              # nb_labels/nb_outputs
            nb_conv_per_level=nb_unet_conv_per_level,
            name='B0_correction_unet',
            final_pred_activation='linear',
        )

        # output of unet should be magnitude of (inverse) distortions. Apply it in each dir
        unet_outputs = unet_model(unet_inputs)
        B0_unmap = KL.Lambda(lambda x: x[..., 0:1], name='B0_unmap')(unet_outputs)
        B0_unwarp1 = KL.Dot(axes=[-1, 1], name='B0_unwarp1')([B0_unmap, B0_dir1])
        B0_unwarp2 = KL.Dot(axes=[-1, 1], name='B0_unwarp2')([B0_unmap, B0_dir2])

        # the unet output should cancel the B0 warp, so the following 2 should be zero warps
        B0_cancelled1 = vxm.layers.ComposeTransform(name='B0_cancelled1')([B0_warp1, B0_unwarp1])
        B0_cancelled2 = vxm.layers.ComposeTransform(name='B0_cancelled2')([B0_warp2, B0_unwarp2])

        # apply the composed warp should ideally get you back to the augmented (undistorted) images
        # adding the mats?_noid gets you back to the original synth images
        warp1_B0_correct = vxm.layers.ComposeTransform(name='warp1_B0_correct')(
            [mats1_noid, B0_cancelled1])
        warp2_B0_correct = vxm.layers.ComposeTransform(name='warp2_B0_correct')(
            [mats2_noid, B0_cancelled2])
        im1_corrected = vxm.layers.SpatialTransformer(
            name='im1_corrected', interp_method='linear',
            fill_value=fill_value)([synth_image1, warp1_B0_correct])
        im2_corrected = vxm.layers.SpatialTransformer(
            name='im2_corrected', interp_method='linear',
            fill_value=fill_value)([synth_image2_no_atrophy, warp2_B0_correct])

        # now generate midspace corrected images
        warp1_midspace_corrected = vxm.layers.ComposeTransform(name='warp1_midspace_corrected')(
            [mats1_noid, B0_cancelled1, half1])
        warp2_midspace_corrected = vxm.layers.ComposeTransform(name='warp2_midspace_corrected')(
            [mats2_noid, B0_cancelled2, half2])
        im1_midspace_corrected = vxm.layers.SpatialTransformer(
            name='im1_midspace_corrected', interp_method='linear',
            fill_value=fill_value)([synth_image1, warp1_midspace_corrected])
        im2_midspace_corrected = vxm.layers.SpatialTransformer(
            name='im2_midspace_corrected', interp_method='linear',
            fill_value=fill_value)([synth_image2_no_atrophy, warp2_midspace_corrected])
        midspace_outputs = KL.Concatenate(axis=-1, name='midspace_outputs')(
            [im1_midspace_corrected, im2_midspace_corrected])

        # apply the composed half and full transforms to images and labels
        # use the synth2 image without atrophy so that we learn a transform that
        # is invariant to atrophy
        # appling aug, B0, inv B0 should get back to original (rigidly augmented) image
        # return labels in fixed/moving space in case user wants them
        outputs_nl = [midspace_outputs, B0_unwarp1, B0_unwarp2]
        if debug:  # make sure some intermediate stuff is in the model
            outputs_nl += [im1_distorted, im2_distorted, im1_corrected,
                           im2_corrected, im2_midspace_no_atrophy]

        # make some things available to the caller for use in loss functions
        # including the undistorted affine-transformed volumes and the nonlinear
        # component of the combined warp

        self.references = ne.tf.modelio.LoadableModel.ReferenceContainer()
        self.references.vxm_affine = vxm_affine
        self.references.unet_model = unet_model

        super().__init__(name=name, inputs=label_input, outputs=outputs_nl)


class VxmSynthComboTest(ne.tf.modelio.LoadableModel):
    """
    VoxelMorph network for combo linear (affine)/dense registration between two images.
    including nonlinear B0 correction.

    synth_image from label maps. Augment it with 2 random affines. Apply B0 distortion
    that is the same magnitude but different orientations (randomly drawn) to each of the
    two images. Then try to learn the inverse - a single scalar map applied in the two
    directions that should cancel the distortions.

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
            name='VxmSynthAffineWithCorrection',
            max_trans=15,
            reserve_encoders=[1e-3],
            max_rot=10,
            min_B0_blur=1,
            max_B0_blur=20,
            max_B0_std=1,
            affine_conv_per_level=1,
            structure_list={},
            Conv=None,
            subsample_atrophy=1,
            reserve_unets=None,
            debug=False,
            unet_input_model=None,
            one_d=False,
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
            seeds=seeds,         # same seeds as gen_model do intensity distributions are same
            **gen_args
        )

        # input a label map and use it to synthesize an image. Remove air
        label_input = KL.Input(shape=tuple(inshape) + (1,), name='label_in')

        # synthesize vectors that give B0 distortion direction
        vec = [1.0, 0.0, 0.0][0:ndims]
        draw_dir = lambda _: tf.random.shuffle(vec)[tf.newaxis, ...]
        sign_flip = lambda x: x * tf.random.shuffle([1.0, -1.0])[0]
        B0_dir1 = KL.Lambda(lambda x: tf.map_fn(draw_dir, x), name='B0_dir1d')(label_input)
        B0_dir2 = KL.Lambda(lambda x: tf.map_fn(draw_dir, x), name='B0_dir2d')(label_input)
        B0_dir1 = KL.Lambda(lambda x: tf.map_fn(sign_flip, x), name='B0_dir1')(B0_dir1)
        B0_dir2 = KL.Lambda(lambda x: tf.map_fn(sign_flip, x), name='B0_dir2')(B0_dir2)

        # synthesize images with atrophy in them
        label_input_with_atrophy = vxms.layers.ResizeLabels(
            structure_list, name='resize_labels', subsample_atrophy=subsample_atrophy)(label_input)

        synth_image, synth_labels = gen_model(label_input)
        atrophy_image, atrophy_labels = gen_model_a(label_input_with_atrophy)
        sli = tf.cast(synth_labels, tf.float32)
        ali = tf.cast(atrophy_labels, tf.float32)

        # extract the separate image contrasts (disabled for now)
        synth_image1 = KL.Lambda(lambda x: x[..., 0:1], name='synth_image1')(synth_image)
        synth_image2 = KL.Lambda(lambda x: x[..., 0:1], name='synth_image2')(atrophy_image)  # 1:2
        synth_image2_no_atrophy = KL.Lambda(lambda x: x[..., 0:1],
                                            name='synth_image2_no_atrophy')(synth_image)

        # sample a rigid transform  and apply it to the image (also disabled)
        B0_map = vxms.layers.DrawB0Map(inshape, name='B0_map_unmasked', max_std=max_B0_std,
                                       min_blur=min_B0_blur, max_blur=max_B0_blur)(B0_dir1)
        skull_mask = tf.cast(synth_labels > 0, tf.float32)
        B0_map = KL.Multiply(name='B0_map')([B0_map, skull_mask])  # no B0 effects in air

        im1_transformed = synth_image1
        im2_transformed = synth_image2
        im2_transformed_no_atrophy = synth_image2_no_atrophy

        B0_map1 = B0_map
        B0_map2 = B0_map   # same B0 maps for now

        # compute B0 warp for each image (same B0 inhomogeneity, different direction)
        B0_warp1_svf = KL.Dot(axes=[-1, 1], name='B0_warp1_svf')([B0_map1, B0_dir1])
        B0_warp2_svf = KL.Dot(axes=[-1, 1], name='B0_warp2_svf')([B0_map2, B0_dir2])
        B0_warp1 = vxm.layers.VecInt(int_steps=5, name='B0_warp1')(B0_warp1_svf)
        B0_warp2 = vxm.layers.VecInt(int_steps=5, name='B0_warp2')(B0_warp2_svf)
        B0_warp1_neg_svf = ne.layers.Negate()(B0_warp1_svf)
        B0_warp2_neg_svf = ne.layers.Negate()(B0_warp2_svf)
        B0_warp1_inv = vxm.layers.VecInt(int_steps=5, name='B0_warp1_inv')(B0_warp1_neg_svf)
        B0_warp2_inv = vxm.layers.VecInt(int_steps=5, name='B0_warp2_inv')(B0_warp2_neg_svf)

        warp1_aug_B0 = B0_warp1   # disabled rigid augmentation
        warp2_aug_B0 = B0_warp2

        # apply the distortions to the images
        im1_distorted = vxm.layers.SpatialTransformer(
            name='im1_distorted', interp_method='linear',
            fill_value=fill_value)([synth_image1, warp1_aug_B0])
        im2_distorted = vxm.layers.SpatialTransformer(
            name='im2_distorted', interp_method='linear',
            fill_value=fill_value)([synth_image2, warp2_aug_B0])
        im2_distorted_no_atrophy = vxm.layers.SpatialTransformer(
            name='im2_distorted_no_atrophy', interp_method='linear',
            fill_value=fill_value)([synth_image2_no_atrophy, warp2_aug_B0])

        labels1_distorted = vxm.layers.SpatialTransformer(
            name='labels1_distorted', interp_method='nearest',
            fill_value=fill_value)([sli, warp1_aug_B0])
        labels2_distorted = vxm.layers.SpatialTransformer(
            name='labels2_distorted', interp_method='nearest',
            fill_value=fill_value)([ali, warp2_aug_B0])
        labels2_distorted_no_atrophy = vxm.layers.SpatialTransformer(
            name='labels2_distorted_no_atrophy', interp_method='nearest',
            fill_value=fill_value)([sli, warp2_aug_B0])

        warp1_synth_to_im2_aff = warp1_aug_B0   # disabled rigid augmentation
        warp2_synth_to_im1_aff = warp2_aug_B0
        warp1_synth_to_midspace = warp1_aug_B0
        warp2_synth_to_midspace = warp2_aug_B0

        # map images to midspace
        im1_midspace = vxm.layers.SpatialTransformer(
            name='im1_midspace', interp_method='linear',
            fill_value=fill_value)([synth_image1, warp1_synth_to_midspace])
        im2_midspace = vxm.layers.SpatialTransformer(
            name='im2_midspace', interp_method='linear',
            fill_value=fill_value)([synth_image2, warp2_synth_to_midspace])
        im2_midspace_no_atrophy = vxm.layers.SpatialTransformer(
            name='im2_midspace_no_atrophy', interp_method='linear',
            fill_value=fill_value)([synth_image2_no_atrophy, warp2_synth_to_midspace])
        labels1_midspace = vxm.layers.SpatialTransformer(
            name='labels1_midspace', interp_method='nearest',
            fill_value=fill_value)([sli, warp1_synth_to_midspace])
        labels2_midspace = vxm.layers.SpatialTransformer(
            name='labels2_midspace', interp_method='nearest',
            fill_value=fill_value)([ali, warp2_synth_to_midspace])
        labels2_midspace_no_atrophy = vxm.layers.SpatialTransformer(
            name='labels_midspace_no_atrophy', interp_method='nearest',
            fill_value=fill_value)([sli, warp2_synth_to_midspace])

        inshape_vox = np.array(inshape).prod()
        B0_dir1_flat = KL.Flatten()(B0_dir1)
        B0_dir2_flat = KL.Flatten()(B0_dir2)
        B0_dir1_dense = KL.Dense(inshape_vox, name='B0_dir1_dense')(B0_dir1_flat)
        B0_dir2_dense = KL.Dense(inshape_vox, name='B0_dir2_dense')(B0_dir2_flat)
        B0_dir1_im = KL.Reshape(tuple(inshape) + (1,), name='B0_dir1_im')(B0_dir1_dense)
        B0_dir2_im = KL.Reshape(tuple(inshape) + (1,), name='B0_dir2_im')(B0_dir2_dense)

        if 0:     # use VxmDense to estimate B0 inverse warps
            if isinstance(unet_features, list):  # convert to vxm feature format
                nb_unet_conv_per_level = len(unet_features[0])
                enc_nf = []
                for level in range(len(unet_features)):
                    for cno in range(nb_unet_conv_per_level):
                        enc_nf.append(unet_features[level][cno])

                dec_nf = enc_nf.copy()
                dec_nf.reverse()
                dec_nf.append(enc_nf[-1])   # add a couple of convs at end
                dec_nf.append(enc_nf[-1])
                nb_unet_features = [enc_nf, dec_nf]
            else:
                nb_unet_features = unet_features
                nb_unet_conv_per_level = 1

            print(nb_unet_features)
            print(f'conv_per_level {nb_unet_conv_per_level}')
            unet_input_list = [im1_midspace, im2_midspace]
            dense_input_model = tf.keras.Model(label_input, unet_input_list)
            nfeats = len(unet_input_list) // 2
            dense_model1 = vxm.networks.VxmDense(inshape,
                                                 nb_unet_features=nb_unet_features,
                                                 bidir=True,
                                                 input_model=dense_input_model,
                                                 src_feats=nfeats,
                                                 trg_feats=nfeats,
                                                 fill_value=fill_value,
                                                 name='dense1',
                                                 nb_unet_conv_per_level=nb_unet_conv_per_level)
            dense_model2 = vxm.networks.VxmDense(inshape,
                                                 nb_unet_features=nb_unet_features,
                                                 bidir=True,
                                                 input_model=dense_input_model,
                                                 src_feats=nfeats,
                                                 trg_feats=nfeats,
                                                 fill_value=fill_value,
                                                 name='dense2',
                                                 nb_unet_conv_per_level=nb_unet_conv_per_level)

            # the midspace warp should be the composition of B0_1 with inv(B0_2)
            ref1 = dense_model1.references
            ref2 = dense_model2.references
            B0_warp1_fwd_est = KL.Lambda(lambda x: x, name='B0_warp1_fwd_est')(ref1.pos_flow)
            B0_warp1_inv_est = KL.Lambda(lambda x: x, name='B0_warp1_inv_est')(ref1.neg_flow)
            B0_warp2_fwd_est = KL.Lambda(lambda x: x, name='B0_warp2_fwd_est')(ref2.pos_flow)
            B0_warp2_inv_est = KL.Lambda(lambda x: x, name='B0_warp2_inv_est')(ref2.neg_flow)

        else:  # estimate scalar fields instead of full warps using a unet
            input_list = [im1_midspace, im2_midspace]
            unet_inputs = KL.Concatenate(axis=-1, name='unet_inputs')(input_list)
            num_outputs = 1 if one_d else ndims
            dense_model = ne.models.unet(
                unet_features,
                tuple(inshape) + (len(input_list),),
                unet_levels,
                3,              # conv size
                2 * num_outputs,              # nb_labels/nb_outputs
                nb_conv_per_level=nb_unet_conv_per_level,
                name='B0_correction_unet',
                final_pred_activation='linear',
            )
            dense_model2 = ne.models.unet(
                unet_features,
                tuple(inshape) + (len(input_list),),
                unet_levels,
                3,              # conv size
                ndims,              # nb_labels/nb_outputs
                nb_conv_per_level=nb_unet_conv_per_level,
                name='B0_reg_unet',
                final_pred_activation='linear',
            )

            # this code works if the unet outputs a ndim-vec and we don't constrain
            # the direction to be in the B0_Dir
            unet_outputs = dense_model(unet_inputs)   # this should be an estimate of B0 map1
            unet_outputs2 = dense_model2(unet_inputs)   # this should be an estimate of warp
            if one_d is True:
                print('creating 1D vector fields')
                B0_map1_est = KL.Lambda(lambda x: x[..., 0:1], name='B0_map1_est')(unet_outputs)
                B0_map2_est = KL.Lambda(lambda x: x[..., 1:2], name='B0_map2_est')(unet_outputs)
                mul_dir = lambda x: x[0] * x[1][:, tf.newaxis, ...]  # scale B0_dir vec field
                warp1_svf = KL.Lambda(mul_dir, name='B0_warp1_svf_est')([B0_map1_est, B0_dir1])
                warp2_svf = KL.Lambda(mul_dir, name='B0_warp2_svf_est')([B0_map2_est, B0_dir2])
            else:
                warp1_svf = KL.Lambda(lambda x: x[..., 0:ndims], name='B0_map1_est')(unet_outputs)
                warp2_svf = KL.Lambda(lambda x: x[..., ndims:], name='B0_map2_est')(unet_outputs)

            warp_mid1_to_2 = vxm.layers.VecInt(int_steps=5, name='mid_warp1to2')(unet_outputs2)
            warp_mid_neg_svf = ne.layers.Negate()(unet_outputs2)
            warp_mid2_to_1 = vxm.layers.VecInt(int_steps=5, name='mid_warp2to1')(warp_mid_neg_svf)

            B0_warp1_fwd_est = vxm.layers.VecInt(int_steps=5, name='B0_warp1_fwd_est')(warp1_svf)
            warp1_neg_svf = ne.layers.Negate()(warp1_svf)
            B0_warp1_inv_est = vxm.layers.VecInt(int_steps=5, name='B0_warp1_inv_est')(
                warp1_neg_svf)
            B0_warp2_fwd_est = vxm.layers.VecInt(int_steps=5, name='B0_warp2_fwd_est')(
                warp2_svf)
            warp2_neg_svf = ne.layers.Negate()(warp2_svf)
            B0_warp2_inv_est = vxm.layers.VecInt(int_steps=5, name='B0_warp2_inv_est')(
                warp2_neg_svf)

        # create outputs for losses
        warp_mid1_to_2 = vxm.layers.ComposeTransform(name='midspace_warp1_to_2_true')(
            [B0_warp1_inv, B0_warp2])
        warp_mid2_to_1 = vxm.layers.ComposeTransform(name='midspace_warp2_to_1_true')(
            [B0_warp2_inv, B0_warp1])
        warp_mid1_to_2_est = vxm.layers.ComposeTransform(name='midspace_warp1_to_2')(
            [B0_warp1_inv_est, B0_warp2_fwd_est])
        warp_mid2_to_1_est = vxm.layers.ComposeTransform(name='midspace_warp2_to_1')(
            [B0_warp2_inv_est, B0_warp1_fwd_est])
        im1_to_im2_est = vxm.layers.SpatialTransformer(
            name='im1_to_im2_est', interp_method='linear', fill_value=fill_value)(
                [im1_distorted, warp_mid1_to_2_est])
        im2_to_im1_est = vxm.layers.SpatialTransformer(
            name='im2_to_im1_est', interp_method='linear', fill_value=fill_value)(
                [im2_distorted, warp_mid2_to_1_est])
        im2_to_im1_no_atrophy_est = vxm.layers.SpatialTransformer(
            name='im2_no_atrophy_to_im1_est', interp_method='linear', fill_value=fill_value)(
                [im2_distorted_no_atrophy, warp_mid2_to_1_est])

        im1_reg_est = KL.Concatenate(axis=-1, name='im1_reg_est')(
            [im2_midspace_no_atrophy, im1_to_im2_est, labels2_midspace])
        im2_reg_est = KL.Concatenate(axis=-1, name='im2_reg_est')(
            [im1_midspace, im2_to_im1_no_atrophy_est, labels1_midspace])

        im1_distorted_est = vxm.layers.SpatialTransformer(
            name='im1_distorted_est', interp_method='linear',
            fill_value=fill_value)([synth_image1, B0_warp1_fwd_est])
        im2_distorted_est = vxm.layers.SpatialTransformer(
            name='im2_distorted_est', interp_method='linear',
            fill_value=fill_value)([synth_image2_no_atrophy, B0_warp2_fwd_est])
        im1_dist_outputs = [im1_distorted, im1_distorted_est, labels1_distorted]
        im2_dist_outputs = [im2_distorted_no_atrophy, im2_distorted_est,
                            labels2_distorted_no_atrophy]
        im1_dist_out = KL.Concatenate(axis=-1, name='im1_dist_l')(im1_dist_outputs)
        im2_dist_out = KL.Concatenate(axis=-1, name='im2_dist_l')(im2_dist_outputs)

        sl = sli[..., tf.newaxis]
        sl2 = KL.Concatenate(axis=3)([sl, sl])
        B0_w1_out = [B0_warp1[..., tf.newaxis], B0_warp1_fwd_est[..., tf.newaxis], sl2]
        B0_w2_out = [B0_warp2[..., tf.newaxis], B0_warp2_fwd_est[..., tf.newaxis], sl2]
        B0_warp1_out = KL.Concatenate(axis=-1, name='B0_warp1_l')(B0_w1_out)
        B0_warp2_out = KL.Concatenate(axis=-1, name='B0_warp2_l')(B0_w2_out)

        sl1 = labels1_midspace[..., tf.newaxis]
        sl1 = KL.Concatenate(axis=3)([sl1, sl1])
        sl2 = labels2_midspace[..., tf.newaxis]
        sl2 = KL.Concatenate(axis=3)([sl2, sl2])
        mw1_out = [warp_mid1_to_2[..., tf.newaxis], warp_mid1_to_2_est[..., tf.newaxis], sl1]
        mw2_out = [warp_mid1_to_2[..., tf.newaxis], warp_mid1_to_2_est[..., tf.newaxis], sl2]
        midspace_warp1_out = KL.Concatenate(axis=-1, name='mid_warp1')(mw1_out)
        midspace_warp2_out = KL.Concatenate(axis=-1, name='mid_warp2')(mw2_out)

        image_outputs = [im1_reg_est, im2_reg_est, im1_dist_out, im2_dist_out]
        warp_outputs = [B0_warp1_out, B0_warp2_out, midspace_warp1_out, midspace_warp2_out]
        smoothness_outputs = [warp_mid1_to_2, warp_mid2_to_1]
        outputs_nl = image_outputs + warp_outputs + smoothness_outputs

        if debug:  # make sure some intermediate stuff is in the model
            outputs_nl += [im1_distorted, im2_distorted, im2_distorted_no_atrophy,
                           im1_distorted_est, im2_distorted_est,
                           warp_mid1_to_2, warp_mid2_to_1]

        self.references = ne.tf.modelio.LoadableModel.ReferenceContainer()
        self.references.dense_model = dense_model

        super().__init__(name=name, inputs=label_input, outputs=outputs_nl)


class VxmSynthComboSaved(ne.tf.modelio.LoadableModel):
    """
    VoxelMorph network for combo linear (affine)/dense registration between two images.
    including nonlinear B0 correction.

    synth_image from label maps. Augment it with 2 random affines. Apply B0 distortion
    that is the same magnitude but different orientations (randomly drawn) to each of the
    two images. Then try to learn the inverse - a single scalar map applied in the two
    directions that should cancel the distortions.

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
            reserve_encoders=[1e-3],
            max_rot=10,
            min_B0_blur=1,
            max_B0_blur=20,
            max_B0_std=1,
            affine_conv_per_level=1,
            structure_list={},
            Conv=None,
            subsample_atrophy=1,
            reserve_unets=None,
            debug=False,
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
            seeds=seeds,         # same seeds as gen_model do intensity distributions are same
            **gen_args
        )

        # input a label map and use it to synthesize an image. Remove air
        label_input = KL.Input(shape=tuple(inshape) + (1,), name='label_in')

        # input vectors that give B0 distortion direction
        draw_dir = lambda _: tf.random.shuffle([1.0, 0.0, 0.0])[tf.newaxis, ...]
        B0_dir1 = KL.Lambda(lambda x: tf.map_fn(draw_dir, x), name='B0_dir1')(label_input)
        B0_dir2 = KL.Lambda(lambda x: tf.map_fn(draw_dir, x), name='B0_dir2')(label_input)

        label_input_with_atrophy = vxms.layers.ResizeLabels(
            structure_list, name='resize_labels', subsample_atrophy=subsample_atrophy)(label_input)

        synth_image, synth_labels = gen_model(label_input)
        atrophy_image, atrophy_labels = gen_model_a(label_input_with_atrophy)

        # skull_mask = tf.cast(synth_labels > 0, tf.float32)

        # extract the separate image contrasts (disabled for now)
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

        # sample B0 map (scalar) then transform it into 2 image spaces to create warps
        B0_map = vxms.layers.DrawB0Map(inshape, name='B0_map', max_std=max_B0_std,
                                       min_blur=min_B0_blur, max_blur=max_B0_blur)(B0_dir1)
        B0_map1 = vxm.layers.SpatialTransformer(
            name='B0_map1', interp_method='linear', fill_value=fill_value)([B0_map, mats1_noid])
        B0_map2 = vxm.layers.SpatialTransformer(
            name='B0_map2', interp_method='linear', fill_value=fill_value)([B0_map, mats2_noid])

        # compute B0 warp for each image (same B0 inhomogeneity, different direction)
        B0_warp1 = KL.Dot(axes=[-1, 1], name='B0_warp1')([B0_map1, B0_dir1])
        B0_warp2 = KL.Dot(axes=[-1, 1], name='B0_warp2')([B0_map2, B0_dir2])

        # composite warps that transform from synth images to B0 warped images
        warp1_aug_B0 = vxm.layers.ComposeTransform(name='warp1_aug_B0')([mats1_noid, B0_warp1])
        warp2_aug_B0 = vxm.layers.ComposeTransform(name='warp2_aug_B0')([mats2_noid, B0_warp2])

        # apply the distortions to the images
        im1_distorted = vxm.layers.SpatialTransformer(
            name='im1_distorted', interp_method='linear',
            fill_value=fill_value)([synth_image1, warp1_aug_B0])
        im2_distorted = vxm.layers.SpatialTransformer(
            name='im2_distorted', interp_method='linear',
            fill_value=fill_value)([synth_image2, warp2_aug_B0])
        im2_distorted_no_atrophy = vxm.layers.SpatialTransformer(
            name='im2_distorted_no_atrophy', interp_method='linear',
            fill_value=fill_value)([synth_image2_no_atrophy, warp2_aug_B0])

        # compute an affine that aligns the two distorted images (loss is based on
        # unseen at test time undistorted images)
        combo_inputs = [im1_distorted, im2_distorted]
        input_model = tf.keras.models.Model(inputs=label_input, outputs=combo_inputs)

        vxm_affine = VxmAffineEncoderThenDense(
            inshape,
            enc_nf=affine_nf,
            input_model=input_model,
            fill_value=fill_value,
            reserve_encoders=reserve_encoders,
            Conv=Conv,
            store_midspace=True,
            bidir=True,
            **kwargs)

        # compose the augmentation/B0 warp with the vxm-computed affine warp transform
        # this is really just the affine alignment of the distorted volumes, but avoids
        # the multiple resamplings (synth->aug->B0 dist->target im)
        if 1:
            affine1 = vxm_affine.references.affine
            affine2 = vxm_affine.references.inv_affine
        else:    # disable the affine
            inv_aug1 = vxm.layers.InvertAffine(name='inv_aug1')(mats1_noid)
            inv_aug2 = vxm.layers.InvertAffine(name='inv_aug2')(mats2_noid)
            affine1 = vxm.layers.ComposeTransform(name='fake_aff1')([inv_aug1, mats2_noid])
            affine2 = vxm.layers.ComposeTransform(name='fake_aff2')([inv_aug2, mats1_noid])

        half1 = vxms.layers.MidspaceTransform(name='half1')(affine1)
        half2 = vxms.layers.MidspaceTransform(name='half2')(affine2)

        warp1_synth_to_im2_aff = vxm.layers.ComposeTransform(name='comp1')([warp1_aug_B0, affine1])
        warp2_synth_to_im1_aff = vxm.layers.ComposeTransform(name='comp2')([warp2_aug_B0, affine2])
        warp1_synth_to_midspace = vxm.layers.ComposeTransform(name='warp1_synth_to_mid')(
            [warp1_aug_B0, half1])
        warp2_synth_to_midspace = vxm.layers.ComposeTransform(name='warp2_synth_to_mid')(
            [warp2_aug_B0, half2])

        # map images to midspace
        im1_midspace = vxm.layers.SpatialTransformer(
            name='im1_midspace', interp_method='linear',
            fill_value=fill_value)([synth_image1, warp1_synth_to_midspace])
        im2_midspace = vxm.layers.SpatialTransformer(
            name='im2_midspace', interp_method='linear',
            fill_value=fill_value)([synth_image2, warp2_synth_to_midspace])
        im2_midspace_no_atrophy = vxm.layers.SpatialTransformer(
            name='im2_midspace_no_atrophy', interp_method='linear',
            fill_value=fill_value)([synth_image2_no_atrophy, warp2_synth_to_midspace])

        # map images to each other's coordinates
        im1_to_im2 = vxm.layers.SpatialTransformer(
            name='im1_to_im2', interp_method='linear',
            fill_value=fill_value)([synth_image1, warp1_synth_to_im2_aff])
        im2_to_im1 = vxm.layers.SpatialTransformer(
            name='im2_to_im1', interp_method='linear',
            fill_value=fill_value)([synth_image2, warp2_synth_to_im1_aff])
        im2_to_im1_no_atrophy = vxm.layers.SpatialTransformer(
            name='im2_to_im1', interp_method='linear',
            fill_value=fill_value)([synth_image2_no_atrophy, warp2_synth_to_im1_aff])

        inshape_vox = np.array(inshape).prod()
        B0_dir1_flat = KL.Flatten()(B0_dir1)
        B0_dir2_flat = KL.Flatten()(B0_dir2)
        B0_dir1_dense = KL.Dense(inshape_vox, name='B0_dir1_dense')(B0_dir1_flat)
        B0_dir2_dense = KL.Dense(inshape_vox, name='B0_dir2_dense')(B0_dir2_flat)
        B0_dir1_im = KL.Reshape(tuple(inshape) + (1,), name='B0_dir1_im')(B0_dir1_dense)
        B0_dir2_im = KL.Reshape(tuple(inshape) + (1,), name='B0_dir2_im')(B0_dir2_dense)
        input_list = [im1_midspace, im2_midspace, B0_dir1_im, B0_dir2_im]
        unet_inputs = KL.Concatenate(name='unet_inputs', axis=-1)(input_list)
        unet_model = ne.models.unet(
            unet_features,
            tuple(inshape) + (len(input_list),),
            unet_levels,
            3,              # conv size
            1,              # nb_labels/nb_outputs
            nb_conv_per_level=nb_unet_conv_per_level,
            name='B0_correction_unet',
            final_pred_activation='linear',
        )

        # output of unet should be magnitude of (inverse) distortions. Apply it in each dir
        unet_outputs = unet_model(unet_inputs)
        B0_unmap = KL.Lambda(lambda x: x[..., 0:1], name='B0_unmap')(unet_outputs)
        B0_unwarp1 = KL.Dot(axes=[-1, 1], name='B0_unwarp1')([B0_unmap, B0_dir1])
        B0_unwarp2 = KL.Dot(axes=[-1, 1], name='B0_unwarp2')([B0_unmap, B0_dir2])

        # the unet output should cancel the B0 warp, so the following 2 should be zero warps
        B0_cancelled1 = vxm.layers.ComposeTransform(name='B0_cancelled1')([B0_warp1, B0_unwarp1])
        B0_cancelled2 = vxm.layers.ComposeTransform(name='B0_cancelled2')([B0_warp2, B0_unwarp2])

        # apply the composed warp should ideally get you back to the augmented (undistorted) images
        # adding the mats?_noid gets you back to the original synth images
        warp1_B0_correct = vxm.layers.ComposeTransform(name='warp1_B0_correct')(
            [mats1_noid, B0_cancelled1])
        warp2_B0_correct = vxm.layers.ComposeTransform(name='warp2_B0_correct')(
            [mats2_noid, B0_cancelled2])
        im1_corrected = vxm.layers.SpatialTransformer(
            name='im1_corrected', interp_method='linear',
            fill_value=fill_value)([synth_image1, warp1_B0_correct])
        im2_corrected = vxm.layers.SpatialTransformer(
            name='im2_corrected', interp_method='linear',
            fill_value=fill_value)([synth_image2_no_atrophy, warp2_B0_correct])

        # now generate midspace corrected images
        warp1_midspace_corrected = vxm.layers.ComposeTransform(name='warp1_midspace_corrected')(
            [mats1_noid, B0_cancelled1, half1])
        warp2_midspace_corrected = vxm.layers.ComposeTransform(name='warp2_midspace_corrected')(
            [mats2_noid, B0_cancelled2, half2])
        im1_midspace_corrected = vxm.layers.SpatialTransformer(
            name='im1_midspace_corrected', interp_method='linear',
            fill_value=fill_value)([synth_image1, warp1_midspace_corrected])
        im2_midspace_corrected = vxm.layers.SpatialTransformer(
            name='im2_midspace_corrected', interp_method='linear',
            fill_value=fill_value)([synth_image2_no_atrophy, warp2_midspace_corrected])
        midspace_outputs = KL.Concatenate(axis=-1, name='midspace_outputs')(
            [im1_midspace_corrected, im2_midspace_corrected])

        # apply the composed half and full transforms to images and labels
        # use the synth2 image without atrophy so that we learn a transform that
        # is invariant to atrophy
        # appling aug, B0, inv B0 should get back to original (rigidly augmented) image
        labels1_corrected = vxm.layers.SpatialTransformer(
            name='labels1_corrected', interp_method='nearest',
            fill_value=fill_value)([tf.cast(synth_labels, dtype=tf.float32), warp1_B0_correct])
        labels2_corrected = vxm.layers.SpatialTransformer(
            name='labels2_corrected', interp_method='nearest',
            fill_value=fill_value)([tf.cast(atrophy_labels, dtype=tf.float32), warp2_B0_correct])

        # apply the composed half and full transforms to images and labels
        labels1_to_im2 = vxm.layers.SpatialTransformer(
            name='labels1_to_im2', interp_method='nearest', fill_value=fill_value)(
                [tf.cast(synth_labels, dtype=tf.float32), warp1_synth_to_im2_aff])
        labels2_to_im1 = vxm.layers.SpatialTransformer(
            name='labels2_to_im1', interp_method='nearest', fill_value=fill_value)(
                [tf.cast(atrophy_labels, dtype=tf.float32), warp1_synth_to_im2_aff])

        moving_outputs = KL.Concatenate(axis=-1, name='moving_l')(
            [im2_distorted, im1_to_im2, labels2_transformed, labels1_to_im2])
        fixed_outputs = KL.Concatenate(axis=-1, name='fixed_l')(
            [im1_distorted, im2_to_im1, labels1_transformed, labels2_to_im1])

        moving_outputs_nl = KL.Concatenate(axis=-1, name='moving_out_nl')(
            [im1_transformed, im1_corrected, labels1_transformed, labels1_corrected])
        fixed_outputs_nl = KL.Concatenate(axis=-1, name='fixed_out_nl')(
            [im2_transformed_no_atrophy, im2_corrected, labels2_transformed, labels2_corrected])

        # return labels in fixed/moving space in case user wants them
        outputs_nl = [moving_outputs_nl, fixed_outputs_nl, B0_unwarp1, B0_unwarp2]
        outputs_nl = [midspace_outputs, B0_unwarp1, B0_unwarp2]
        if debug:
            outputs_nl += [im1_distorted, im2_distorted,
                           im2_corrected, im2_midspace_no_atrophy]

        # these are (unobservable at test) affine aligned undisotrted images
        affine1_comp = vxm.layers.ComposeTransform(name='affine1_comp')([mats1_noid, affine1])
        affine2_comp = vxm.layers.ComposeTransform(name='affine2_comp')([mats2_noid, affine2])
        im1_affine = vxm.layers.SpatialTransformer(name='im1_affine', fill_value=0)(
            [synth_image1, affine1_comp])
        im2_affine = vxm.layers.SpatialTransformer(name='im2_affine', fill_value=0)(
            [synth_image2_no_atrophy, affine2_comp])
        affine_out1 = KL.Concatenate(axis=-1, name='affine_out1')(
            [im2_transformed_no_atrophy, im1_affine])
        affine_out2 = KL.Concatenate(axis=-1, name='affine_out2')([im1_transformed, im2_affine])

        # make some things available to the caller for use in loss functions
        # including the undistorted affine-transformed volumes and the nonlinear
        # component of the combined warp

        self.references = ne.tf.modelio.LoadableModel.ReferenceContainer()
        self.references.vxm_affine = vxm_affine
        self.references.unet_model = unet_model
        self.references.B0_cancelled1 = B0_cancelled1
        self.references.B0_cancelled2 = B0_cancelled2
        self.references.affine_out1 = affine_out1
        self.references.affine_out2 = affine_out2
        self.references.im1 = synth_image1
        self.references.im2 = synth_image2
        self.references.im1_corrected = im1_corrected
        self.references.im2_corrected = im2_corrected
        self.references.moving_outputs_nl = moving_outputs_nl
        self.references.fixed_outputs_nl = fixed_outputs_nl
        self.references.outputs_nl = outputs_nl

        super().__init__(name=name, inputs=label_input, outputs=outputs_nl)


def remove_skull(mri, skull_labels=None):
    if skull_labels is None:
        skull_labels = [259, 258, 165]

    inshape = tf.shape(mri)
    skull_mask = tf.ones(inshape, tf.bool)
    for label in skull_labels:
        lmask = tf.where(mri == label, tf.zeros(inshape, tf.bool), tf.ones(inshape, tf.bool))
        skull_mask = tf.logical_and(skull_mask, lmask)

    mri = tf.logical_and(tf.cast(mri, tf.bool), skull_mask)

    return mri


class VxmSynthRigid(ne.tf.modelio.LoadableModel):
    """
    VoxelMorph network for combo linear (affine)/dense registration between two images.
    including nonlinear B0 correction.

    synth_image from label maps. Augment it with 2 random affines. Apply B0 distortion
    that is the same magnitude but different orientations (randomly drawn) to each of the
    two images. Then try to learn the inverse - a single scalar map applied in the two
    directions that should cancel the distortions.

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
            fill_value=None,
            name='VxmSynthRigid',
            max_trans=15,
            reserve_encoders=[1e-3],
            max_rot=10,
            min_B0_blur=1,
            max_B0_blur=20,
            max_B0_std=1,
            affine_conv_per_level=1,
            structure_list={},
            subsample_atrophy=1,
            debug=False,
            **kwargs):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            name: Model name. Default is 'vxmSynthRigid'.
            subsample_atrophy - if not None, specifies what fraction of the total number of
                                structures to simulate atrophy in for each sample
        """

        """
        some notes on naming. The sequence of transforms goes:
        synth->transformed (augmentation) -> B0 distorted

        So the images are synth, transformed, distorted then an affine midspace transform
        and finally a nonlinear dense that goes from midspace 1 to midspace 2 (and also
        estimates a decomposition into B0 warps for the 2 images).

        variables beginning with i are images, w are warps, l are label maps, b is brain mask,
        and a are affine transforms
        the spaces they live in are:
        s - synth
        a - rigid augmentation
        d - B0 distorted
        m - midspace

        note also that the synth space is shared so doesn't need a numerical suffix 
        (i.e. s, not s1)
        so i_a1 is image1 after rigid augmentation and w_s_to_d1 is a warp that goes from
        synth image 1 to distorted image 1. na suffix indicates it is the image or label map
        with no atrophy (e.g. i_m2_na)
        also note that the 'a' and 'd' spaces are the same from an affine perspective, so the
        a_s_to_a would be the same as a_s_to_d

        warps and images with _e are estimated and _t are true (constructed using the 
        synthesized warps and affines)
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
            seeds=seeds,         # same seeds as gen_model do intensity distributions are same
            **gen_args
        )

        # input a label map and use it to synthesize an image. Remove air
        label_input = KL.Input(shape=tuple(inshape) + (1,), name='label_in')

        # synthesize vectors that give B0 distortion direction
        vec = [1.0, 0.0, 0.0][0:ndims]
        draw_dir = lambda _: tf.random.shuffle(vec)[tf.newaxis, ...]
        sign_flip = lambda x: x * tf.random.shuffle([1.0, -1.0])[0]
        B0_dir1 = KL.Lambda(lambda x: tf.map_fn(draw_dir, x), name='B0_dir1d')(label_input)
        B0_dir2 = KL.Lambda(lambda x: tf.map_fn(draw_dir, x), name='B0_dir2d')(label_input)
        B0_dir1 = KL.Lambda(lambda x: tf.map_fn(sign_flip, x), name='B0_dir1')(B0_dir1)
        B0_dir2 = KL.Lambda(lambda x: tf.map_fn(sign_flip, x), name='B0_dir2')(B0_dir2)

        # synthesize images with atrophy in them
        label_input_with_atrophy = vxms.layers.ResizeLabels(
            structure_list, name='resize_labels', subsample_atrophy=subsample_atrophy)(
                label_input)

        synth_image, synth_labels = gen_model(label_input)
        atrophy_image, atrophy_labels = gen_model_a(label_input_with_atrophy)

        # extract the separate image contrasts (disabled for now)
        synth_image1 = KL.Lambda(lambda x: x[..., 0:1], name='synth_image1')(synth_image)
        synth_image2 = KL.Lambda(lambda x: x[..., 0:1], name='synth_image2')(atrophy_image)  # 1:2
        synth_image2_no_atrophy = KL.Lambda(lambda x: x[..., 0:1],
                                            name='synth_image2_no_atrophy')(synth_image)

        # sample rigid augmentation transforms and apply them to the images
        i_a1, a_s_to_a1 = vxms.layers.AffineAugment(
            name='i_a1', max_trans=max_trans, max_rot=max_rot,
            return_mats=True, limit_to_fov=True)([synth_image1])
        i_a2, a_s_to_a2 = vxms.layers.AffineAugment(
            name='i_a2', max_trans=max_trans, max_rot=max_rot,
            return_mats=True, limit_to_fov=True)([synth_image2])
        i_a2_na = vxm.layers.SpatialTransformer(
            name='i_a2_na', interp_method='linear',
            fill_value=fill_value)([synth_image2_no_atrophy, a_s_to_a2])
        l_a2_na = vxm.layers.SpatialTransformer(
            name='l_a2_na', interp_method='nearest',
            fill_value=fill_value)([tf.cast(synth_labels, tf.float32), a_s_to_a2])
        l_a1 = vxm.layers.SpatialTransformer(name='l_a1', interp_method='nearest',
                                             fill_value=fill_value)([
                                                 tf.cast(synth_labels, tf.float32), a_s_to_a1])

        # brain mask vols
        b_a1 = vxm.layers.SpatialTransformer(name='b_a1', interp_method='linear',
                                             fill_value=fill_value)([brain_mask, a_s_to_a1])
        b_a2_na = vxm.layers.SpatialTransformer(name='b_a2_na', interp_method='linear',
                                                fill_value=fill_value)([brain_mask, a_s_to_a2])
        b_a2 = vxm.layers.SpatialTransformer(name='b_a2', interp_method='linear',
                                             fill_value=fill_value)([brain_mask_wa, a_s_to_a2])

        # compute an affine that aligns the two distorted images (loss is based on
        # unseen at test time undistorted images)
        combo_inputs = [i_d1, i_d2]
        input_model = tf.keras.models.Model(inputs=label_input, outputs=combo_inputs)

        if estimate_affine:
            if use_decomp:
                vxm_affine = VxmAffineDecomposeWarp(
                    inshape,
                    nb_unet_features=affine_nf,
                    input_model=input_model,
                    fill_value=fill_value,
                    nb_unet_conv_per_level=2,
                    bidir=True,
                    nb_unet_levels=4)
            else:
                vxm_affine = VxmAffineEncoderThenDense(
                    inshape,
                    enc_nf=affine_nf,
                    input_model=input_model,
                    fill_value=fill_value,
                    reserve_encoders=reserve_encoders,
                    Conv=Conv,
                    store_midspace=True,
                    **kwargs)
        else:
            vxm_affine = None

        # compose the augmentation/B0 warp with the vxm-computed affine warp transform
        # this is really just the affine alignment of the distorted volumes, but avoids
        # the multiple resamplings (synth->aug->B0 dist->target im)

        # the true affine transforms go from one augmented/distorted image to the other
        a_a1_to_s = vxm.layers.InvertAffine(name='inv_aug1')(a_s_to_a1)
        a_a2_to_s = vxm.layers.InvertAffine(name='inv_aug2')(a_s_to_a2)
        a_a1_to_a2 = vxm.layers.ComposeTransform(name='true_aff1')([a_a1_to_s, a_s_to_a2])
        a_a2_to_a1 = vxm.layers.ComposeTransform(name='true_aff2')([a_a2_to_s, a_s_to_a1])

        # now create midspace transforms
        a_a1_to_m1 = vxms.layers.MidspaceTransform(name='half1_true')(a_a1_to_a2)
        a_a2_to_m2 = vxms.layers.MidspaceTransform(name='half2_true')(a_a2_to_a1)
        a_m1_to_a1 = vxm.layers.InvertAffine(name='half1_inv')(a_a1_to_m1)
        a_m2_to_a2 = vxm.layers.InvertAffine(name='half2_inv')(a_a2_to_m2)

        i_a1_to_m1 = vxm.layers.SpatialTransformer(
            name='im1_midspace', interp_method='linear',
            fill_value=fill_value)([i_a1, a_a1_to_m1])
        i_a2_to_m2 = vxm.layers.SpatialTransformer(
            name='im2_midspace', interp_method='linear',
            fill_value=fill_value)([i_a2, a_a2_to_m2])
        i_a1_to_a2 = vxm.layers.SpatialTransformer(
            name='im2_to_a1', interp_method='linear',
            fill_value=fill_value)([i_a1, a_a1_to_a2])
        if estimate_affine:
            a_a1_to_a2_est = vxm_affine.references.affine
            a_a2_to_a1_est = vxm_affine.references.inv_affine
        else:    # disable the affine and use true transforms instead
            a_a1_to_a2_est = a_a1_to_a2
            a_a2_to_a1_est = a_a2_to_a1

        # transforms from distorted images to the midspace between them
        a_a1_to_m1_est = vxms.layers.MidspaceTransform(name='half1_est')(a_a1_to_a2_est)
        a_a2_to_m2_est = vxms.layers.MidspaceTransform(name='half2_est')(a_a1_to_a2_est)
        a_m1_to_a1_est = vxm.layers.InvertAffine(name='half1_inv_est')(a_a1_to_m1_est)
        a_m2_to_a2_est = vxm.layers.InvertAffine(name='half2_inv_est')(a_a2_to_m2_est)

        # map images to each other
        i_d2_to_d1_est = vxm.layers.SpatialTransformer(
            name='i_d2_to_d1_est', interp_method='linear',
            fill_value=fill_value)([i_d2, a_a2_to_a1_est])
        i_d1_to_d2_est = vxm.layers.SpatialTransformer(
            name='i_d1_to_d2_est', interp_method='linear',
            fill_value=fill_value)([i_d1, a_a1_to_a2_est])

        # map images to midspace
        i_m1 = vxm.layers.SpatialTransformer(name='im1_midspace', interp_method='linear',
                                             fill_value=fill_value)([synth_image1, w_s_to_m1])
        i_m2 = vxm.layers.SpatialTransformer(name='im2_midspace', interp_method='linear',
                                             fill_value=fill_value)([synth_image2, w_s_to_m2])
        i_m2_na = vxm.layers.SpatialTransformer(
            name='im2_midspace_no_atrophy', interp_method='linear',
            fill_value=fill_value)([synth_image2_no_atrophy, w_s_to_m2])
        b_m1 = vxm.layers.SpatialTransformer(name='b_m1', interp_method='linear',
                                             fill_value=fill_value)([brain_mask, w_s_to_m1])
        b_m2 = vxm.layers.SpatialTransformer(name='b_m2', interp_method='linear',
                                             fill_value=fill_value)([brain_mask_wa, w_s_to_m2])
        b_m2_na = vxm.layers.SpatialTransformer(name='b_m2_na', interp_method='linear',
                                                fill_value=fill_value)([brain_mask, w_s_to_m2])

        # create outputs for losses

        # build true warps that go from midspace im1->im2 and visa-versa
        w_m1_to_m2 = vxm.layers.ComposeTransform(name='warp_mid1_to_2_true')(
            [a_m1_to_a1, B0_warp1_inv, a_a1_to_s, a_s_to_a2, B0_warp2, a_a2_to_m2])
        w_m2_to_m1 = vxm.layers.ComposeTransform(name='warp_mid2_to_1_true')(
            [a_m2_to_a2, B0_warp2_inv, a_a2_to_s, a_s_to_a1, B0_warp1, a_a1_to_m1])

        i_d1_est = vxm.layers.SpatialTransformer(
            name='i_d1_est', interp_method='linear',
            fill_value=fill_value)([i_a1, B0_warp1_fwd_est])
        i_d2_est = vxm.layers.SpatialTransformer(
            name='i_d2_est', interp_method='linear',
            fill_value=fill_value)([i_a2_na, B0_warp2_fwd_est])
        im1_dist_outputs = [i_d1, i_d1_est, b_d1]
        im2_dist_outputs = [i_d2_na, i_d2_est, b_d2_na]
        im1_dist_out = KL.Concatenate(axis=-1, name='im1_dist_l')(im1_dist_outputs)
        im2_dist_out = KL.Concatenate(axis=-1, name='im2_dist_l')(im2_dist_outputs)

        # map brainmask to estimated affine locations
        b_a1_to_a2_est = vxm.layers.SpatialTransformer(
            name='b_a1_to_a2_est', interp_method='linear',
            fill_value=fill_value)([b_a1, a_a1_to_a2_est])
        b_a2_to_a1_est = vxm.layers.SpatialTransformer(
            name='b_a2_to_a1_est', interp_method='linear',
            fill_value=fill_value)([b_a2, a_a2_to_a1_est])

        # build estimated affines from synth to avoid multiple interpolations
        a_s_to_a2_est = vxm.layers.ComposeTransform(name='a_s_to_a2_est')(
            [a_s_to_a1, a_a1_to_a2_est])
        a_s_to_a1_est = vxm.layers.ComposeTransform(name='a_s_to_a1_est')(
            [a_s_to_a2, a_a2_to_a1_est])
        i_a1_to_a2_est = vxm.layers.SpatialTransformer(name='i_a1_to_a2_est', fill_value=0)(
            [synth_image1, a_s_to_a2_est])
        i_a2na_to_a1_est = vxm.layers.SpatialTransformer(name='i_a2na_to_a1_est', fill_value=0)(
            [synth_image2_no_atrophy, a_s_to_a1_est])
        if debug:  # make sure some intermediate stuff is in the model
            i_a2_to_a1_est = vxm.layers.SpatialTransformer(name='i_a2_to_a1_est', fill_value=0)(
                [i_a2, a_a2_to_a1_est])
            outputs_nl += [i_d1, i_d2, i_d2_na, i_a2_to_a1_est,
                           i_d1_est, i_d2_est, B0_warp1_inv_est, B0_warp2_inv_est]

        aout1 = [i_a2_na, i_a1_to_a2_est, b_a2_na, b_a1_to_a2_est]
        aout2 = [i_a1, i_a2na_to_a1_est, b_a1, b_a2_to_a1_est]
        affine_out1 = KL.Concatenate(axis=-1, name='affine_out1')(aout1)
        affine_out2 = KL.Concatenate(axis=-1, name='affine_out2')(aout2)
        # affine_out1 = KL.Concatenate(axis=-1, name='affine_out1')([i_a2_na, i_a1_to_a2_est])
        # affine_out2 = KL.Concatenate(axis=-1, name='affine_out2')([i_a1, i_a2na_to_a1_est])
        self.references = ne.tf.modelio.LoadableModel.ReferenceContainer()
        self.references.affine_out1 = affine_out1
        self.references.affine_out2 = affine_out2
        self.references.vxm_affine = vxm_affine

        super().__init__(name=name, inputs=label_input, outputs=outputs_nl)


class VxmSynthComboTest2(ne.tf.modelio.LoadableModel):
    """
    VoxelMorph network for combo linear (affine)/dense registration between two images.
    including nonlinear B0 correction.

    synth_image from label maps. Augment it with 2 random affines. Apply B0 distortion
    that is the same magnitude but different orientations (randomly drawn) to each of the
    two images. Then try to learn the inverse - a single scalar map applied in the two
    directions that should cancel the distortions.

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
            name='VxmSynthAffineWithCorrection',
            max_trans=15,
            reserve_encoders=[1e-3],
            max_rot=10,
            min_B0_blur=1,
            max_B0_blur=20,
            max_B0_std=1,
            affine_conv_per_level=1,
            structure_list={},
            Conv=None,
            subsample_atrophy=1,
            reserve_unets=None,
            debug=False,
            unet_input_model=None,
            one_d=False,
            use_decomp=False,
            estimate_affine=True,
            **kwargs):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            name: Model name. Default is 'vxm_affine'.
            subsample_atrophy - if not None, specifies what fraction of the total number of
                                structures to simulate atrophy in for each sample
        """

        """
        some notes on naming. The sequence of transforms goes:
        synth->transformed (augmentation) -> B0 distorted

        So the images are synth, transformed, distorted then an affine midspace transform
        and finally a nonlinear dense that goes from midspace 1 to midspace 2 (and also
        estimates a decomposition into B0 warps for the 2 images).

        variables beginning with i are images, w are warps, l are label maps, b is brain mask,
        and a are affine transforms
        the spaces they live in are:
        s - synth
        a - rigid augmentation
        d - B0 distorted
        m - midspace

        note also that the synth space is shared so doesn't need a numerical suffix 
        (i.e. s, not s1)
        so i_a1 is image1 after rigid augmentation and w_s_to_d1 is a warp that goes from
        synth image 1 to distorted image 1. na suffix indicates it is the image or label map
        with no atrophy (e.g. i_m2_na)
        also note that the 'a' and 'd' spaces are the same from an affine perspective, so the
        a_s_to_a would be the same as a_s_to_d

        warps and images with _e are estimated and _t are true (constructed using the 
        synthesized warps and affines)
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
            seeds=seeds,         # same seeds as gen_model do intensity distributions are same
            **gen_args
        )

        # input a label map and use it to synthesize an image. Remove air
        label_input = KL.Input(shape=tuple(inshape) + (1,), name='label_in')

        # synthesize vectors that give B0 distortion direction
        vec = [1.0, 0.0, 0.0][0:ndims]
        draw_dir = lambda _: tf.random.shuffle(vec)[tf.newaxis, ...]
        sign_flip = lambda x: x * tf.random.shuffle([1.0, -1.0])[0]
        B0_dir1 = KL.Lambda(lambda x: tf.map_fn(draw_dir, x), name='B0_dir1d')(label_input)
        B0_dir2 = KL.Lambda(lambda x: tf.map_fn(draw_dir, x), name='B0_dir2d')(label_input)
        B0_dir1 = KL.Lambda(lambda x: tf.map_fn(sign_flip, x), name='B0_dir1')(B0_dir1)
        B0_dir2 = KL.Lambda(lambda x: tf.map_fn(sign_flip, x), name='B0_dir2')(B0_dir2)

        # synthesize images with atrophy in them
        label_input_with_atrophy = vxms.layers.ResizeLabels(
            structure_list, name='resize_labels', subsample_atrophy=subsample_atrophy)(
                label_input)

        synth_image, synth_labels = gen_model(label_input)
        atrophy_image, atrophy_labels = gen_model_a(label_input_with_atrophy)

        # extract the separate image contrasts (disabled for now)
        synth_image1 = KL.Lambda(lambda x: x[..., 0:1], name='synth_image1')(synth_image)
        synth_image2 = KL.Lambda(lambda x: x[..., 0:1], name='synth_image2')(atrophy_image)  # 1:2
        synth_image2_no_atrophy = KL.Lambda(lambda x: x[..., 0:1],
                                            name='synth_image2_no_atrophy')(synth_image)

        # sample a rigid transform  and apply it to the image (also disabled)
        B0_map = vxms.layers.DrawB0Map(inshape, name='B0_map_unmasked', max_std=max_B0_std,
                                       min_blur=min_B0_blur, max_blur=max_B0_blur)(B0_dir1)
        skull_mask = tf.cast(synth_labels > 0, tf.float32)
        brain_mask = tf.cast(remove_skull(synth_labels), tf.float32)
        brain_mask_wa = tf.cast(remove_skull(atrophy_labels), tf.float32)
        B0_map = KL.Multiply(name='B0_map')([B0_map, skull_mask])  # no B0 effects in air

        # sample rigid augmentation transforms and apply them to the images
        i_a1, a_s_to_a1 = vxms.layers.AffineAugment(
            name='i_a1', max_trans=max_trans, max_rot=max_rot,
            return_mats=True, limit_to_fov=True)([synth_image1])
        i_a2, a_s_to_a2 = vxms.layers.AffineAugment(
            name='i_a2', max_trans=max_trans, max_rot=max_rot,
            return_mats=True, limit_to_fov=True)([synth_image2])
        i_a2_na = vxm.layers.SpatialTransformer(
            name='i_a2_na', interp_method='linear',
            fill_value=fill_value)([synth_image2_no_atrophy, a_s_to_a2])
        l_a2_na = vxm.layers.SpatialTransformer(
            name='l_a2_na', interp_method='nearest',
            fill_value=fill_value)([tf.cast(synth_labels, tf.float32), a_s_to_a2])
        l_a1 = vxm.layers.SpatialTransformer(name='l_a1', interp_method='nearest',
                                             fill_value=fill_value)([
                                                 tf.cast(synth_labels, tf.float32), a_s_to_a1])

        # brain mask vols
        b_a1 = vxm.layers.SpatialTransformer(name='b_a1', interp_method='linear',
                                             fill_value=fill_value)([brain_mask, a_s_to_a1])
        b_a2_na = vxm.layers.SpatialTransformer(name='b_a2_na', interp_method='linear',
                                                fill_value=fill_value)([brain_mask, a_s_to_a2])
        b_a2 = vxm.layers.SpatialTransformer(name='b_a2', interp_method='linear',
                                             fill_value=fill_value)([brain_mask_wa, a_s_to_a2])

        # transform the susceptibility maps into the new coordinates
        B0_map1 = vxm.layers.SpatialTransformer(
            name='B0_map1', interp_method='linear', fill_value=fill_value)([B0_map, a_s_to_a1])
        B0_map2 = vxm.layers.SpatialTransformer(
            name='B0_map2', interp_method='linear', fill_value=fill_value)([B0_map, a_s_to_a2])

        # compute B0 warp for each image (same B0 inhomogeneity, different direction)
        B0_warp1_svf = KL.Dot(axes=[-1, 1], name='B0_warp1_svf')([B0_map1, B0_dir1])
        B0_warp2_svf = KL.Dot(axes=[-1, 1], name='B0_warp2_svf')([B0_map2, B0_dir2])
        B0_warp1 = vxm.layers.VecInt(int_steps=5, name='B0_warp1')(B0_warp1_svf)
        B0_warp2 = vxm.layers.VecInt(int_steps=5, name='B0_warp2')(B0_warp2_svf)
        B0_warp1_neg_svf = ne.layers.Negate()(B0_warp1_svf)
        B0_warp2_neg_svf = ne.layers.Negate()(B0_warp2_svf)
        B0_warp1_inv = vxm.layers.VecInt(int_steps=5, name='B0_warp1_inv')(B0_warp1_neg_svf)
        B0_warp2_inv = vxm.layers.VecInt(int_steps=5, name='B0_warp2_inv')(B0_warp2_neg_svf)

        # composite warps that transform from synth images to B0 warped images
        w_s_to_d1 = vxm.layers.ComposeTransform(name='w_s_to_d1')([a_s_to_a1, B0_warp1])
        w_s_to_d2 = vxm.layers.ComposeTransform(name='w_s_to_d2')([a_s_to_a2, B0_warp2])

        # apply the distortions to the images
        i_d1 = vxm.layers.SpatialTransformer(
            name='i_d1', interp_method='linear',
            fill_value=fill_value)([synth_image1, w_s_to_d1])
        i_d2 = vxm.layers.SpatialTransformer(
            name='i_d2', interp_method='linear',
            fill_value=fill_value)([synth_image2, w_s_to_d2])
        i_d2_na = vxm.layers.SpatialTransformer(
            name='i_d2_na', interp_method='linear',
            fill_value=fill_value)([synth_image2_no_atrophy, w_s_to_d2])

        b_d1 = vxm.layers.SpatialTransformer(name='b_d1', interp_method='linear',
                                             fill_value=fill_value)([brain_mask, w_s_to_d1])
        b_d2 = vxm.layers.SpatialTransformer(name='b_d2', interp_method='linear',
                                             fill_value=fill_value)([brain_mask_wa, w_s_to_d2])
        b_d2_na = vxm.layers.SpatialTransformer(name='b_d2_na', interp_method='linear',
                                                fill_value=fill_value)([brain_mask, w_s_to_d2])

        # compute an affine that aligns the two distorted images (loss is based on
        # unseen at test time undistorted images)
        combo_inputs = [i_d1, i_d2]
        input_model = tf.keras.models.Model(inputs=label_input, outputs=combo_inputs)

        if estimate_affine:
            if use_decomp:
                vxm_affine = VxmAffineDecomposeWarp(
                    inshape,
                    nb_unet_features=affine_nf,
                    input_model=input_model,
                    fill_value=fill_value,
                    nb_unet_conv_per_level=2,
                    bidir=True,
                    nb_unet_levels=4)
            else:
                vxm_affine = VxmAffineEncoderThenDense(
                    inshape,
                    enc_nf=affine_nf,
                    input_model=input_model,
                    fill_value=fill_value,
                    reserve_encoders=reserve_encoders,
                    Conv=Conv,
                    store_midspace=True,
                    **kwargs)
        else:
            vxm_affine = None

        # compose the augmentation/B0 warp with the vxm-computed affine warp transform
        # this is really just the affine alignment of the distorted volumes, but avoids
        # the multiple resamplings (synth->aug->B0 dist->target im)

        # the true affine transforms go from one augmented/distorted image to the other
        a_a1_to_s = vxm.layers.InvertAffine(name='inv_aug1')(a_s_to_a1)
        a_a2_to_s = vxm.layers.InvertAffine(name='inv_aug2')(a_s_to_a2)
        a_a1_to_a2 = vxm.layers.ComposeTransform(name='true_aff1')([a_a1_to_s, a_s_to_a2])
        a_a2_to_a1 = vxm.layers.ComposeTransform(name='true_aff2')([a_a2_to_s, a_s_to_a1])

        # now create midspace transforms
        a_a1_to_m1 = vxms.layers.MidspaceTransform(name='half1_true')(a_a1_to_a2)
        a_a2_to_m2 = vxms.layers.MidspaceTransform(name='half2_true')(a_a2_to_a1)
        a_m1_to_a1 = vxm.layers.InvertAffine(name='half1_inv')(a_a1_to_m1)
        a_m2_to_a2 = vxm.layers.InvertAffine(name='half2_inv')(a_a2_to_m2)

        i_a1_to_m1 = vxm.layers.SpatialTransformer(
            name='im1_midspace', interp_method='linear',
            fill_value=fill_value)([i_a1, a_a1_to_m1])
        i_a2_to_m2 = vxm.layers.SpatialTransformer(
            name='im2_midspace', interp_method='linear',
            fill_value=fill_value)([i_a2, a_a2_to_m2])
        i_a1_to_a2 = vxm.layers.SpatialTransformer(
            name='im2_to_a1', interp_method='linear',
            fill_value=fill_value)([i_a1, a_a1_to_a2])
        if estimate_affine:
            a_a1_to_a2_est = vxm_affine.references.affine
            a_a2_to_a1_est = vxm_affine.references.inv_affine
        else:    # disable the affine and use true transforms instead
            a_a1_to_a2_est = a_a1_to_a2
            a_a2_to_a1_est = a_a2_to_a1

        # transforms from distorted images to the midspace between them
        a_a1_to_m1_est = vxms.layers.MidspaceTransform(name='half1_est')(a_a1_to_a2_est)
        a_a2_to_m2_est = vxms.layers.MidspaceTransform(name='half2_est')(a_a1_to_a2_est)
        a_m1_to_a1_est = vxm.layers.InvertAffine(name='half1_inv_est')(a_a1_to_m1_est)
        a_m2_to_a2_est = vxm.layers.InvertAffine(name='half2_inv_est')(a_a2_to_m2_est)

        # these transforms go from synth image to midspace image
        w_s_to_m1 = vxm.layers.ComposeTransform(name='w_s_to_m1')([w_s_to_d1, a_a1_to_m1])
        w_s_to_m2 = vxm.layers.ComposeTransform(name='w_s_to_m2')([w_s_to_d2, a_a2_to_m2])

        # map images to each other
        i_d2_to_d1_est = vxm.layers.SpatialTransformer(
            name='i_d2_to_d1_est', interp_method='linear',
            fill_value=fill_value)([i_d2, a_a2_to_a1_est])
        i_d1_to_d2_est = vxm.layers.SpatialTransformer(
            name='i_d1_to_d2_est', interp_method='linear',
            fill_value=fill_value)([i_d1, a_a1_to_a2_est])

        # map images to midspace
        i_m1 = vxm.layers.SpatialTransformer(
            name='im1_midspace', interp_method='linear',
            fill_value=fill_value)([synth_image1, w_s_to_m1])
        i_m2 = vxm.layers.SpatialTransformer(
            name='im2_midspace', interp_method='linear',
            fill_value=fill_value)([synth_image2, w_s_to_m2])
        i_m2_na = vxm.layers.SpatialTransformer(
            name='im2_midspace_no_atrophy', interp_method='linear',
            fill_value=fill_value)([synth_image2_no_atrophy, w_s_to_m2])
        b_m1 = vxm.layers.SpatialTransformer(
            name='b_m1', interp_method='linear',
            fill_value=fill_value)([brain_mask, w_s_to_m1])
        b_m2 = vxm.layers.SpatialTransformer(
            name='b_m2', interp_method='linear',
            fill_value=fill_value)([brain_mask_wa, w_s_to_m2])
        b_m2_na = vxm.layers.SpatialTransformer(
            name='b_m2_na', interp_method='linear',
            fill_value=fill_value)([brain_mask, w_s_to_m2])

        # now compute dense transforms that align the midspace images
        # and decompose them into estimates of the individual B0 warps
        # using two unets. The first estimates the B0 warps and the second
        # estimates the midspace alignment. They will be linked via a loss
        inshape_vox = np.array(inshape).prod()
        B0_dir1_flat = KL.Flatten()(B0_dir1)
        B0_dir2_flat = KL.Flatten()(B0_dir2)
        B0_dir1_dense = KL.Dense(inshape_vox, name='B0_dir1_dense')(B0_dir1_flat)
        B0_dir2_dense = KL.Dense(inshape_vox, name='B0_dir2_dense')(B0_dir2_flat)
        B0_dir1_im = KL.Reshape(tuple(inshape) + (1,), name='B0_dir1_im')(B0_dir1_dense)
        B0_dir2_im = KL.Reshape(tuple(inshape) + (1,), name='B0_dir2_im')(B0_dir2_dense)

        input_list1_B0 = [i_d1, i_d2_to_d1_est]   # distorted images for estimating B0 warps
        input_list2_B0 = [i_d2, i_d1_to_d2_est]   # distorted images for estimating B0 warps
        input_list_mid = [i_m1, i_m2]   # midspace for estimating nl registration
        unet1_inputs_B0 = KL.Concatenate(axis=-1, name='unet1_inputs_B0')(input_list1_B0)
        unet2_inputs_B0 = KL.Concatenate(axis=-1, name='unet2_inputs_B0')(input_list2_B0)
        unet_inputs_mid = KL.Concatenate(axis=-1, name='unet_inputs_mid')(input_list_mid)
        num_outputs = 1 if one_d else ndims
        dense_model1 = ne.models.unet(
            unet_features,
            tuple(inshape) + (len(input_list1_B0),),
            unet_levels,
            3,              # conv size
            num_outputs,              # nb_labels/nb_outputs
            nb_conv_per_level=nb_unet_conv_per_level,
            name='B0_correction_unet1',
            final_pred_activation='linear',
        )
        dense_model2 = ne.models.unet(
            unet_features,
            tuple(inshape) + (len(input_list2_B0),),
            unet_levels,
            3,              # conv size
            num_outputs,              # nb_labels/nb_outputs
            nb_conv_per_level=nb_unet_conv_per_level,
            name='B0_correction_unet2',
            final_pred_activation='linear',
        )
        dense_model_mid = ne.models.unet(
            unet_features,
            tuple(inshape) + (len(input_list_mid),),
            unet_levels,
            3,              # conv size
            ndims,              # nb_labels/nb_outputs
            nb_conv_per_level=nb_unet_conv_per_level,
            name='dense_model_mid',
            final_pred_activation='linear',
        )

        # this code works if the unet outputs a ndim-vec and we don't constrain
        # the direction to be in the B0_Dir
        unet_outputs1 = dense_model1(unet1_inputs_B0)   # this should be an estimate of B0 map1
        unet_outputs2 = dense_model2(unet2_inputs_B0)   # this should be an estimate of B0 map2
        unet_outputs_mid = dense_model_mid(unet_inputs_mid)   # this should be an estimate of warp
        if one_d is True:
            print('creating 1D vector fields')
            B0_map1_est = KL.Lambda(lambda x: x[..., 0:1], name='B0_map1_est')(unet_outputs1)
            B0_map2_est = KL.Lambda(lambda x: x[..., 0:1], name='B0_map2_est')(unet_outputs2)
            mul_dir = lambda x: x[0] * x[1][:, tf.newaxis, ...]  # scale B0_dir vec field
            warp1_svf = KL.Lambda(mul_dir, name='B0_warp1_svf_est')([B0_map1_est, B0_dir1])
            warp2_svf = KL.Lambda(mul_dir, name='B0_warp2_svf_est')([B0_map2_est, B0_dir2])
        else:
            warp1_svf = KL.Lambda(lambda x: x[..., 0:ndims], name='B0_map1_est')(unet_outputs1)
            warp2_svf = KL.Lambda(lambda x: x[..., 0:ndims], name='B0_map2_est')(unet_outputs2)

        # these warps go from one midspace image to the other midspace image
        w_m1_to_m2_est = vxm.layers.VecInt(int_steps=5, name='mid_warp1to2')(unet_outputs_mid)
        warp_mid_neg_svf = ne.layers.Negate()(unet_outputs_mid)
        w_m2_to_m1_est = vxm.layers.VecInt(int_steps=5, name='mid_warp2to1')(warp_mid_neg_svf)

        B0_warp1_fwd_est = vxm.layers.VecInt(int_steps=5, name='B0_warp1_fwd_est')(warp1_svf)
        warp1_neg_svf = ne.layers.Negate()(warp1_svf)
        B0_warp1_inv_est = vxm.layers.VecInt(int_steps=5, name='B0_warp1_inv_est')(warp1_neg_svf)
        B0_warp2_fwd_est = vxm.layers.VecInt(int_steps=5, name='B0_warp2_fwd_est')(warp2_svf)
        warp2_neg_svf = ne.layers.Negate()(warp2_svf)
        B0_warp2_inv_est = vxm.layers.VecInt(int_steps=5, name='B0_warp2_inv_est')(warp2_neg_svf)

        # create outputs for losses

        # build true warps that go from midspace im1->im2 and visa-versa
        w_m1_to_m2 = vxm.layers.ComposeTransform(name='warp_mid1_to_2_true')(
            [a_m1_to_a1, B0_warp1_inv, a_a1_to_s, a_s_to_a2, B0_warp2, a_a2_to_m2])
        w_m2_to_m1 = vxm.layers.ComposeTransform(name='warp_mid2_to_1_true')(
            [a_m2_to_a2, B0_warp2_inv, a_a2_to_s, a_s_to_a1, B0_warp1, a_a1_to_m1])

        # warps from synth_image1 -> distorted image 2 and visa-versa
        w_s_to_m2_est = vxm.layers.ComposeTransform(name='warp_im1_to_2_est')(
            [a_s_to_a1, B0_warp1_fwd_est, a_a1_to_m1, w_m1_to_m2_est])
        w_s_to_m1_est = vxm.layers.ComposeTransform(name='warp_im2_to_1_est')(
            [a_s_to_a2, B0_warp2_fwd_est, a_a2_to_m2, w_m2_to_m1_est])
        im1_to_im2_est = vxm.layers.SpatialTransformer(
            name='im1_to_im2_est', interp_method='linear', fill_value=fill_value)(
                [synth_image1, w_s_to_m2_est])
        im2_to_im1_est = vxm.layers.SpatialTransformer(
            name='im2_to_im1_est', interp_method='linear', fill_value=fill_value)(
                [synth_image2, w_s_to_m1_est])
        im2_to_im1_no_atrophy_est = vxm.layers.SpatialTransformer(
            name='im2_no_atrophy_to_im1_est', interp_method='linear', fill_value=fill_value)(
                [synth_image2_no_atrophy, w_s_to_m1_est])

        im1_reg_est = KL.Concatenate(axis=-1, name='im1_reg_est')(
            [i_m2_na, im1_to_im2_est, b_m2])
        im2_reg_est = KL.Concatenate(axis=-1, name='im2_reg_est')(
            [i_m1, im2_to_im1_no_atrophy_est, b_m1])

        i_d1_est = vxm.layers.SpatialTransformer(
            name='i_d1_est', interp_method='linear',
            fill_value=fill_value)([i_a1, B0_warp1_fwd_est])
        i_d2_est = vxm.layers.SpatialTransformer(
            name='i_d2_est', interp_method='linear',
            fill_value=fill_value)([i_a2_na, B0_warp2_fwd_est])
        im1_dist_outputs = [i_d1, i_d1_est, b_d1]
        im2_dist_outputs = [i_d2_na, i_d2_est, b_d2_na]
        im1_dist_out = KL.Concatenate(axis=-1, name='im1_dist_l')(im1_dist_outputs)
        im2_dist_out = KL.Concatenate(axis=-1, name='im2_dist_l')(im2_dist_outputs)

        # have to concat the labels so they have the same dim as the warp
        B0_w1_out = [B0_warp1, B0_warp1_fwd_est, b_d1]
        B0_w2_out = [B0_warp2, B0_warp2_fwd_est, b_d2]
        B0_warp1_out = KL.Concatenate(axis=-1, name='B0_warp1_l')(B0_w1_out)
        B0_warp2_out = KL.Concatenate(axis=-1, name='B0_warp2_l')(B0_w2_out)

        # [..., 0:2] is real warp 2:4 is est warp and [4] is labels for masking
        mw1_out = [w_m1_to_m2, w_m1_to_m2_est, b_m1]
        mw2_out = [w_m2_to_m1, w_m2_to_m1_est, b_m2]
        midspace_warp1_out = KL.Concatenate(axis=-1, name='mid_warp1')(mw1_out)
        midspace_warp2_out = KL.Concatenate(axis=-1, name='mid_warp2')(mw2_out)

        image_outputs = [im1_reg_est, im2_reg_est, im1_dist_out, im2_dist_out]
        warp_outputs = [B0_warp1_out, B0_warp2_out, midspace_warp1_out, midspace_warp2_out]
        smoothness_outputs = [w_m1_to_m2, w_m2_to_m1]
        outputs_nl = image_outputs + warp_outputs + smoothness_outputs

        # map brainmask to estimated affine locations
        b_a1_to_a2_est = vxm.layers.SpatialTransformer(
            name='b_a1_to_a2_est', interp_method='linear',
            fill_value=fill_value)([b_a1, a_a1_to_a2_est])
        b_a2_to_a1_est = vxm.layers.SpatialTransformer(
            name='b_a2_to_a1_est', interp_method='linear',
            fill_value=fill_value)([b_a2, a_a2_to_a1_est])

        # build estimated affines from synth to avoid multiple interpolations
        a_s_to_a2_est = vxm.layers.ComposeTransform(name='a_s_to_a2_est')(
            [a_s_to_a1, a_a1_to_a2_est])
        a_s_to_a1_est = vxm.layers.ComposeTransform(name='a_s_to_a1_est')(
            [a_s_to_a2, a_a2_to_a1_est])
        i_a1_to_a2_est = vxm.layers.SpatialTransformer(name='i_a1_to_a2_est', fill_value=0)(
            [synth_image1, a_s_to_a2_est])
        i_a2na_to_a1_est = vxm.layers.SpatialTransformer(name='i_a2na_to_a1_est', fill_value=0)(
            [synth_image2_no_atrophy, a_s_to_a1_est])
        if debug:  # make sure some intermediate stuff is in the model
            i_a2_to_a1_est = vxm.layers.SpatialTransformer(name='i_a2_to_a1_est', fill_value=0)(
                [i_a2, a_a2_to_a1_est])
            outputs_nl += [i_d1, i_d2, i_d2_na, i_a2_to_a1_est,
                           i_d1_est, i_d2_est, B0_warp1_inv_est, B0_warp2_inv_est]

        aout1 = [i_a2_na, i_a1_to_a2_est, b_a2_na, b_a1_to_a2_est]
        aout2 = [i_a1, i_a2na_to_a1_est, b_a1, b_a2_to_a1_est]
        affine_out1 = KL.Concatenate(axis=-1, name='affine_out1')(aout1)
        affine_out2 = KL.Concatenate(axis=-1, name='affine_out2')(aout2)
        # affine_out1 = KL.Concatenate(axis=-1, name='affine_out1')([i_a2_na, i_a1_to_a2_est])
        # affine_out2 = KL.Concatenate(axis=-1, name='affine_out2')([i_a1, i_a2na_to_a1_est])
        self.references = ne.tf.modelio.LoadableModel.ReferenceContainer()
        self.references.dense_model = dense_model2
        self.references.affine_out1 = affine_out1
        self.references.affine_out2 = affine_out2
        self.references.vxm_affine = vxm_affine

        super().__init__(name=name, inputs=label_input, outputs=outputs_nl)


###############################################################################
# SynthStrip models
###############################################################################

class SynthStripWithWarp(ne.tf.modelio.LoadableModel):
    """
    SynthStrip model for learning subject-to-subject registration from images
    with arbitrary contrasts synthesized from label maps.

    Author: brf2
    """

    @ne.modelio.store_config_args
    def __init__(self, inshape, labels_in, labels_out,
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 src_feats=1,
                 output_onehot=False,
                 gen_args={}):
        """
        Parameters:
            inshape: Input shape, e.g. (160, 160, 192).
            labels_in: List of all labels included in the training segmentations.
            labels_out: List of labels to encode in the output one-hot maps.
            gen_args: Keyword arguments passed to the internal generative model.
        """

        ndims = len(inshape)
        assert ndims in [1, 2, 3], \
            'ndims should be one of 1, 2, or 3. found: %d' % ndims

        inshape = tuple(inshape)

        # take input label map and synthesize an image from it
        # (was vxm.networks.SynthMorphGenerative)
        gen_model = ne.models.labels_to_image(
            inshape,
            labels_in,
            labels_out,
            id=0,
            return_def=False,
            one_hot=False,
            **gen_args
        )

        synth_image, synth_labels = gen_model.outputs

        vxm_dense = vxm.networks.VxmDense(inshape)
        warp_model = tf.keras.models.Model(vxm_dense.inputs, vxm_dense.references.pos_flow)
        warp = warp_model([synth_image, synth_image])

        warped_mesh = KL.Lambda(lambda x: tf.map_fn(
            vxms.utils.transform_grid, x, fn_output_signature=tf.float32))(warp)

        # build a unet to apply to the synthetic image and strip it
        unet_model = ne.models.unet(
            nb_unet_features,
            tuple(inshape) + (4,),
            nb_unet_levels,
            ndims * (3,),
            1,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            final_pred_activation='linear'
        )

        unet_input = KL.Concatenate(name='unet_input', axis=-1)([synth_image, warped_mesh])
        unet_outputs = unet_model(unet_input)

        # output the prob of brain and the warped label map
        # so that the loss function can compute brain/nonbrain
        stacked_output = KL.Concatenate(axis=-1, name='image_label')(
            [unet_outputs, tf.cast(synth_labels, tf.float32)])

        super().__init__(inputs=gen_model.inputs, outputs=stacked_output)

        # cache pointers to important layers and tensors for future reference
        self.references = ne.tf.modelio.LoadableModel.ReferenceContainer()
        self.references.unet = unet_model     # this is the stripping net
        self.references.gen_model = gen_model
        self.references.synth_image = synth_image

        # construct stripping model
        input_image = KL.Input(shape=tuple(inshape) + (1,), name='strip_input')
        image_warp = warp_model([input_image, input_image])
        image_mesh = KL.Lambda(lambda x: tf.transpose(
            ne.utils.volshape_to_ndgrid(x.shape[1:-1]), (1, 2, 3, 0)))(input_image)
        warped_image_mesh = KL.Lambda(lambda x: tf.map_fn(
            vxms.utils.transform_grid, x, fn_output_signature=tf.float32))(image_warp)
        unet_image_input = KL.Concatenate(name='unet_input', axis=-1)(
            [input_image, warped_image_mesh])
        unet_image_outputs = unet_model(unet_image_input)
        self.references.strip_model = tf.keras.models.Model(
            input_image, unet_image_outputs)

    def get_strip_model(self):
        return self.references.strip_model


###############################################################################
# Template Creation Networks
###############################################################################


class TemplateCreationSemiSupervised(ne.tf.modelio.LoadableModel):
    """
    VoxelMorph network to generate an unconditional template image.
    """

    @ne.modelio.store_config_args
    def __init__(self, inshape,
                 nclasses=32,
                 mean_cap=100,
                 atlas_feats=1,
                 src_feats=1,
                 feature_detector=False,
                 **kwargs):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nclasses: the number of classes in the atlas to learn
            atlas_feats: Number of atlas/template features. Default is 1.
            src_feats: Number of source image features. Default is 1.
            kwargs: Forwarded to the internal TemplateCreation model.
        """

        # build base templatecreation model
        model = vxm.networks.TemplateCreation(
            inshape, atlas_feats=atlas_feats, src_feats=src_feats, **kwargs)
        model.output_names = ['atlas_to_sub', 'sub_to_atlas', 'mean_warp_atlas_to_sub',
                              'warp_atlas_to_sub']

        # add an atlas layer
        init = KI.RandomNormal(mean=0.0, stddev=1e-7)
        atlas_seg_layer = ne.layers.LocalParamWithInput(name='parc_atlas',
                                                        shape=(*inshape, nclasses),
                                                        mult=1.0,
                                                        initializer=init)
        atlas_seg_tensor = atlas_seg_layer(model.inputs[0])

        # map the atlas to subject space for the loss function
        stin = [atlas_seg_tensor, model.references.pos_flow]
        atlas_seg_in_sub_lin = vxm.layers.SpatialTransformer(interp_method='linear',
                                                             name='atlas_seg_in_sub_lin',
                                                             fill_value=None)(stin)
        atlas_seg_in_sub = KL.Activation('softmax', name='atlas_in_sub_seg')(atlas_seg_in_sub_lin)
        atlas_mean_in_sub = KL.Lambda(lambda x: x, name='atlas_mean_in_sub')(model.outputs[0])
        sub_in_atlas = KL.Lambda(lambda x: x, name='sub_in_atlas')(model.outputs[1])
        atlas_mean = KL.Lambda(lambda x: x, name='atlas_mean')(model.outputs[2])
        atlas_warp = KL.Lambda(lambda x: x, name='atlas_to_sub_warp')(model.outputs[3])
        # model.outputs[0:2] + [model.outputs[3], atlas_in_subject_space])
        linear_outputs = [
            atlas_mean_in_sub,
            sub_in_atlas,
            atlas_seg_in_sub_lin,
            atlas_warp,
        ]
        model_linear = tf.keras.Model(model.inputs, linear_outputs)

        # initialize the keras model. Use same outputs as before but swap in
        # softmax output instead of linear ones
        model_outputs = model_linear.outputs[0:2] + [
            atlas_seg_in_sub,
            model_linear.outputs[-1]
        ]
        super().__init__(inputs=model_linear.inputs, outputs=model_outputs)

        # cache pointers to important layers and tensors for future reference
        self.references = model.references
        self.references.atlas_seg_layer = atlas_seg_layer
        self.references.atlas_seg_tensor = atlas_seg_tensor
        self.references.template_model = model
        self.references.model_linear = model_linear
        self.references.mean_stream = model.outputs[2]

    def set_seg_atlas(self, atlas):
        return self.set_atlas_seg(atlas)

    def set_atlas_seg(self, atlas):
        """
        Sets the atlas weights.
        """
        if atlas.shape[1] == -1:
            atlas = np.reshape(atlas, atlas.shape[1:])
        self.references.atlas_seg_layer.set_weights([atlas])

    def set_atlas(self, atlas):
        return self.references.template_model.set_atlas(atlas)

    def get_atlas(self):
        return self.references.template_model.get_atlas()

    def get_seg_atlas(self):
        """
        gets the atlas segmentation weights.
        """
        #        return self.references.atlas_seg_layer.get_weights()[0].squeeze()
        layer = model_semi.get_layer(model_semi.references.atlas_seg_layer.name.split('/')[0])
        return layers.weights[0]

    def get_registration_model(self):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        return tf.keras.Model(self.inputs, self.references.pos_flow)

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
        atrf = [img_input, warp_model.output]
        y_img = layers.SpatialTransformer(
            interp_method=interp_method, fill_value=fill_value)(atrf)
        prin = [src, trg, img]
        return tf.keras.Model(
            inputs=(*warp_model.inputs, img_input), outputs=y_img).predict(prin)


class TemplateCreationAffineSemiSupervised(ne.tf.modelio.LoadableModel):
    """
    VoxelMorph network to generate an unconditional template image.
    """

    @ne.modelio.store_config_args
    def __init__(self, inshape,
                 nclasses=32,
                 mean_cap=100,
                 atlas_feats=1,
                 src_feats=1,
                 fill_value=None,
                 softmax=False,
                 **kwargs):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nclasses: the number of classes in the atlas to learn
            atlas_feats: Number of atlas/template features. Default is 1.
            src_feats: Number of source image features. Default is 1.
            kwargs: Forwarded to the internal TemplateCreation model.
        """

        # build base templatecreation model
        model = TemplateCreationAffine(
            inshape, atlas_feats=atlas_feats, src_feats=src_feats, fill_value=fill_value, **kwargs)
        model.output_names = ['atlas_to_sub', 'sub_to_atlas', 'atlas']

        # add an atlas layer
        init = KI.RandomNormal(mean=0.0, stddev=1e-7)
        atlas_seg_layer = ne.layers.LocalParamWithInput(name='parc_atlas',
                                                        shape=(*inshape, nclasses),
                                                        mult=1.0,
                                                        initializer=init)
        atlas_seg_tensor = atlas_seg_layer(model.inputs[0])

        # map the atlas to subject space for the loss function
        stin = [atlas_seg_tensor, model.references.pos_flow]
        atlas_in_subject_space = vxm.layers.SpatialTransformer(interp_method='linear',
                                                               name='atlas_in_sub',
                                                               fill_value=fill_value)(stin)

        if softmax:
            parc_atlas_out = KL.Activation('softmax', name='atlas_out')(atlas_in_subject_space)
        else:
            parc_atlas_out = atlas_in_subject_space

        model_semi_linear = tf.keras.Model(
            [model.inputs], model.outputs[0:2] + [atlas_in_subject_space])

        outputs = model_semi_linear.outputs[0:2] + [parc_atlas_out]
        if len(model.outputs) == 4:  # it includes a nonlinear component so return warp
            outputs += [model.outputs[-1]]

        # initialize the keras model
        super().__init__(inputs=model_semi_linear.inputs, outputs=outputs)

        # cache pointers to important layers and tensors for future reference
        self.references = model.references
        self.references.atlas_seg_layer = atlas_seg_layer
        self.references.atlas_seg_tensor = atlas_seg_tensor
        self.references.template_model = model
        self.references.model_linear = model_semi_linear

    def get_seg_atlas(self, softmax=False):
        """
        Sets the atlas weights.
        """
        if softmax:
            parc_atlas_out = KL.Activation('softmax', name='atlas_out')(
                self.references.atlas_seg_layer.get_weights())
            return parc_atlas_out[0].squeeze()
        else:
            return self.references.atlas_seg_layer.get_weights()[0].squeeze()

    def set_atlas__seg(self, atlas):
        return self.set_seg_atlas(atlas)

    def get_atlas__seg(self, softmax=False):
        return self.get_seg_atlas(softmax=softmax)

    def set_seg_atlas(self, atlas):
        """
        Sets the atlas weights.
        """
        # if atlas.shape[1]:
        # atlas = np.reshape(atlas, atlas.shape[1:])
        self.references.atlas_seg_layer.set_weights([atlas])

    def set_atlas(self, atlas):
        return self.references.template_model.set_atlas(atlas)

    def get_atlas(self):
        return self.references.template_model.get_atlas()

    def get_registration_model(self):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        return tf.keras.Model(self.inputs, self.references.pos_flow)

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
        atrf = [img_input, warp_model.output]
        y_img = layers.SpatialTransformer(
            interp_method=interp_method, fill_value=fill_value)(atrf)
        prin = [src, trg, img]
        return tf.keras.Model(
            inputs=(*warp_model.inputs, img_input), outputs=y_img).predict(prin)


class TemplateCreationAffine(ne.modelio.LoadableModel):
    """
    VoxelMorph network to generate an unconditional template image.
    """

    @ne.modelio.store_config_args
    def __init__(self, inshape, nb_unet_features=None, mean_cap=100, atlas_feats=1, src_feats=1,
                 fill_value=None, feature_detector=False, dense_combo=False, **kwargs):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features.
                See VxmDense documentation for more information.
            mean_cap: Cap for mean stream. Default is 100.
            atlas_feats: Number of atlas/template features. Default is 1.
            src_feats: Number of source image features. Default is 1.
            kwargs: Forwarded to the internal VxmDense model.
        """

        # configure inputs
        source_input = tf.keras.Input(shape=[*inshape, src_feats], name='source_input')

        # pre-warp (atlas) model
        atlas_layer = ne.layers.LocalParamWithInput(
            shape=(*inshape, atlas_feats),
            mult=1.0,
            initializer=KI.RandomNormal(mean=0.0, stddev=1e-7),
            name='atlas'
        )
        atlas_tensor = atlas_layer(source_input)
        warp_input_model = tf.keras.Model(inputs=[source_input], outputs=[
                                          atlas_tensor, source_input])

        # warp model
        if feature_detector:
            if dense_combo:
                vxm_model = VxmJointAverage(inshape, input_model=warp_input_model, bidir=True, 
                                            **kwargs)
                neg_flow = vxm_model.outputs[1]
            else:
                vxm_model = VxmAffineFeatureDetector(inshape, input_model=warp_input_model)
                vxm_model.references.affine = pos_flow  # for use by caller
                neg_flow = vxm_model.references.inv_transform

            pos_flow = vxm_model.outputs[0]
            atlas_mapped_to_subject = vxm.layers.SpatialTransformer(
                name='atlas_to_sub', fill_value=fill_value)(
                    [atlas_tensor, pos_flow])
            subject_mapped_to_atlas = vxm.layers.SpatialTransformer(
                name='sub_to_atlas', fill_value=fill_value)(
                    [source_input, neg_flow])

        else:
            vxm_model = VxmAffineEncoderThenDense(inshape, nb_unet_features, bidir=True,
                                                  input_model=warp_input_model, **kwargs)

            # extract tensors from stacked model
            atlas_mapped_to_subject = KL.Lambda(lambda x: x, name='atlas_to_sub')(
                vxm_model.outputs[0])
            subject_mapped_to_atlas = KL.Lambda(lambda x: x, name='sub_to_atlas')(
                vxm_model.outputs[1])
            pos_flow = vxm_model.references.pos_flow
            neg_flow = vxm_model.references.neg_flow

        # get mean stream of negative flow
        mean_stream = ne.layers.MeanStream(name='mean_stream', cap=mean_cap)(neg_flow)

        outputs = [atlas_mapped_to_subject, subject_mapped_to_atlas, mean_stream]
        if dense_combo:
            outputs += [pos_flow]

        # initialize the keras model
        super().__init__(inputs=[source_input], outputs=outputs)

        # cache pointers to important layers and tensors for future reference
        self.references = ne.modelio.LoadableModel.ReferenceContainer()
        self.references.atlas_layer = atlas_layer
        self.references.mean_stream = mean_stream
        self.references.atlas_tensor = atlas_tensor
        self.references.vxm_model = vxm_model
        self.references.pos_flow = pos_flow
        self.references.neg_flow = neg_flow
        self.references.atlas_to_subject_warp = pos_flow
        self.references.subject_to_atlas_warp = neg_flow

    def set_atlas(self, atlas):
        """
        Sets the atlas weights.
        """
        if atlas.shape[1]:
            atlas = np.reshape(atlas, atlas.shape[1:])
        self.references.atlas_layer.set_weights([atlas])

    def get_atlas(self):
        """
        Sets the atlas weights.
        """
        return self.references.atlas_layer.get_weights()[0].squeeze()

    def get_registration_model(self):
        """
        Returns a reconfigured model to predict only the final transform.
        """
        return tf.keras.Model(self.inputs, self.references.pos_flow)

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
        y_img = layers.SpatialTransformer(interp_method=interp_method,
                                          fill_value=fill_value)([img_input, warp_model.output])
        inputs = (*warp_model.inputs, img_input)
        return tf.keras.Model(inputs=inputs, outputs=y_img).predict([src, trg, img])


class VxmDenseWithPositionalEncoding(ne.modelio.LoadableModel):
    """
    VoxelMorph network with positional encoding.
    """
    @ne.modelio.store_config_args
    def __init__(self, inshape,
                 input_model=None,
                 src_feats=1,
                 trg_feats=1,
                 npos=None,
                 pad_size=0,
                 bidir=False,
                 name='vxm_dense_pe',
                 **kwargs):

        if input_model is None:
            source = tf.keras.Input(shape=(*inshape, src_feats), name='%s_source_input' % name)
            target = tf.keras.Input(shape=(*inshape, trg_feats), name='%s_target_input' % name)
            input_model = tf.keras.Model(inputs=[source, target], outputs=[source, target])
        else:
            source, target = input_model.outputs[:2]

        if npos is not None and npos > 0:
            source_pe = nes.layers.ConcatWithPositionalEncoding(npos, pad_size)(source)
            target_pe = nes.layers.ConcatWithPositionalEncoding(npos, pad_size)(target)
            input_model = tf.keras.Model(inputs=input_model.inputs, outputs=[source_pe, target_pe])

        vxm_model = vxm.networks.VxmDense(inshape, input_model=input_model, bidir=bidir, **kwargs)

        pos_flow = vxm_model.references.pos_flow
        y_source = vxm.layers.SpatialTransformer(interp_method='linear',
                                                 name=f'pos_transformer')([source, pos_flow])
        outputs = [y_source]

        if bidir:
            neg_flow = vxm_model.references.neg_flow
            y_target = vxm.layers.SpatialTransformer(interp_method='linear',
                                                     name=f'neg_transformer')([target, neg_flow])
            outputs.append(y_target)

        super().__init__(inputs=input_model.inputs, outputs=outputs)

        # cache pointers to important layers and tensors for future reference
        self.references = ne.modelio.LoadableModel.ReferenceContainer()
        self.references.vxm_model = vxm_model
        self.references.unet_model = vxm_model.references.unet_model
        self.references.source = source
        self.references.target = target
        self.references.svf = vxm_model.references.svf
        self.references.preint_flow = vxm_model.references.preint_flow
        self.references.postint_flow = vxm_model.references.postint_flow
        self.references.pos_flow = vxm_model.references.pos_flow
        self.references.neg_flow = vxm_model.references.neg_flow
        self.references.y_source = y_source
        self.references.y_target = y_target
        self.references.hyp_input = vxm_model.references.hyp_input


class VxmAffineFeatureDetector(AbstractVxmModel):
    """
    Symmetric registration networks that detects learned features separately in
    in the input images and fits a linear transform between their barycenters.

    Author:
        mu40

    If you find this function useful, please consider citing:
        M Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca
        SynthMorph: learning contrast-invariant registration without acquired images
        IEEE Transactions on Medical Imaging (TMI), 41 (3), 543-558, 2022
        https://doi.org/10.1109/TMI.2021.3116879
    """

    @ne.modelio.store_config_args
    def __init__(self,
                 in_shape=None,
                 num_chan=1,
                 num_feat=64,
                 enc_nf=(256,) * 4,
                 dec_nf=(256,) * 0,
                 add_nf=(256,) * 4,
                 per_level=1,
                 dropout=0,
                 half_res=True,
                 weighted=True,
                 rigid=False,
                 name='vxm_affine',
                 input_model=None):
        """
        Parameters:
            in_shape: Spatial input shape, for example `(192,) * 3` in 3D.
            num_chan: Number of input image channels.
            num_feat: Number of features derived from each input image.
            enc_nf: List of filter numbers for the downsampling path.
            dec_nf: List of filter numbers for the upsampling path.
            add_nf: List of filter numbers for additional convolutions.
            per_level: Number of times we apply each convolutional filter.
            dropout: Spatial dropout rate applied during training.
            half_res: Omit the highest-level convolution to improve efficiency.
            weighted: Use a weighted instead of ordinary least squares.
            rigid: Estimate a rigid instead of a full affine transform.
            name: Model name.
            input_model Input model replacing the default input layers.
        """
        # Inputs.
        if input_model is None:
            source = tf.keras.Input(shape=(*in_shape, num_chan))
            target = tf.keras.Input(shape=(*in_shape, num_chan))
            input_model = tf.keras.Model(*[(source, target)] * 2)
        source, target = input_model.outputs[:2]

        # Dimensionality.
        in_shape = np.asarray(source.shape[1:-1])
        num_dim = len(in_shape)
        assert num_dim in (2, 3), 'only 2D and 3D supported'

        # Layers.
        conv = getattr(KL, f'Conv{num_dim}D')
        pool = getattr(KL, f'MaxPool{num_dim}D')
        drop = getattr(KL, f'SpatialDropout{num_dim}D')
        up = getattr(KL, f'UpSampling{num_dim}D')
        prop = dict(kernel_size=3, padding='same')

        # Internal feature detector.
        inp = tf.keras.Input(shape=(*in_shape, num_chan))
        x = pool()(inp) if half_res else inp

        # Encoder.
        enc = []
        for n in enc_nf:
            for _ in range(per_level):
                x = conv(n, **prop)(x)
                x = drop(dropout)(x)
                x = KL.LeakyReLU(0.2)(x)
            enc.append(x)
            x = pool()(x)

        # Decoder.
        for n in dec_nf:
            for _ in range(per_level):
                x = conv(n, **prop)(x)
                x = drop(dropout)(x)
                x = KL.LeakyReLU(0.2)(x)
            x = KL.concatenate([up()(x), enc.pop()])

        # Additional convolutions.
        for n in add_nf:
            x = conv(n, **prop)(x)
            x = drop(dropout)(x)
            x = KL.LeakyReLU(0.2)(x)

        # Features. Cannot currently run XLA with 'relu', so use a custom ReLU.
        # Always sum and fit affine with at least single precision.
        relu = lambda x: tf.math.maximum(x, 0)
        x = conv(num_feat, activation=relu, **prop)(x)
        if tf.keras.mixed_precision.global_policy().compute_dtype == 'float16':
            x = tf.cast(x, tf.float32)
        net = tf.keras.Model(inp, outputs=x)
        feat_1 = net(source)
        feat_2 = net(target)

        # Barycenters.
        prop = dict(axes=range(1, num_dim + 1), normalize=True, shift_center=True)
        cen_1 = ne.utils.barycenter(feat_1, **prop) * in_shape
        cen_2 = ne.utils.barycenter(feat_2, **prop) * in_shape

        # Weights.
        axes = range(1, num_dim + 1)
        pow_1 = tf.reduce_sum(feat_1, axis=axes)
        pow_2 = tf.reduce_sum(feat_2, axis=axes)
        pow_1 /= tf.reduce_sum(pow_1, axis=-1, keepdims=True)
        pow_2 /= tf.reduce_sum(pow_2, axis=-1, keepdims=True)
        weights = pow_1 * pow_2

        # Least-squares fit.
        aff = vxm.utils.fit_affine(cen_1, cen_2, weights=weights if weighted else None)
        if rigid:
            aff = vxm.utils.affine_matrix_to_params(aff)
            aff = aff[:, :num_dim * (num_dim + 1) // 2]
            aff = vxm.layers.ParamsToAffineMatrix(ndims=num_dim)(aff)
        if tf.keras.mixed_precision.global_policy().compute_dtype == 'float16':
            aff = tf.cast(aff, tf.float16)

        # References.
        super().__init__(input_model.inputs, outputs=aff, name=name)
        self.references = ne.modelio.LoadableModel.ReferenceContainer()
        self.references.transform = aff
        self.references.inv_transform = vxm.layers.InvertAffine()(aff)
        self.references.features = (feat_1, feat_2)


###############################################################################
# Spherical Networks
###############################################################################

class SphereMorphWithAtlasBuilding(ne.modelio.LoadableModel):
    @ne.modelio.store_config_args
    def __init__(self,
                 input_model=None,
                 input_shape=None,
                 num_ft=None,
                 nb_unet_features=None,
                 loss_fn=None,
                 metric_fn=None,
                 metric_name=None,
                 is_var=False,
                 is_bidir=True,
                 pad_size=0,
                 is_atlas_trainable=True,
                 pos_enc=0,
                 name='smab',
                 **kwargs):
        """
        :param input_model: if input a model, the input of this model is the output of input_model
        :param input_shape: the shape of input tensor
        :param num_ft: number of features
        :param nb_unet_features: number of features in the unet, see VxmDense
        :param loss_fn: a tuple of loss functions to be used in loss end layers
        :param metric_fn: a tuple of metric functions to be used in loss end layers
        :param metric_name: a tuple of metric names to be used in loss end layers
        :param is_var: whether to use variance layer
        :param is_bidir: whether to use bidirectional flow
        :param pad_size: padding size
        :param is_atlas_trainable: whether the atlas is trainable
        :param pos_enc: whether to concatenate positional encoding into the input
        :param name: name of the model
        :param kwargs: other arguments for VxmDense

        :return: a model with three scalar outputs (four if bidir)
        :note: the usage of variance layer is not fully tested, need to be cautious when using it
        """

        # config inputs
        if input_model is None:
            if input_shape is None or num_ft is None:
                raise ValueError('input_shape and num_ft must be provided if input_model is None')
            this_input = KL.Input(shape=[*input_shape, num_ft], name='%s_input' % name)
            model_inputs = [this_input]
        else:
            if len(input_model.outputs) == 1:
                this_input = input_model.outputs[0]
            else:
                this_input = KL.concatenate(input_model.outputs, name='%s_input_concat' % name)
            model_inputs = input_model.inputs
            input_shape = this_input.shape[1:-1]
            num_ft = this_input.shape[-1]

        # build atlas mean
        mean_layer = layers.SphericalLocalParamWithInput(
            shape=(*input_shape, num_ft),
            mult=1.0,
            initializer=KI.RandomNormal(mean=0.0, stddev=1e-3),
            pad_size=pad_size,
            trainable=is_atlas_trainable,
            name=f'atlas_mean'
        )
        atlas_mean = mean_layer(this_input)

        # positive warp is subject -> atlas
        vxm_model_input = tf.keras.Model(inputs=model_inputs, outputs=[this_input, atlas_mean])

        # build vxm model depending on if positional encoding is used
        # no harm to always turn on bidir here and get the neg_flow in the vxm model
        if pos_enc > 0:
            vxm_model = vxms.networks.VxmDenseWithPositionalEncoding(
                input_shape, npos=pos_enc, pad_size=pad_size, nb_unet_features=nb_unet_features,
                bidir=True, input_model=vxm_model_input, int_resolution=1, **kwargs)
        else:
            vxm_model = vxm.networks.VxmDense(
                input_shape, nb_unet_features=nb_unet_features,
                bidir=True, input_model=vxm_model_input, int_resolution=1, **kwargs)

        # get positive and negative flows
        pos_flow = vxm_model.references.pos_flow
        neg_flow = vxm_model.references.neg_flow

        warped_subject = vxm_model.references.y_source
        warped_atlas = vxm_model.references.y_target

        # build atlas variance
        if is_var:
            var_layer = nes.layers.VarianceStream(forgetting_factor=0.99, name='atlas_variance')
            atlas_var = var_layer([warped_subject, atlas_mean])
            # transform variance layer back to subject space if bidirectional
            if is_bidir:
                subject_var = vxm.layers.SpatialTransformer(
                    interp_method='linear', name=f'subject_var')([atlas_var, neg_flow])
        # get loss functions
        if is_bidir:
            loss_pos_fn, loss_neg_fn, loss_reg_fn, loss_ms_fn = loss_fn
        else:
            loss_pos_fn, loss_reg_fn, loss_ms_fn = loss_fn

        # get metric functions and names
        if metric_fn is not None:
            if is_bidir:
                metric_pos, metric_neg, metric_reg, metric_ms = metric_fn
                mn_pos, mn_neg, mn_reg, mn_ms = metric_name
            else:
                metric_pos, metric_reg, metric_ms = metric_fn
                mn_pos, mn_reg, mn_ms = metric_name
        else:
            metric_pos, metric_neg, metric_reg, metric_ms = None, None, None, None
            mn_pos, mn_neg, mn_reg, mn_ms = None, None, None, None

        # construct loss end layers and evaluate losses
        # data loss in the atlas space (warped using the positive flow)
        pos_loss_layer = nes.layers.LossEndPoint(loss_pos_fn, name='atlas_space',
                                                 metric_fn=metric_pos, metric_name=mn_pos)
        if is_var:
            atlas_loss = pos_loss_layer([atlas_mean, warped_subject, atlas_var])
        else:
            atlas_loss = pos_loss_layer([atlas_mean, warped_subject])

        if is_bidir:
            # data loss in the subject space (warped using the negative flow)
            neg_loss_layer = nes.layers.LossEndPoint(loss_neg_fn, name='subject_space',
                                                     metric_fn=metric_neg, metric_name=mn_neg)
            if is_var:
                subject_loss = neg_loss_layer([this_input, warped_atlas, subject_var])
            else:
                subject_loss = neg_loss_layer([this_input, warped_atlas])

        # regularization loss
        reg_loss = nes.layers.create_loss_end([pos_flow], loss_reg_fn, name='flow',
                                              metric_fn=metric_reg, metric_name=mn_reg)
        # mean stream loss
        mean_stream = ne.layers.MeanStream(name='mean_stream')(pos_flow)
        ms_loss = nes.layers.create_loss_end([mean_stream], loss_ms_fn, name='loss_mean_stream',
                                             metric_fn=metric_ms, metric_name=mn_ms)

        # initialize the model with loss end points, where outputs are scalar losses
        if is_bidir:
            model_outputs = [atlas_loss, subject_loss, reg_loss, ms_loss]
        else:
            model_outputs = [atlas_loss, reg_loss, ms_loss]

        super().__init__(inputs=model_inputs, outputs=model_outputs)

        # until this point, the model construction is done
        # below we construct a separate model for generating outputs
        # without loss end points
        model_no_lep = tf.keras.Model(inputs=model_inputs,
                                      outputs=[warped_subject, warped_atlas, pos_flow, neg_flow])

        # cache pointers to important layers and tensors for future reference
        self.references = ne.modelio.LoadableModel.ReferenceContainer()
        self.references.mean_layer = mean_layer
        self.references.atlas_mean = atlas_mean
        self.references.is_var = is_var
        if is_var:
            self.references.var_layer = var_layer
            self.references.atlas_var = atlas_var
            self.references.subject_var = subject_var
        self.references.vxm_model = vxm_model
        self.references.model_no_lep = model_no_lep
        self.references.warped_subject = warped_subject
        self.references.warped_atlas = warped_atlas
        self.references.pos_flow = pos_flow
        self.references.neg_flow = neg_flow
        self.references.model_register_to_atlas = None

    def set_atlas_mean(self, data):
        if data.shape[1]:
            data = np.reshape(data, data.shape[1:])
        self.references.mean_layer.set_weights([data])

    def get_atlas_mean(self):
        return self.references.mean_layer.get_weights()[0].squeeze()

    def set_atlas_var(self, data):
        if self.references.is_var:
            if data.shape[1]:
                data = np.reshape(data, data.shape[1:])
            self.references.var_layer.set_weights([data])
        else:
            raise NotImplementedError('variance layer not enabled')

    def get_atlas_var(self):
        if self.references.is_var:
            return self.references.var_layer.get_weights()[0].squeeze()
        else:
            return None

    def get_atlas_std(self):
        if self.references.is_var:
            return np.sqrt(self.get_atlas_var())
        else:
            return None

    def get_model_outputs(self, src, batch_size=32):
        if not isinstance(src, (list, tuple)):
            src = [src]
        return self.references.model_no_lep.predict(src, batch_size=batch_size)

    def get_warped_subject(self, src, batch_size=32):
        return self.get_model_outputs(src, batch_size=batch_size)[0]

    def get_warped_atlas(self, src, batch_size=32):
        return self.get_model_outputs(src, batch_size=batch_size)[1]

    def get_positive_flow(self, src):
        return self.get_model_outputs(src)[2]

    def get_negative_flow(self, src):
        return self.get_model_outputs(src)[3]

    def register_to_atlas(self, src, img, interp_method='linear', fill_value=None, batch_size=32):
        img_input = tf.keras.Input(shape=img.shape[1:])
        st_layer = vxm.layers.SpatialTransformer(interp_method=interp_method, fill_value=fill_value)
        img_output = st_layer([img_input, self.references.pos_flow])
        model_inputs = (*self.inputs, img_input)
        model_outputs = [img_output]

        # cache the model if not done before and the input image shape is the same
        # this avoid building the model every time the function is called
        if (self.references.model_register_to_atlas is None) or \
           (self.references.model_register_to_atlas.inputs[1].shape[1:] != img.shape[1:]):
            model_register = tf.keras.Model(inputs=model_inputs, outputs=model_outputs)
            self.references.model_register_to_atlas = model_register
        else:
            model_register = self.references.model_register_to_atlas

        if isinstance(src, (list, tuple)):
            data_in = [*src, img]
        else:
            data_in = [src, img]

        return model_register.predict(data_in, batch_size=batch_size)


# SphereMorph is a subclass of SphereMorphWithAtlasBuilding without
# a trainable atlas and positional encoding
class SphereMorph(SphereMorphWithAtlasBuilding):
    @ne.modelio.store_config_args
    def __init__(self,
                 input_model=None,
                 input_shape=None,
                 num_ft=None,
                 nb_unet_features=None,
                 loss_fn=None,
                 metric_fn=None,
                 metric_name=None,
                 is_var=False,
                 is_bidir=True,
                 pad_size=0,
                 name='spm',
                 **kwargs):

        is_atlas_trainable = False
        pos_enc = 0
        super().__init__(input_model=input_model,
                         input_shape_ft=input_shape,
                         num_ft=num_ft,
                         nb_unet_features=nb_unet_features,
                         loss_fn=loss_fn,
                         metric_fn=metric_fn,
                         metric_name=metric_name,
                         is_var=is_var,
                         is_bidir=is_bidir,
                         pad_size=pad_size,
                         is_atlas_trainable=is_atlas_trainable,
                         pos_enc=pos_enc,
                         name=name,
                         **kwargs)


###############################################################################
# September 2023 iteration of joint, affine, deformable SynthMorph models.
###############################################################################

def VxmAffineFeatureDetectorAverage(
    in_shape=None,
    input_model=None,
    num_chan=1,
    num_key=64,
    enc_nf=[256] * 4,
    dec_nf=[],
    add_nf=[256] * 4,
    per_level=1,
    dropout=0,
    half_res=True,
    weighted=True,
    rigid=False,
    make_dense=True,
    bidir=False,
    return_trans_to_mid_space=False,
    return_trans_to_half_res=False,
    return_key=False,
):
    """Symmetric affine-registration network using single-image feature detection.

    Internally, the model computes the transform in a centered frame at full resolution. However,
    matrix transforms returned with `make_dense=False` operate on zero-based indices to facilitate
    composition, in particular for changing resolutions. Thus, any subsequent SpatialTransformer or
    ComposeTransform calls must use `shift_center=False`.

    While the predicted transforms always apply to images at full resolution, you can use the flag
    `return_trans_to_half_res=True` to obtain transforms producing outputs at half resolution for
    faster training. Careful: this requires adequately setting the SpatialTransformer output
    `shape` when applying transforms.

    Parameters:
        in_shape: Spatial dimensions of the input images as an iterable.
        input_model: Model whose outputs will be used as data inputs, and whose inputs will be used
            as inputs to the returned model, as an alternative to specifying `in_shape`.
        num_chan: Number of input-image channels.
        num_key: Number of output feature maps giving rise to key points.
        enc_nf: Number of convolutional encoder filters at each level as an interable. The model
            will downsample by a factor of 2 after each convolution.
        dec_nf: Number of convolutional decoder filters at each level as an iterable. The model
            will upsample by a factor of 2 after each convolution.
        add_nf: Number of additional convolutional filters applied at the end, as an iterable. The
            model will maintain the resolution after each convolution.
        per_level: Number of times the model will repeat each encoding and decoding convolution.
        dropout: Dropout rate applied after each internal convolution.
        half_res: For efficiency, halve the input-image resolution before registration.
        weighted: Fit the transforms using weighted instead of ordinary least squares.
        rigid: Discard scaling and shear to return a rigid transform.
        make_dense: Return a dense displacement field instead of a matrix transform.
        bidir: In addition to the transform from image 1 to image 2, also return the inverse. The
            transforms apply to full-resolution images but may end half way and/or at half
            resolution, depending on `return_trans_to_mid_space` and `return_trans_to_half_res`.
        return_trans_to_mid_space: Return transforms from the input images to an affine mid-space.
            Careful: your loss inputs must reflect this choice, and training with large transforms
            may lead to NaN loss values. You can change the option after training.
        return_trans_to_half_res: Return transforms from source images at full resolution to target
            images at half resolution. You can change the option after training.
        return_key: Append the output feature maps to the model outputs.

    Returns:
        Symmetric affine-registration model returning a transform from the first to the second
        input image. With `bidir`, the second output will transform the second to the first image.

    Author:
        mu40

    If you find this model useful, please cite:
        Anatomy-specific acquisition-agnostic affine registration learned from fictitious images
        M Hoffmann, A Hoopes, B Fischl*, AV Dalca* (*equal contribution)
        SPIE Medical Imaging: Image Processing, 12464, p 1246402, 2023
        https://doi.org/10.1117/12.265325

    """
    # Inputs.
    if input_model is None:
        source = tf.keras.Input(shape=(*in_shape, num_chan))
        target = tf.keras.Input(shape=(*in_shape, num_chan))
        input_model = tf.keras.Model(*[(source, target)] * 2)
    source, target = input_model.outputs[:2]

    # Dimensions.
    full_shape = np.asarray(source.shape[1:-1])
    half_shape = full_shape // 2
    num_dim = len(full_shape)
    assert num_dim in (2, 3), 'only 2D and 3D supported'
    assert not return_trans_to_half_res or half_res, 'only for `half_res=True`'

    # Layers.
    conv = getattr(KL, f'Conv{num_dim}D')
    pool = getattr(KL, f'MaxPool{num_dim}D')
    drop = getattr(KL, f'SpatialDropout{num_dim}D')
    up = getattr(KL, f'UpSampling{num_dim}D')

    # Static transforms. Function names refer to effect on coordinates.
    dtype = tf.keras.mixed_precision.global_policy().compute_dtype

    def tensor(x):
        x = tf.constant(x[None, :-1, :], dtype)
        return tf.repeat(x, repeats=tf.shape(source)[0], axis=0)

    def cen(shape):
        mat = np.eye(num_dim + 1)
        mat[:-1, -1] = -0.5 * (shape - 1)
        return tensor(mat)

    def un_cen(shape):
        mat = np.eye(num_dim + 1)
        mat[:-1, -1] = +0.5 * (shape - 1)
        return tensor(mat)

    def scale(fact):
        mat = np.diag((*[fact] * num_dim, 1))
        return tensor(mat)

    if half_res:
        prop = dict(fill_value=0, shape=half_shape, shift_center=False)
        source = vxm.layers.SpatialTransformer(**prop)((source, scale(2)))
        target = vxm.layers.SpatialTransformer(**prop)((target, scale(2)))

    # Internal U-Net. Encoder.
    inp = tf.keras.Input(shape=(*source.shape[1:-1], num_chan))
    x = inp
    prop = dict(kernel_size=3, padding='same')
    enc = []
    for n in enc_nf:
        for _ in range(per_level):
            x = conv(n, **prop)(x)
            x = drop(dropout)(x)
            x = KL.LeakyReLU(0.2)(x)
        enc.append(x)
        x = pool(dtype=tf.float32)(x)

    # Decoder.
    for n in dec_nf:
        for _ in range(per_level):
            x = conv(n, **prop)(x)
            x = drop(dropout)(x)
            x = KL.LeakyReLU(0.2)(x)
        x = KL.concatenate([up()(x), enc.pop()])

    # Additional convolutions.
    for n in add_nf:
        x = conv(n, **prop)(x)
        x = drop(dropout)(x)
        x = KL.LeakyReLU(0.2)(x)

    # Features.
    x = conv(num_key, activation='relu', **prop)(x)
    unet = tf.keras.Model(inp, outputs=x)

    # Always sum and fit affine with single precision.
    key_1 = unet(source)
    key_2 = unet(target)
    if tf.keras.mixed_precision.global_policy().compute_dtype == 'float16':
        key_1 = tf.cast(key_1, tf.float32)
        key_2 = tf.cast(key_2, tf.float32)

    # Barycenters.
    prop = dict(axes=range(1, num_dim + 1), normalize=True, shift_center=True, dtype=dtype)
    cen_1 = ne.utils.barycenter(key_1, **prop) * full_shape
    cen_2 = ne.utils.barycenter(key_2, **prop) * full_shape

    # Weights.
    axes = range(1, num_dim + 1)
    pow_1 = tf.reduce_sum(key_1, axis=axes)
    pow_2 = tf.reduce_sum(key_2, axis=axes)
    pow_1 /= tf.reduce_sum(pow_1, axis=-1, keepdims=True)
    pow_2 /= tf.reduce_sum(pow_2, axis=-1, keepdims=True)
    weights = pow_1 * pow_2

    # Least squares and average, since the fit is not symmetric.
    aff_1 = vxm.utils.fit_affine(cen_1, cen_2, weights=weights if weighted else None)
    aff_2 = vxm.utils.fit_affine(cen_2, cen_1, weights=weights if weighted else None)
    aff_1 = 0.5 * (vxm.utils.invert_affine(aff_2) + aff_1)

    if rigid:
        aff_1 = vxm.utils.affine_matrix_to_params(aff_1)
        aff_1 = aff_1[:, :num_dim * (num_dim + 1) // 2]
        aff_1 = vxm.layers.ParamsToAffineMatrix(ndims=num_dim)(aff_1)

    # Mid space. Before scaling at either side.
    aff_2 = vxm.utils.invert_affine(aff_1)
    if return_trans_to_mid_space:
        aff_1 = vxm.utils.make_square_affine(aff_1)
        aff_1 = tf.linalg.sqrtm(aff_1)[:, :-1, :]

        aff_2 = vxm.utils.make_square_affine(aff_2)
        aff_2 = tf.linalg.sqrtm(aff_2)[:, :-1, :]

    # Affine transform operating in index space, for full-resolution inputs.
    prop = dict(shift_center=False)
    aff_1 = vxm.layers.ComposeTransform(**prop)((un_cen(full_shape), aff_1, cen(full_shape)))
    aff_2 = vxm.layers.ComposeTransform(**prop)((un_cen(full_shape), aff_2, cen(full_shape)))
    out = (aff_1, aff_2)

    if return_trans_to_half_res:
        out = [(x, scale(2)) for x in out]
        out = [vxm.layers.ComposeTransform(shift_center=False)(x) for x in out]

    if tf.keras.mixed_precision.global_policy().compute_dtype == 'float16':
        out = [tf.cast(x, tf.float16) for x in out]

    if make_dense:
        shape = half_shape if return_trans_to_half_res else full_shape
        out = [vxm.layers.AffineToDenseShift(shape, shift_center=False)(x) for x in out]

    # Additional outputs.
    if return_key:
        out.extend([key_1, key_2])
    if not bidir:
        out = out[::2]
    if len(out) == 1:
        out = out[0]

    return tf.keras.Model(input_model.inputs, out)


def VxmDenseAverage(
    in_shape=None,
    input_model=None,
    num_chan=1,
    enc_nf=[256] * 4,
    dec_nf=[256] * 4,
    add_nf=[256] * 4,
    per_level=1,
    int_steps=5,
    upsample=True,
    half_res=False,
    dropout=0,
    bidir=False,
    average=False,
    hyp_num=1,
    hyp_den=[],
):
    """Symmetric deformable-registration network.

    Internally, the model computes the transform in a centered frame at full resolution. However,
    matrix transforms returned with `make_dense=False` operate on zero-based indices to facilitate
    composition, in particular for changing resolutions. Thus, any subsequent SpatialTransformer or
    ComposeTransform calls must use `shift_center=False`.

    While the predicted transforms always apply to images at full resolution, you can use the flag
    `return_trans_to_half_res=True` to obtain transforms producing outputs at half resolution for
    faster training. Careful: this requires adequately setting the SpatialTransformer output
    `shape` when applying transforms.

    Parameters:
        in_shape: Spatial dimensions of the input images as an iterable.
        input_model: Model whose outputs will be used as data inputs, and whose inputs will be used
            as inputs to the returned model, as an alternative to specifying `in_shape`.
        num_chan: Number of input-image channels.
        enc_nf: Number of convolutional encoder filters at each level as an interable. The model
            will downsample by a factor of 2 after each convolution.
        dec_nf: Number of convolutional decoder filters at each level as an iterable. The model
            will upsample by a factor of 2 after each convolution.
        add_nf: Number of additional convolutional filters applied at the end, as an iterable. The
            model will maintain the resolution after each convolution.
        per_level: Number of times the model will repeat each encoding and decoding convolution.
        int_steps: Number of integration steps used to compute the displacement field from the SVF.
            If `0`, the model directly predicts the displacement field instead of an SVF.
        upsample: Resize the output tensor if its shape is smaller than `in_shape`.
        half_res: For efficiency, halve the input-image resolution before registration.
        dropout: Dropout rate applied after each internal convolution.
        rigid: Discard scaling and shear to return a rigid transform.
        bidir: In addition to the transform from image 1 to image 2, also return the inverse. With
            `int_steps=0`, the model returns the - probably useless - negated forward transform.
        average: Efficiently predict twice, reversing the input image order, and average to obtain
            produce symmetric transforms.
        hyp_num: Number of hyperparameter inputs for predicting the weights of the registration
            model with a hypernetwork. The option has no effect with `hyp_den=[]`.
        hyp_den: Fully-connected output units for each layer of the hypernetwork as an iterable.
            Pass `hyp_den=[]`, if you do not want a hypernetwork.

    Returns:
        Symmetric deformable-registration model returning a transform from the first to the second
        input image. With `bidir`, the second output will transform the second to the first image.

    Author:
        mu40

    If you find this model useful, please cite:
        M Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca
        SynthMorph: learning contrast-invariant registration without acquired images
        IEEE Transactions on Medical Imaging (TMI), 41 (3), 543-558, 2022
        https://doi.org/10.1109/TMI.2021.3116879

    """
    if input_model is None:
        hyp_in = tf.keras.Input(shape=[hyp_num])
        source = tf.keras.Input(shape=(*in_shape, num_chan))
        target = tf.keras.Input(shape=(*in_shape, num_chan))
        inputs = (hyp_in, source, target) if hyp_den else (source, target)
        input_model = tf.keras.Model(inputs, inputs)
    *hyp_in, source, target = input_model.outputs

    in_shape = np.asarray(source.shape[1:-1])
    num_dim = len(in_shape)
    assert num_dim in (2, 3), 'only 2D and 3D supported'

    pool = getattr(KL, f'MaxPool{num_dim}D')
    drop = getattr(KL, f'SpatialDropout{num_dim}D')
    up = getattr(KL, f'UpSampling{num_dim}D')

    # Hypernetwork.
    if hyp_den:
        x = hyp_in[0]
        for n in hyp_den:
            x = KL.Dense(n, activation='relu')(x)
            x = KL.Dropout(rate=dropout)(x)
        hyp_out = x

    # Convolution.
    def conv(x, filters):
        prop = dict(filters=filters, kernel_size=3, padding='same')
        if hyp_den:
            return ne.layers.HyperConvFromDense(num_dim, **prop)((x, hyp_out))

        return getattr(KL, f'Conv{num_dim}D')(**prop)(x)

    # Encoder.
    x = KL.concatenate((source, target))
    if half_res:
        x = pool()(x)
    enc = [x]
    for n in enc_nf:
        for _ in range(per_level):
            x = conv(x, filters=n)
            x = KL.LeakyReLU(0.2)(x)
            x = drop(rate=dropout)(x)
        enc.append(x)
        x = pool(dtype=tf.float32)(x)

    # Decoder.
    for n in dec_nf:
        for _ in range(per_level):
            x = conv(x, filters=n)
            x = KL.LeakyReLU(0.2)(x)
            x = drop(rate=dropout)(x)
        x = KL.concatenate([up()(x), enc.pop()])

    # Additional convolutions.
    for n in add_nf:
        x = conv(x, filters=n)
        x = KL.LeakyReLU(0.2)(x)
        x = drop(rate=dropout)(x)

    # SVF (or transform if no integration or averaging).
    fw = conv(x, filters=num_dim)

    # Averaging (before integration).
    if average:
        model = tf.keras.Model(input_model.inputs, outputs=fw)
        fw = model((*hyp_in, source, target))
        bw = model((*hyp_in, target, source))
        fw = 0.5 * (fw - bw)

    # Integration.
    bw = fw * -1
    if int_steps > 0:
        fw = vxm.layers.VecInt(method='ss', int_steps=int_steps)(fw)
        bw = vxm.layers.VecInt(method='ss', int_steps=int_steps)(bw)

    # Rescaling.
    zoom = source.shape[1] // fw.shape[1]
    if upsample and zoom > 1:
        fw = vxm.layers.RescaleTransform(zoom)(fw)
        bw = vxm.layers.RescaleTransform(zoom)(bw)

    out = (fw, bw) if bidir else fw
    return tf.keras.Model(input_model.inputs, out)


class VxmJointAverage(ne.tf.modelio.LoadableModel):

    @ne.modelio.store_config_args
    def __init__(self,
                 in_shape=None,
                 input_model=None,
                 num_chan=1,
                 bidir=False,
                 zero_affine=False,
                 override_affine=False,
                 return_trans_to_half_res=False,
                 return_tot=True,
                 return_def=False,
                 return_aff=False,
                 **kwargs):
        """Wrapper model for symmetric joint affine-deformable registration.

        To save memory, the registration runs at half resolution. We downsample full-resolution 
        inputs within this model to avoid resampling twice. The returned transforms apply to 
        full-resolution images. For efficient training, pass `return_trans_to_half_res=True` to 
        return transforms from full-resolution inputs to half-resolution outputs. This requires
        apropriately setting the SpatialTransformer `shape` in the training script. Concatenating 
        matrices is cheap but requires transforms operating on zero-based indices when changing
        resolution: `shift_center=True` would un-shift by the wrong shape - set it to `False` or 
        suffer.

        Unilateral version that does not predict warps in affine mid-space.

        Parameters:
        in_shape: Spatial dimensions of the input images as an iterable.
        input_model: Model whose outputs will be used as data inputs, and whose inputs will be used
        as inputs to the returned model, as an alternative to specifying `in_shape`.
        num_chan: Number of input-image channels.
        bidir: In addition to transforms from image 1 to image 2, also return each inverse. The
        transforms apply to full-resultion images but may end at half resolution, depending on
        `return_trans_to_half_res`.
        zero_affine: Set the predicted affine transform to identity. This is inefficient but
        enables deformable-only registration without manipulating the model or its weights.
        override_affine: Override the predicted affine transform with an affine matrix appended to
        the model inputs, transforming zero-based indices of image 2 to those of image 1.
        return_trans_to_half_res: Return transforms from source images at full resolution to target
        images at half resolution. You can change the option after training.
        return_tot: Return the composed affine-deformable transform.
        return_def: Return the deformable transform.
        return_aff: Return the affine transform.
        kwargs: Arguments to the affine and deformable sub-models, prepended with 'aff.' and
            'def.', respectively. See `VxmAffineFeatureDetectorAverage` and `VxmDenseAverage`.

    Returns:
        Symmetric joint registration model returning transforms from the first to the second input
        image. With `bidir`, every other output will transform the second to the first image.

    Author:
        mu40

    If you find this model useful, please cite:
        Anatomy-aware and acquisition-agnostic joint registration with SynthMorph
        M Hoffmann, A Hoopes, DN Greve, B Fischl*, AV Dalca* (*equal contribution)
        arXiv preprint arXiv:2301.11329
        https://doi.org/10.48550/arXiv.2301.11329

    """
        # Sub-network options.
        def extract_options(arg, prefix):
            keys = [k for k in arg if k.startswith(prefix)]
            return {k[len(prefix):]: arg.pop(k) for k in keys}
        arg_aff = extract_options(kwargs, prefix='aff.')
        arg_def = extract_options(kwargs, prefix='def.')
        assert not kwargs, f'unknown arguments {kwargs}'

        if input_model is None:
            hyp = tf.keras.Input(shape=[arg_def.get('hyp_num', 1)])
            source = tf.keras.Input(shape=(*in_shape, num_chan))
            target = tf.keras.Input(shape=(*in_shape, num_chan))
            inputs = (source, target)
            if arg_def.get('hyp_den'):
                inputs = (hyp, *inputs)
            input_model = tf.keras.Model(inputs, inputs)
        *hyp, source, target = input_model.outputs

        # Dimensions.
        full_shape = np.asarray(source.shape[1:-1])
        half_shape = full_shape // 2
        num_dim = len(full_shape)
        prop = dict(fill_value=0, shape=half_shape, shift_center=False)

        # Static transforms. Function names refer to effect on coordinates.
        def tensor(x):
            dtype = tf.keras.mixed_precision.global_policy().compute_dtype
            x = tf.constant(x[None, :-1, :], dtype)
            return tf.repeat(x, repeats=tf.shape(source)[0], axis=0)

        def cen(shape):
            mat = np.eye(num_dim + 1)
            mat[:-1, -1] = -0.5 * (shape - 1)
            return tensor(mat)

        def un_cen(shape):
            mat = np.eye(num_dim + 1)
            mat[:-1, -1] = +0.5 * (shape - 1)
            return tensor(mat)

        def scale(fact):
            mat = np.diag((*[fact] * num_dim, 1))
            return tensor(mat)

        # Affine registration. Transforms will operate in half-resolution index
        # space and transform all the way from one image to the other.
        arg_aff.update(
            in_shape=half_shape,
            make_dense=False,
            half_res=False,
            bidir=True,
        )
        prop = dict(fill_value=0, shape=half_shape, shift_center=False)
        inp_1 = vxm.layers.SpatialTransformer(**prop)((source, scale(2)))
        inp_2 = vxm.layers.SpatialTransformer(**prop)((target, scale(2)))
        aff_1, aff_2 = VxmAffineFeatureDetectorAverage(**arg_aff)((inp_1, inp_2))

        # For initialization with full-to-full zero-based index transforms.
        assert not (override_affine and zero_affine), 'incompatible options'
        if override_affine:
            affine = tf.keras.Input(shape=(num_dim, num_dim + 1))
            input_model = tf.keras.Model(
                inputs=(input_model.inputs, affine),
                outputs=(input_model.outputs, affine),
            )
            *hyp, source, target, affine = input_model.outputs

            aff_1 = aff_1 * 0 + affine
            aff_1 = vxm.layers.ComposeTransform()((scale(0.5), aff_1, scale(2)))
            aff_2 = vxm.utils.invert_affine(aff_1)

        # Set affine to zero to estimate a deformable registration only, for an
        # image pair assumed to be in affine alignment. For computing Jacobians.
        if zero_affine:
            aff_1 = aff_1 * 0 + scale(1)
            aff_2 = aff_2 * 0 + scale(1)

        # Deformable input. Affine transforms from full to half resolution.
        aff_1 = vxm.layers.ComposeTransform(shift_center=False)((scale(2), aff_1))
        aff_2 = vxm.layers.ComposeTransform(shift_center=False)((scale(2), aff_2))
        mov_1 = vxm.layers.SpatialTransformer(**prop)((source, aff_1))

        # Deformable registration. Inputs at half resolution already.
        arg_def.update(half_res=False)
        model_def = VxmDenseAverage(half_shape, average=True, bidir=True, **arg_def)
        def_1, def_2 = model_def((*hyp, mov_1, inp_2))

        # Warps from full to half resolution. Converts matrices to dense transforms
        # using the half-resolution shape derived from the warps.
        tot_1 = vxm.layers.ComposeTransform(shift_center=False)((aff_1, def_1))
        tot_2 = vxm.layers.ComposeTransform(shift_center=False)((aff_2, def_2))

        # Do not interpolate deformation fields with `fill_value=0`.
        down = vxm.layers.AffineToDenseShift(full_shape, shift_center=False)(scale(0.5))
        if not return_trans_to_half_res:
            tot_1 = vxm.layers.ComposeTransform()((tot_1, down))
            tot_2 = vxm.layers.ComposeTransform()((tot_2, down))
            def_1 = vxm.layers.ComposeTransform(shift_center=False)((scale(2), def_1, down))
            def_2 = vxm.layers.ComposeTransform(shift_center=False)((scale(2), def_2, down))
            aff_1 = vxm.layers.ComposeTransform(shift_center=False)((aff_1, scale(0.5)))
            aff_2 = vxm.layers.ComposeTransform(shift_center=False)((aff_2, scale(0.5)))

        # Outputs. Pick model defaults to facilitate evaluation.
        out = []
        if return_tot:
            out.extend([tot_1, tot_2])
        if return_def:
            out.extend([def_1, def_2])
        if return_aff:
            out.extend([aff_1, aff_2])
        if not bidir:
            out = out[::2]

        self.references = ne.tf.modelio.LoadableModel.ReferenceContainer()
        self.references.vxm_dense = model_def

        super().__init__(inputs=input_model.inputs, outputs=out)
        # return tf.keras.Model(input_model.inputs, out)
