

# third party
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as KL

# ours
import voxelmorph as vxm
import neurite as ne
import neurite_sandbox as nes


class HyperCVAE(tf.keras.Model):
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
                 nb_hyp_params=1,
                 nb_hyp_layers=4,
                 nb_hyp_units=64,
                 enc_dense=None,
                 name='hyper_vxm_dense'):

        # build hypernetwork
        hyp_input = tf.keras.Input(shape=[nb_hyp_params], name='%s_hyp_input' % name)
        hyp_last = hyp_input
        for n in range(nb_hyp_layers):
            hyp_last = KL.Dense(nb_hyp_units,
                                activation='relu',
                                name='%s_hyp_dense_%d' % (name, n + 1))(hyp_last)

        hyp_enc_tensor = None
        if nb_hyp_params == 2:
            hyp_enc_tensor = KL.Lambda(lambda x: x[:, 1], name='normalize_hyp2_input')(hyp_input)

        # all inputs
        model_ae_init = VAE(inshape + [3],
                            nb_features=nb_unet_features,
                            nb_levels=nb_unet_levels,
                            feat_mult=unet_feat_mult,
                            nb_conv_per_level=nb_unet_conv_per_level,
                            half_res=unet_half_res,
                            hyp_input=hyp_input,
                            hyp_tensor=hyp_last,
                            enc_dense=enc_dense,
                            hyp_enc_tensor=hyp_enc_tensor,
                            name='%s_ae' % name)

        ds_input = KL.Input(inshape + [1], name='ds_input')
        full_input = KL.Input(inshape + [1], name='blob_input')
        mask_input = KL.Input(inshape + [1], name='mask_input')
        ae_inputs = [ds_input, full_input, mask_input]
        ae_inputs_stack = KL.Concatenate(axis=-1)(ae_inputs)
        model_ae = tf.keras.models.Model(ae_inputs + [hyp_input],
                                         model_ae_init([ae_inputs_stack, hyp_input]))

        vae_outputs = model_ae.outputs + \
            [model_ae.layers[-1].get_layer('%s_ae_mu' % name).get_output_at(1),
             model_ae.layers[-1].get_layer('%s_ae_logvar' % name).get_output_at(1)]

        nb_unet_features_unet = [f for f in nb_unet_features]
        nb_unet_features_unet[-1].append(1)
        model_unet = vxm.networks.Unet(inshape + [model_ae.output.shape[-1] + 2],
                                       nb_features=nb_unet_features_unet,
                                       nb_levels=nb_unet_levels,
                                       feat_mult=unet_feat_mult,
                                       nb_conv_per_level=nb_unet_conv_per_level,
                                       half_res=unet_half_res,
                                       hyp_input=hyp_input,
                                       hyp_tensor=hyp_last,
                                       name='%s_unet' % name)
        new_unet_inputs_stack = KL.Concatenate(axis=-1)([model_ae.output, ds_input, mask_input])
        outputs = [model_unet([new_unet_inputs_stack, hyp_input])] + vae_outputs[1:]

        super().__init__(inputs=ae_inputs + [hyp_input], outputs=outputs, name=name)

        # model vae
        self.model_vae = tf.keras.Model(ae_inputs + [hyp_input], vae_outputs)

        # print(model_ae_init.summary(line_length=120))

        # z model decoder
        enc_shape = model_ae.layers[-1].get_layer('%s_ae_sample' % name).get_output_at(1).shape[1:]
        new_z_input = KL.Input(enc_shape, name='z_input')
        t = new_z_input
        layer_names = [f.name for f in model_ae_init.layers]
        for name in layer_names[layer_names.index('%s_ae_sample' % name) + 1:]:
            lay = model_ae_init.get_layer(name)
            if isinstance(lay.input, list):
                t = lay([t, lay.input[-1]])
            else:
                t = lay(t)

        # final z model
        new_out = KL.Concatenate(axis=-1)([t, ds_input, mask_input])
        self.model_z = tf.keras.models.Model([new_z_input, ds_input, mask_input, hyp_input],
                                             [model_unet([new_out, hyp_input])])

    def hyper_loss_model(self):
        """
        prep a model where the outputs compute the losses.
        This is useful because of Eager complications of using model parts in the loss
        """
        # prepare useful Tensors
        hyp = self.inputs[-1]
        flat_hyp = K.flatten(hyp[:, 0])
        out_img = self.outputs[0]
        out_mu = self.outputs[1]
        out_logvar = self.outputs[2]
        full_input = self.inputs[1]  # input ae branch sees

        print(flat_hyp, out_mu)

        # prepare losses
        vae_loss = nes.losses.VAE()
        hmse_loss = lambda x: x[0] * K.sum(K.batch_flatten((x[1] - x[2])**2), 1)
        hmu_loss = lambda x: (1 - x[0]) * vae_loss.kl_mu(0, x[1])
        hvar_loss = lambda x: (1 - x[0]) * vae_loss.kl_log_sigma(0, x[1])

        # loss output tensors
        out_v = KL.Lambda(hmse_loss, name='mse_loss')([flat_hyp, out_img, full_input])
        out_mu = KL.Lambda(hmu_loss, name='KL_mu_loss')([flat_hyp, out_mu])
        out_logvar = KL.Lambda(hvar_loss, name='KL_logvar_loss')([flat_hyp, out_logvar])

        # return model
        return tf.keras.Model(self.inputs, [out_v, out_mu, out_logvar])


class VAE(tf.keras.Model):
    """
    A vae architecture that builds off either an input keras model or input shape. Layer features
    can be specified directly as a list of encoder and decoder features or as a single integer along
    with a number of unet levels. The default network features per layer (when no options are
    specified) are:
        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]

    This network specifically does not subclass LoadableModel because it's meant to be a core,
    internal model for more complex networks, and is not meant to be saved/loaded independently.
    """

    def __init__(self,
                 inshape=None,
                 input_model=None,
                 nb_features=None,
                 nb_levels=None,
                 max_pool=2,
                 feat_mult=1,
                 nb_conv_per_level=1,
                 do_res=False,
                 half_res=False,
                 hyp_enc_tensor=None,
                 hyp_input=None,
                 hyp_tensor=None,
                 enc_dense=None,
                 name='vae'):
        """
        Parameters:
            inshape: Optional input tensor shape (including features). e.g. (192, 192, 192, 2).
            input_model: Optional input model that feeds directly into the unet before concatenation
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer.
                If None (default), the unet features are defined by the default config described in
                the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer.
                Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer.
                Default is 1.
            nb_conv_per_level: Number of convolutions per unet level. Default is 1.
            half_res: Skip the last decoder upsampling. Default is False.
            hyp_input: Hypernetwork input tensor. Enables HyperConvs if provided. Default is None.
            hyp_tensor: Hypernetwork final tensor. Enables HyperConvs if provided. Default is None.
            name: Model name - also used as layer name prefix. Default is 'unet'.
        """

        # have the option of specifying input shape or input model
        if input_model is None:
            if inshape is None:
                raise ValueError('inshape must be supplied if input_model is None')
            unet_input = KL.Input(shape=inshape, name='%s_input' % name)
            model_inputs = [unet_input]
        else:
            unet_input = KL.concatenate(input_model.outputs, name='%s_input_concat' % name)
            model_inputs = input_model.inputs

        # add hyp_input tensor if provided
        if hyp_input is not None:
            if not isinstance(hyp_input, (list, tuple)):
                hyp_input = [hyp_input]
            model_inputs = model_inputs + hyp_input

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = vxm.networks.default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')

        ndims = len(unet_input.get_shape()) - 2
        assert ndims in (1, 2, 3), 'ndims should be one of 1, 2, or 3. found: %d' % ndims
        MaxPooling = getattr(KL, 'MaxPooling%dD' % ndims)

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            max_pool = [max_pool] * nb_levels

        # configure encoder (down-sampling path)
        enc_layers = []
        last = unet_input
        for level in range(nb_levels - 1):
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                layer_name = '%s_enc_conv_%d_%d' % (name, level, conv)
                last = vxm.networks._conv_block(last, nf, name=layer_name, do_res=do_res,
                                                hyp_tensor=hyp_tensor)
            enc_layers.append(last)

            # temporarily use maxpool since downsampling doesn't exist in keras
            last = MaxPooling(max_pool[level], name='%s_enc_pooling_%d' % (name, level))(last)

        if enc_dense is not None:
            last_shape = last.shape[1:]
            last = KL.Flatten(name='%s_flatten' % name)(last)
            mu = ne.layers.HyperDenseFromDense(enc_dense, name='%s_mu' % name)([last, hyp_tensor])
            logvar = ne.layers.HyperDenseFromDense(enc_dense,
                                                   name='%s_logvar' % name)([last, hyp_tensor])

        else:
            # get mu and sigma
            mu = vxm.networks._conv_block(last, nf, name='%s_mu' % name, do_res=do_res,
                                          include_activation=False,
                                          hyp_tensor=hyp_tensor)
            logvar = vxm.networks._conv_block(last, nf, name='%s_logvar' % name, do_res=do_res,
                                              include_activation=False,
                                              hyp_tensor=hyp_tensor)

        # sample

        if hyp_enc_tensor is not None:
            last = ne.layers.SampleNormalLogVar(name='%s_sample_tmp' % name)([mu, logvar])

            assert enc_dense is not None, 'hyperencoding is only possible with dense right now'
            mask_enc = lambda x: nes.utils.mask_encoding(x * enc_dense, enc_dense)
            oh = KL.Lambda(mask_enc, name='mask_encoding')(hyp_enc_tensor)
            last = KL.Lambda(lambda x: x[0] * tf.cast(x[1], tf.float32),
                             name='%s_sample' % name)([last, oh])
        else:
            last = ne.layers.SampleNormalLogVar(name='%s_sample' % name)([mu, logvar])

        # undo
        if enc_dense is not None:
            last = ne.layers.HyperDenseFromDense(np.prod(last_shape),
                                                 name='%s_undense' % name)([last, hyp_tensor])
            last = KL.Reshape(last_shape, name='%s_resize' % name)(last)

        # configure decoder (up-sampling path)
        for level in range(nb_levels - 1):
            real_level = nb_levels - level - 2
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                layer_name = '%s_dec_conv_%d_%d' % (name, real_level, conv)
                last = vxm.networks._conv_block(last, nf, name=layer_name, do_res=do_res,
                                                hyp_tensor=hyp_tensor)
            if not half_res or level < (nb_levels - 2):
                layer_name = '%s_dec_upsample_%d' % (name, real_level)
                last = _upsample_block(last, factor=max_pool[real_level], name=layer_name)

        # now we take care of any remaining convolutions
        for num, nf in enumerate(final_convs):
            layer_name = '%s_dec_final_conv_%d' % (name, num)
            last = vxm.networks._conv_block(last, nf, name=layer_name, hyp_tensor=hyp_tensor)

        super().__init__(inputs=model_inputs, outputs=last, name=name)


def _upsample_block(x, factor=2, name=None):
    """
    Specific upsampling and concatenation layer for unet.
    """
    ndims = len(x.get_shape()) - 2
    assert ndims in (1, 2, 3), 'ndims should be one of 1, 2, or 3. found: %d' % ndims
    UpSampling = getattr(KL, 'UpSampling%dD' % ndims)

    return UpSampling(size=(factor,) * ndims, name=name)(x)
