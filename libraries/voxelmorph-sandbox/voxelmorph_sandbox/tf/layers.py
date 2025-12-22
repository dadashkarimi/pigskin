# python built-in imports

# third party imports
import tensorflow as tf
from tensorflow.keras import layers as KL
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
import numpy as np

# local imports
import neurite as ne
import neurite_sandbox as nes
import neurite as ne
import voxelmorph as vxm
import voxelmorph_sandbox as vxms


class RandomSpatialTransform(Layer):
    """
    Random Spatial Transform layer.
    This layer warps each channel in each sample in the input using a random,
        smoothed deformation field.
    Deformations fields are unique for each sample and for each channel.
    """

    def __init__(self,
                 amp_sigma,
                 smooth_sigma,
                 amp_mean=0.0,
                 deform_prediction=False,
                 batch_size=None,
                 **kwargs):
        """
        supply batch_size for faster operation, but then the model has to be used *only* with that
        batch size
        Parameters:
            amp_sigma: Standard deviation of normal distribution that deformation fields are sampled
                        from.
            smooth_sigma: Standard deviation of Gaussian kernel used to smooth deformation fields.
            amp_mean: Mean of normal distribution that deformation fields are smapled from.
            deform_prediction: whether to use spatial deformations during model testing/evaluation.
            batch_size (optional): Number of samples in an input batch. Needs to be fixed during
                initialization of layer because is used by reshape function called when deformation
                fields are smoothed.
        """
        self.amp_sigma = amp_sigma
        self.amp_mean = amp_mean
        self.smooth_sigma = smooth_sigma
        self.deform_prediction = deform_prediction
        self.batch_size = batch_size

        super(RandomSpatialTransform, self).__init__(**kwargs)

    def build(self, input_shape):
        self.built = True
        super(RandomSpatialTransform, self).build(input_shape)

    def call(self, x, training=None, return_field=False):
        """
        Parameters:
            x (Tensor): input Tensor [N, *vol_shape, C]
            training: whether call to layer is occurring during  model training or evaluation.
                Used to determine whether to apply transforms if self.deform_prediction is False.
            return_field bool: whether to return the deformation field
                in addition to the transformed input
        """
        if training is None:
            training = K.learning_phase()

        if self.batch_size is not None:
            """
            AVD Note: this **sometimes** fails in a keras model call (e.g. model.fit) because
            tf.shape(x)[0] actually becomes self.batch_size regardless of the data! I'm not sure, 
            this feels like a bug in keras, somehow x's shape gets to be self.batch_size, probably 
            having to do  with the operations we do later on. Also only happens sometimes, not 
            always?
            In those cases, tf.print(tf.shape(x)[0]) also prints self.batch_size in eager calls.
            """
            tf.debugging.assert_equal(tf.shape(x)[0],
                                      self.batch_size,
                                      message='RandomSpatialTransform: Tensor has wrong batch size '
                                      '{} instead of {}'.format(tf.shape(x)[0], self.batch_size))

        if self.deform_prediction:
            output = self.deform_random(x, return_field)
        else:
            id = array_ops.identity(x)
            output = training * self.deform_random(x, return_field) + (1 - training) * id

        return output

    def deform_random(self, x, return_field):
        """ Deform Tensor x of size [N, *vol_shape, C] with N x C random deformation fields
        Args:
            x (Tensor): Tensor to deform
            return_field (bool): whether to return the deformation fields themselves
                of size [N, *vol_shape, C, D]
        Returns:
            Tensor: deformed Tensor of the same size as x
        """
        vol_shape = tf.shape(x)
        ndim = tf.shape(vol_shape[1:-1])

        # will be N x *vol_shape x C x ndim
        field_shape = tf.concat([vol_shape, ndim], axis=0)

        # generate field for each batch too!
        # TODO: maybe use vxms.utils.batch_random_warp?
        field_rnd = tf.random.normal(shape=field_shape, mean=self.amp_mean, stddev=self.amp_sigma)

        if self.amp_sigma:
            field_blur = self.smooth_fields(field_rnd)
        else:
            field_blur = field_rnd

        # transform
        if self.batch_size is None:
            single_trf_fcn = lambda x: vxm.utils.transform(*x)
            moved_x = tf.map_fn(single_trf_fcn, [x, field_blur], dtype=tf.float32)
        else:
            moved_x = vxm.utils.batch_transform(x, field_blur, batch_size=self.batch_size)

        if return_field:
            # Return the field as well as the output (need this for making visualizations)
            return moved_x, field_blur
        else:
            return moved_x

    def smooth_fields(self, x):
        """ spatially smooth data of size [B, *vol_shape, C, D],
        where N is batch size, C is channels, D is volume dims (1, 2 or 3)
        TODO: might want to clean up or make sure it's consistent with ne.layers.GaussianBlur
        Args:
            x (Tensor): Tensor of size [B, *vol_shape, C, D]
        Returns:
            Tensor: spatially smoothed version of Tensor
        """
        # collapse to [B, *vol_shape, C * D]
        x_shape = K.shape(x)
        x_reshaped = K.reshape(x, tf.concat([x_shape[:-2], [-1]], 0))

        # use batch gaussian smoothing utility
        x_blur = nes.utils.batch_gaussian_smoothing(x_reshaped, self.smooth_sigma)

        # reshape back to [B, *vol_shape, C, D]
        output = K.reshape(x_blur, x_shape)

        # normalized output
        normalized_output = output * self.amp_sigma / tf.math.reduce_std(
            output, axis=range(len(x_shape) - 1), keepdims=True)

        return normalized_output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'amp_sigma': self.amp_sigma,
            'amp_mean': self.amp_mean,
            'smooth_sigma': self.smooth_sigma,
            'deform_prediction': self.deform_prediction
        })
        return config


class ApplyB0Warp(Layer):
    """
    layer to apply a B0 distortion (or corrrection) to an image
    """

    def __init__(self, fill_value=0, **kwargs):
        self.fill_value = fill_value
        super(ApplyB0Warp, self).__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def compute_output_shape(self, input_shape, **kwargs):
        return (input_shape[0])

    def call(self, image):
        """
        Parameters
            trf: affine transform either as a matrix with shape (N, N + 1)
            or a flattened vector with shape (N * (N + 1))
        """
        transformed_image = vxm.layers.SpatialTransformer(
            name='B0_warped', fill_value=0)([image, self.warp[tf.newaxis]])
        return transformed_image

    def _apply_xform(self, image):
        transformed_image = vxm.layers.SpatialTransformer(
            name='B0_warped', fill_value=0)([image, self.warp])
        return transformed_image


class DrawB0Warps(Layer):
    """
    layer to draw a random B0 susceptibility map  and a pair of directions, then create
    a distortion map in those directions
    """

    def __init__(self, inshape, min_blur=0.25, max_blur=20, max_std=1, **kwargs):
        self.inshape = inshape
        self.min_blur = min_blur
        self.max_blur = max_blur
        self.max_std = max_std
        super(DrawB0Warps, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'min_blur': self.min_blur,
            'max_blur': self.max_blur,
            'max_std': self.max_std
        })
        return config

    def build(self, inshape):
        super().build(inshape)

    def compute_output_shape(self, input_shape, **kwargs):
        return (input_shape[0])

    def call(self, B0_dirs):
        """
        Parameters
        [B0_dir1, B0_dir2]
        """
        norms = tf.map_fn(
            self._draw_single_xform,
            B0_dirs[0],
            fn_output_signature=tf.float32)

        warps1 = tf.map_fn(
            self._project_onto_direction,
            [B0_dirs[0], norms],
            fn_output_signature=tf.float32)
        warps2 = tf.map_fn(
            self._project_onto_direction,
            [B0_dirs[1], norms],
            fn_output_signature=tf.float32)

        return warps1, warps2

    def _project_onto_direction(self, inputs):
        B0_dir = inputs[0]
        norm = inputs[1]
        vec_field = norm[..., tf.newaxis] * tf.transpose(B0_dir)
        return vec_field

    def _draw_single_xform(self, B0_dirs):
        warp = vxms.utils.augment.draw_b0_distortion(
            self.inshape,
            read_dir=B0_dirs[0],
            max_std=self.max_std, min_blur=self.min_blur, max_blur=self.max_blur)
        return tf.linalg.norm(warp, axis=-1)


class DrawB0Map(Layer):
    """
    layer to draw a random B0 susceptibility map and return it
    """

    def __init__(self, inshape,
                 min_blur=0.25,
                 max_blur=20,
                 max_std=1,
                 low_B0_pct=.8,
                 low_B0_std=.5,
                 **kwargs):

        self.inshape = inshape
        self.min_blur = min_blur
        self.max_blur = max_blur
        self.max_std = max_std
        self.low_B0_pct = low_B0_pct
        self.low_B0_std = low_B0_std
        super(DrawB0Map, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'min_blur': self.min_blur,
            'max_blur': self.max_blur,
            'max_std': self.max_std,
            'low_B0_pct': self.low_B0_pct,
            'low_B0_std': self.low_B0_std
        })
        return config

    def build(self, inshape):
        super().build(inshape)

    def compute_output_shape(self, input_shape, **kwargs):
        return (input_shape[0])

    def call(self, B0_dirs):
        """
        Parameters
        """
        norms = tf.map_fn(
            self._draw_single_xform,
            B0_dirs,
            fn_output_signature=tf.float32)

        return norms

    def _draw_single_xform(self, B0_dirs):
        B0_map = vxms.utils.augment.draw_B0_map(
            self.inshape,
            max_std=self.max_std, min_blur=self.min_blur, max_blur=self.max_blur,
            low_B0_pct=self.low_B0_pct, low_B0_std=self.low_B0_std)
        return B0_map


class MidspaceTransforms(Layer):
    """
    layer to compute a pair of midspace transforms from a single svf
    """

    def __init__(self, int_steps=7, **kwargs):
        self.int_steps = int_steps
        super(MidspaceTransforms, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'int_steps': self.int_steps
        })
        return config

    def build(self, inshape):
        super().build(inshape)

    def compute_output_shape(self, input_shape, **kwargs):
        return ((2,) + input_shape)

    def call(self, svf):
        """
        Parameters
        svf - the svf to compute the fwd and inv midspaces from
        """
        svf = KL.Lambda(lambda x: x / 2)(svf)
        fwd_warp = vxm.layers.VecInt(int_steps=self.int_steps)(svf)
        neg_svf = ne.layers.Negate()(svf)
        inv_warp = vxm.layers.VecInt(int_steps=self.int_steps)(neg_svf)

        return fwd_warp, inv_warp


class MidspaceTransform(Layer):
    """
    layer to compute a matrix sqrt, adding and subtracting the identity as needed. 
    Only for affine right now
    """

    def __init__(self, add_identity=True, **kwargs):

        self.add_identity = True
        super(MidspaceTransform, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'add_identity': self.add_identity
        })
        return config

    def build(self, inshape):
        super().build(inshape)

    def compute_output_shape(self, input_shape, **kwargs):
        return (input_shape)

    def call(self, trf):
        """
        Parameters
        trf - the transform to compute the midspace/sqrt from
        """
        norms = tf.map_fn(
            self._compute_single_midspace,
            trf,
            fn_output_signature=tf.float32)

        return norms

    def _compute_single_midspace(self, trf):
        inshape = trf.get_shape().as_list()
        rows = inshape[0]
        cols = inshape[-1]
        if rows == 12 or (rows == 3 and cols == 4):
            ndims = 3
            if rows == 12:
                trf = tf.reshape(trf, [3, 4])
        elif (rows == 2 and (cols == 2 or cols == 3)) or rows == 6:
            ndims = 2
            if rows * cols == 6:
                trf = tf.reshape(trf, [2, 3])
        else:
            assert 0, f'MidspaceTransform: input shape of unknown dimension {inshape}'

        row_aug = tf.reshape(
            tf.convert_to_tensor(ndims * [0.] + [1.], dtype=tf.float32), (1, ndims + 1))
        sqrt_affine = tf.linalg.sqrtm(tf.concat([trf, row_aug], axis=0))
        return sqrt_affine[:ndims, :]


class AffineAugmentWithB0(Layer):
    """
    layer to apply affine augmentation and B0 distortion to an image
    """

    def __init__(self, interp_method='linear', fill_value=0, **kwargs):
        self.fill_value = fill_value
        self.interp_method = interp_method
        super(AffineAugmentWithB0, self).__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def compute_output_shape(self, input_shape, **kwargs):
        return (input_shape[0])

    def call(self, inputs):
        """
        Parameters
           [[input image list], [B0_dir list]]
        take a list of images and B0_dirs and draws and applies the warps
        """
        im_input = inputs[0]
        B0_dir = inputs[1]

        transformed_images = tf.map_fn(
            self._xform_single_image, im_input, dtype=tf.float32)
        distorted_images = tf.map_fn(
            self._distort_single_image, [transformed_images, B0_dir], dtype=tf.float32)
        return [transformed_images, distorted_images]

    def _xform_single_image(self, im_input):
        inshape = im_input.get_shape()[:-1].as_list()
        mat = vxms.utils.augment.draw_affine_matrix(
            shift=[30, 30, 30], rotate=[10, 10, 10])
        im_transformed = vxm.layers.SpatialTransformer(
            fill_value=self.fill_value,
            interp_method=interp)(
                [im_input[tf.newaxis], mat[tf.newaxis]])

        return im_transformed[0, ...]

    def _distort_single_image(self, inputs):
        im_transformed = inputs[0]
        B0_dir = inputs[1]
        inshape = im_transformed.get_shape()[:-1].as_list()
        B0_warp = vxms.utils.augment.draw_b0_distortion(
            inshape, B0_dir, max_std=2)
        distorted_image = vxm.layers.SpatialTransformer(
            fill_value=self.fill_value)(
                [im_transformed[tf.newaxis], B0_warp[tf.newaxis]])

        return distorted_image[0, ...]


class AffineAugment(Layer):
    """
    layer to apply affine augmentation to a list of images
    if limit_to_fov is True the transform will be limited to prevent cropping
    of the bounding box of the first input image.

    TO-DO: support add_identity and shift_center for the limit_to_fov
    parameters:
            interp_method='linear',
            max_trans=20,              - max translation for random draw
            max_rot=10,                - max rotation for random draw
            fill_value=0,
            return_mats=False,         - whether to return the random affine or not
            limit_to_fov=False,        - limit the affines so that image doesn't leave fov
      TODO only rigid for now - need to extend for full affine.
    """

    def __init__(
            self,
            interp_method='linear',
            max_trans=20,
            max_rot=10,
            max_scale=None,
            max_shear=None,
            fill_value=0,
            return_mats=False,
            limit_to_fov=False,
            **kwargs):
        self.fill_value = fill_value
        self.interp_method = interp_method
        self.max_scale = max_scale
        self.max_shear = max_shear
        self.max_trans = max_trans
        self.return_mats = return_mats
        self.limit_to_fov = limit_to_fov
        self.max_rot = max_rot
        super(AffineAugment, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'fill_value': self.fill_value,
            'interp_method': self.interp_method,
            'return_mats': self.return_mats,
            'max_trans': self.max_trans,
            'max_rot': self.max_rot,
            'limit_to_fov': self.limit_to_fov
        })
        return config

    def build(self, input_shape):
        super().build(input_shape)

    def compute_output_shape(self, input_shape, **kwargs):
        return (input_shape[0])

    def call(self, inputs):
        """
        Parameters
           [input image1, input image2, ...]
        """
        if type(inputs) is list == 0:
            inputs = [inputs]

        self.mats = tf.map_fn(
            self._create_single_xform,
            inputs[0],
            fn_output_signature=tf.float32)

        outputs = []
        for inp in inputs:
            outp = tf.map_fn(
                self._xform_single_image,
                [inp, self.mats],
                fn_output_signature=tf.float32)
            outputs.append(outp)

        outputs += [self.mats]

        return outputs

    def _xform_single_image(self, inputs):
        im_input = inputs[0]
        mat = inputs[1]

        # this is a bit of a hack to allow nearest for label maps
        # have to change the output type to be consistent
        # (TO DO: allow user to pass a list of output types)
        if im_input.dtype == tf.int32:
            im_input = tf.cast(im_input, tf.float32)
            interp_method = 'nearest'
        else:
            interp_method = 'linear'
        inshape = im_input.get_shape()[:-1].as_list()
        im_transformed = vxm.layers.SpatialTransformer(
            fill_value=self.fill_value,
            interp_method=interp_method)(
                [im_input[tf.newaxis], mat[tf.newaxis]])
        return im_transformed[0, ...]

    def _create_single_xform(self, im_input):
        inshape = im_input.get_shape().as_list()
        ndims = len(inshape) - 1
        shift = ndims * [self.max_trans]
        rot = [self.max_rot] if ndims == 2 else 3 * [self.max_rot]
        mat = vxms.utils.augment.draw_affine_matrix(shift=shift, rotate=rot,
                                                    scale=self.max_scale, shear=self.max_shear,
                                                    is_2d=(ndims == 2))

        if self.limit_to_fov:
            pad = ([[self.max_trans, self.max_trans]] * ndims) + [[0, 0]]
            im_padded = tf.pad(im_input, pad)
            # , add_identity=False taken out
            im_trans = vxm.layers.SpatialTransformer(
                fill_value=0, interp_method='nearest')(
                    [im_padded[tf.newaxis], mat[tf.newaxis]])[0]

            bbox = vxms.utils.utils.bboxND(im_trans[..., 0])
            pad_offset = tf.transpose(tf.convert_to_tensor(
                4 * (ndims - 1) * [ndims * [self.max_trans] + [0]],
                dtype=tf.float32))
            bbox -= pad_offset  # remove the padding

            # compute the translations that will bring the object back into the fov
            trans = []
            for ind in range(ndims):
                if tf.reduce_max(bbox[ind]) > (inshape[ind] - 1):
                    tx = tf.reduce_max(bbox[ind]) - (inshape[ind] - 1)
                elif tf.reduce_min(bbox[ind]) < 0:
                    tx = tf.reduce_min(bbox[ind])
                else:
                    tx = 0.0
                trans.append(tx)

            if ndims == 3:
                mat_trans_into_fov = tf.convert_to_tensor([
                    [1, 0, 0, trans[0]],
                    [0, 1, 0, trans[1]],
                    [0, 0, 1, trans[2]],
                    [0, 0, 0, 1]
                ])
                row = tf.reshape(tf.convert_to_tensor([0, 0, 0, 1], dtype=tf.float32), (1, 4))
            else:
                mat_trans_into_fov = tf.convert_to_tensor([
                    [1., 0., trans[0]],
                    [0., 1., trans[1]],
                    [0., 0., 1]
                ])
                row = tf.reshape(tf.convert_to_tensor([0, 0, 1], dtype=tf.float32), (1, 3))

            mat_aug = tf.concat([mat, row], axis=0)
            mat = (mat_aug @ mat_trans_into_fov)[0:ndims, :]

        return mat


class Mask(Layer):
    ''' mask the input layer with a random mask with a specified % off '''

    def __init__(self, zero_pct=.1, **kwargs):
        super(Mask, self).__init__(**kwargs)
        self.zero_pct = zero_pct

    def get_config(self):
        config = super(Mask, self).get_config()

        # Specify here all the values for the constructor's parameters
        config['zero_pct'] = self.zero_pct

        return config

    def build(self, input_shape):
        super(Mask, self).build(input_shape)
        self._create_mask(input_shape)
        # Create a non-trainable weight.

    def call(self, x):
        super(Mask, self).call(x)
        if not hasattr(self, 'mask'):
            input_shape = x.get_shape().as_list()[1:]
            self._create_mask(input_shape)

        output = x * self.mask
        return output

    def compute_output_shape(self, input_shape):
        return self.inputs_shape

    def _create_mask(self, input_shape):
        num_samples = np.prod(np.array(input_shape))
        mask_initializer = tf.cast(
            tf.random.categorical(
                tf.math.log([[self.zero_pct, 1 - self.zero_pct]]),
                num_samples,
                dtype=tf.int32),
            tf.float32)
        self.mask = self.add_weight(name='mask',
                                    shape=(num_samples,),
                                    initializer='zeros',
                                    trainable=False)
        mask_initializer = np.random.choice(
            a=[0, 1],
            size=(num_samples,),
            p=[self.zero_pct, 1 - self.zero_pct])
        self.set_weights([mask_initializer])


class B0Augment(Layer):
    """
    layer to draw and apply a B0 distortion to an image
    """

    def __init__(self, interp_method='linear',
                 max_std=2,
                 fill_value=0,
                 min_blur=8,
                 max_blur=32,
                 **kwargs):
        self.fill_value = fill_value
        self.max_std = max_std
        self.min_blur = min_blur
        self.max_blur = max_blur
        self.interp_method = interp_method
        super(B0Augment, self).__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def compute_output_shape(self, input_shape, **kwargs):
        return (input_shape[0])

    def call(self, inputs):
        """
        Parameters
           [input image, B0_dir]

        """
        im_input = inputs[0]
        B0_dir = inputs[1]
        distorted_images = tf.map_fn(
            self._distort_single_image,
            [im_input, B0_dir],
            fn_output_signature=im_input.dtype)
        return distorted_images

    def _distort_single_image(self, inputs):
        im_transformed = inputs[0]
        B0_dir = inputs[1]
        inshape = im_transformed.get_shape()[:-1].as_list()
        B0_warp = vxms.utils.augment.draw_b0_distortion(
            inshape,
            B0_dir,
            max_std=self.max_std,
            min_blur=self.min_blur,
            max_blur=self.max_blur)
        distorted_image = vxm.layers.SpatialTransformer(
            fill_value=self.fill_value)(
                [im_transformed[tf.newaxis], B0_warp[tf.newaxis]])

        return distorted_image[0, ...]


class ResizeLabels(Layer):
    """
    layer to draw and apply random atrophy/growth of labels to a label map
    """

    def __init__(self, structure_list, subsample_atrophy=1, **kwargs):
        self.structure_list = structure_list
        self.subsample_atrophy = subsample_atrophy
        super(ResizeLabels, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ResizeLabels, self).build(input_shape)

    def compute_output_shape(self, input_shape, **kwargs):
        return (input_shape[0])

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'structure_list': self.structure_list,
            'subsample_atrophy': self.subsample_atrophy
        })
        return config

    def call(self, inputs):
        """
        Parameters
           [input image, B0_dir]

        """
        if len(self.structure_list) == 0:
            return inputs

        label_input = inputs
        atrophy_images = tf.map_fn(
            self._resize_labels_single_image, label_input,
            fn_output_signature=label_input.dtype)
        return atrophy_images

    def _resize_labels_single_image(self, inputs):
        label_input = inputs
        structure_list = self.structure_list
        inshape = label_input.get_shape()[:-1].as_list()
        resized_labels = vxms.utils.augment.resize_labels(
            label_input, structure_list, subsample_atrophy=self.subsample_atrophy)

        return resized_labels


class AddLesions(Layer):
    """
    layer to randomly add lesion labels to the interior of the wm
    """

    def __init__(self,
                 lesion_label=77,
                 unique_label=100,
                 insert_labels=[2, 41],
                 interior_dist=3,
                 max_labels=4,
                 max_label_vol=20,
                 **kwargs):

        self.lesion_label = lesion_label
        self.unique_label = unique_label
        self.insert_labels = insert_labels
        self.interior_dist = interior_dist
        self.max_labels = max_labels
        self.max_label_vol = max_label_vol
        super(AddLesions, self).__init__(**kwargs)

    def build(self, input_shape):
        super(AddLesions, self).build(input_shape)

    def compute_output_shape(self, input_shape, **kwargs):
        return (input_shape[0])

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'lesion_label': self.lesion_label,
            'unique_label': self.unique_label,
            'interior_dist': self.interior_dist,
            'insert_labels': self.insert_labels,
            'max_label_vol': self.max_label_vol,
        })
        return config

    def call(self, inputs):
        """
        Parameters

        """

        label_input = inputs
        lesion_images = tf.map_fn(
            self._add_lesions_single_image, label_input,
            fn_output_signature=label_input.dtype)
        return lesion_images

    def _add_lesions_single_image(self, inputs):
        label_input = inputs
        inshape = label_input.get_shape()[:-1].as_list()
        lesion_image = vxms.utils.augment.insert_lesions(
            label_input, self.lesion_label, self.unique_label,
            self.insert_labels, self.interior_dist, self.max_labels, self.max_label_vol)

        return lesion_image


class InvertSingleDirection(Layer):
    """
    Inverts an affine transform. The transform must represent
    the shift between images (not over the identity).
    """

    #    def build(self, input_shape):
    #        self.ndims = extract_affine_ndims(input_shape[1:])
    #        super().build(input_shape)

    def compute_output_shape(self, input_shape, **kwargs):
        return (input_shape[0], self.ndims * (self.ndims + 1))

    def call(self, inputs):
        """
        Parameters
            trf: affine transform either as a matrix with shape (N, N + 1)
            or a flattened vector with shape (N * (N + 1))
        """
        return tf.map_fn(self._single_invert, inputs, dtype=tf.float32)

    def _single_invert(self, inputs):
        mag_field = inputs[0]
        direction = inputs[1]
        warp = mag_field @  direction
        mf1 = mag_field[tf.newaxis, ...]
        w1 = warp[tf.newaxis, ...]
        mag_field_warped = vxm.layers.SpatialTransformer(fill_value=0)([mf1, w1])[0, ...]
        inverse_transform = -mag_field_warped @ direction
        return inverse_transform


class TensorToAffineMatrix(Layer):
    """
    Dense layer to predict an affine matrix from an arbitrary tensor.
    """

    def __init__(self,
                 ndims=3,
                 transform_type='affine',
                 predict_matrix_params=False,
                 rescale_translations=1.0,
                 rescale_non_translations=1.0,
                 use_deg=False,
                 kernel_initializer=None,
                 **kwargs):
        """
        Parameters:
            ndims: Number of dimensions for transform.
            transform_type: 'affine', 'rigid', or 'shearless'.
            predict_matrix_params: Directly predict matrix, not affine parameters.
            rescale_translations: Rescale predicted translation parameters.
            rescale_non_translations: Rescale predicted non-translation parameters.
            kernel_initializer: If None, defaults to random uniform between +/- 1e-5.
        """

        self.use_deg = use_deg
        self.ndims = ndims
        self.transform_type = transform_type
        self.predict_matrix_params = predict_matrix_params
        self.rescale_translations = rescale_translations
        self.rescale_non_translations = rescale_non_translations
        self.kernel_initializer = kernel_initializer
        if self.kernel_initializer is None:
            self.kernel_initializer = tf.keras.initializers.RandomUniform(minval=-1e-5, maxval=1e-5)
        super().__init__(**kwargs)

    def build(self, input_shape):

        if self.predict_matrix_params:

            if self.transform_type != 'affine':
                raise ValueError('transform_type must be affine for predict_matrix_params.')

            self.affine_converter = vxm.layers.AddIdentity()

            nb_params = self.ndims * (self.ndims + 1)

            self.scaling = np.ones((self.ndims, self.ndims + 1))
            self.scaling[:, -1] = self.rescale_translations
            self.scaling[:, :-1] = self.rescale_non_translations

        else:
            self.affine_converter = vxm.layers.ParamsToAffineMatrix(ndims=self.ndims,
                                                                    deg=self.use_deg,
                                                                    shift_scale=True)
            if self.transform_type == 'rigid':
                nb_params = 6 if self.ndims == 3 else 3
            elif self.transform_type in ('shearless', 'rigid+scale'):
                nb_params = 9 if self.ndims == 3 else 5
            elif self.transform_type == 'affine':
                nb_params = 12 if self.ndims == 3 else 6
            elif self.transform_type == 'trans':
                nb_params = 3 if self.ndims == 3 else 2
            else:
                raise ValueError('transform_type must be rigid, trans, shearless, or affine.')

            self.scaling = np.ones(nb_params)
            self.scaling[:self.ndims] = self.rescale_translations
            self.scaling[self.ndims:] = self.rescale_non_translations

        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(nb_params, kernel_initializer=self.kernel_initializer)

    def compute_output_shape(self, input_shape, **kwargs):
        return (input_shape[0], self.ndims, self.ndims + 1)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'ndims': self.ndims,
            'use_deg': self.use_deg,
            'transform_type': self.transform_type,
            'predict_matrix_params': self.predict_matrix_params,
            'rescale_translations': self.rescale_translations,
            'rescale_non_translations': self.rescale_non_translations,
            'kernel_initializer': self.kernel_initializer,
            'scaling': self.scaling,
            'flatten': self.flatten,
            'dense': self.dense,
            'affine_converter': self.affine_converter,
        })
        return config

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense(x)
        if self.predict_matrix_params:
            x = tf.reshape(x, (x.shape[0], self.ndims, self.ndims + 1))
        x = x * tf.convert_to_tensor(self.scaling, dtype='float32')
        x = self.affine_converter(x)
        return x


class RigidMotionSynth(Layer):
    """
    [Experimental] rigid motion simulation.
    Takes in an image-space scan, and simulates k-space motion. uses
        vxms.utils.simulate_rigid_motion

    Author:
        adalca
    """

    def __init__(self,
                 max_shift,
                 max_rotate,
                 nb_motion,
                 shuffle_lines=False,
                 kspace_min_perc=0.5,
                 kspace_max_perc=0.75,
                 line_axis=0,
                 **kwargs):
        self.max_shift = max_shift
        self.max_rotate = max_rotate
        self.nb_motion = nb_motion
        self.shuffle_lines = shuffle_lines
        self.kspace_min_perc = kspace_min_perc
        self.kspace_max_perc = kspace_max_perc
        self.line_axis = line_axis
        """
        Args:
            max_shift: the maximum shift in voxels per motion
            max_rotate: the maximum rotation in degrees per motion
            nb_motion: the number of motion estimates
            shuffle_lines: whether each 'segment' acquires random lines. This may be more realistic.
            kspace_min_perc: the minimum percentage of acquisitions to wait until the first motion 
                happens
            kspace_max_perc: the maximum percentage of acquisitions to wait until the last motion 
                happens

        TODO: keep track of affine motion used in a list of reference tensors
        """

        super().__init__(**kwargs)

    def build(self, input_shape):
        self.in_shape = input_shape[1:-1]
        self.in_shape = input_shape[1:-1]
        super().build(input_shape)

    def call(self, x):
        max_float = tf.cast(tf.shape(x)[1], tf.float32)
        minval = max_float * self.kspace_min_perc
        maxval = max_float * self.kspace_max_perc
        idx_replace = [tf.cast(tf.random.uniform((1,), minval=minval, maxval=maxval), tf.int32)
                       for _ in range(self.nb_motion)]
        image = vxms.utils.simulate_rigid_motion(x, self.max_shift, self.max_rotate, idx_replace,
                                                 line_axis=self.line_axis,
                                                 shuffle_lines=self.shuffle_lines)
        return tf.math.real(image)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'max_shift': self.max_shift,
            'max_rotate': self.max_rotate,
            'nb_motion': self.nb_motion,
            'shuffle_lines': self.shuffle_lines,
            'kspace_min_perc': self.kspace_min_perc,
            'kspace_max_perc': self.kspace_max_perc,
            'line_axis': self.line_axis,
        })
        return config


class SphericalLocalParamWithInput(ne.layers.LocalParamWithInput):
    """
    LocalParamWithInput layer with spherical unpadding and padding
    """
    def __init__(self, shape, initializer='RandomNormal', mult=1.0, pad_size=0, **kwargs):
        self.pad_size = pad_size
        super(SphericalLocalParamWithInput, self).__init__(shape, initializer, mult, **kwargs)

    def call(self, x):
        xslice = K.batch_flatten(x)[:, 0:1]
        b = xslice * tf.zeros((1,)) + tf.ones((1,))
        img = K.flatten(self.kernel * self.biasmult)[tf.newaxis, ...]
        y = K.reshape(K.dot(b, img), [-1, *self.shape])
        if self.pad_size > 0:
            y = nes.utils.unpad_2d_image(y, self.pad_size)
            y = nes.utils.pad_2d_image_spherically(y, self.pad_size)
            y.set_shape(self.compute_output_shape(x.shape))
        return y
