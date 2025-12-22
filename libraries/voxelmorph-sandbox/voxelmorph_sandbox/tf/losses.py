# built in
import os

# third party
import tensorflow as tf
import tensorflow.keras.backend as K


class WeightedIntensityDiv(object):
    """
    Weighted Intentity Divergence.    
    """

    def __init__(self, alpha=2, model=None, mask=None, loss_mult=1):
        self.alpha = 1
        self.model = model

        self.mask = None
        if mask is not None:
            self.mask = K.variable(mask)

        self.loss_mult = loss_mult

    def weighted_intensity_div(self, y_true, y_pred):
        """
        weighted intensity divergence: 

            gx = spatial gradient of y_true
            c = abs(y_true - y_pred), cost is:
            gc = spatial gradient of c

            exp(-a * gx) * (gc**2)

        c measures the difference in the images, and gc measures gradient of these differences
        When neighbors in gx are close in intensity, gx ~ 0 -> first time is high (close to 1)
        and therefore when gx ~= 0, differences in the two images should be similar for 
        spatial neighbors (note: it doesn't matter too much *what* the differences are, just 
        that the spatial gradient is low) 

        When neightbors in gx are not close in intensity, gx ~ 1 -> first term is lower, 
        so the second term matters less

        Note: this is not for color!
        """

        # necessary spatial gradients
        if self.mask is None:
            y_true_spgrad = self._abs_grad(y_true)
            y_change_spgrad = self._abs_grad(y_true - y_pred)
        else:
            y_true_spgrad = self._abs_grad(y_true * self.mask)
            y_change_spgrad = self._abs_grad((y_true - y_pred) * self.mask)

        # sum over directions
        s = 0
        for d in range(len(y_true_spgrad)):
            wt = K.exp(-self.alpha * y_true_spgrad[d])
            q = wt * (y_change_spgrad[d] ** 2)
            # if self.mask is not None:
            #     # print(q.get_shape(), self.mask.get_shape())
            #     q = q * self.mask
            s += tf.reduce_mean(q)

        return s / len(y_true_spgrad) * self.loss_mult

    def _abs_grad(self, y):
        if len(y.get_shape()) == 5:
            # need to separate out mask from y_pred
            dy = tf.abs(y[:, 1:, :, :, :] - y[:, :-1, :, :, :])
            dx = tf.abs(y[:, :, 1:, :, :] - y[:, :, :-1, :, :])
            dz = tf.abs(y[:, :, :, 1:, :] - y[:, :, :, :-1, :])
            return (dy, dx, dz)

        elif len(y.get_shape()) == 4:
            dy = tf.abs(y[:, 1:, :, :] - y[:, :-1, :, :])
            dx = tf.abs(y[:, :, 1:, :] - y[:, :, :-1, :])
            return (dy, dx)

        else:
            raise ValueError(
                'incorrect number of dimensions, currently only implemented for 2D and 3D')


class LNMI:

    def __init__(self, bin_centers, vol_size,
                 sigma_ratio=0.5, max_clip=1, crop_background=False, patch_size=1):
        """
        Local mutual information loss for image-image pairs.
        Author: Courtney Guo
        """
        print("vxm info: local mutual information loss is experimental", file=sys.stderr)
        self.vol_size = vol_size
        self.max_clip = max_clip
        self.patch_size = patch_size
        self.crop_background = crop_background
        self.vol_bin_centers = K.variable(bin_centers)
        self.num_bins = len(bin_centers)
        self.sigma = np.mean(np.diff(bin_centers)) * sigma_ratio
        self.preterm = K.variable(1 / (2 * np.square(self.sigma)))

    def local_mi(self, y_true, y_pred):
        # reshape bin centers to be (1, 1, B)
        o = [1, 1, 1, 1, self.num_bins]
        vbc = K.reshape(self.vol_bin_centers, o)

        # compute padding sizes
        patch_size = self.patch_size
        x, y, z = self.vol_size
        x_r = -x % patch_size
        y_r = -y % patch_size
        z_r = -z % patch_size
        pad_dims = [[0, 0]]
        pad_dims.append([x_r // 2, x_r - x_r // 2])
        pad_dims.append([y_r // 2, y_r - y_r // 2])
        pad_dims.append([z_r // 2, z_r - z_r // 2])
        pad_dims.append([0, 0])
        padding = tf.constant(pad_dims)

        # compute image terms
        # num channels of y_true and y_pred must be 1
        I_a = K.exp(- self.preterm * K.square(tf.pad(y_true, padding, 'CONSTANT') - vbc))
        I_a /= K.sum(I_a, -1, keepdims=True)

        I_b = K.exp(- self.preterm * K.square(tf.pad(y_pred, padding, 'CONSTANT') - vbc))
        I_b /= K.sum(I_b, -1, keepdims=True)

        I_a_patch = tf.reshape(I_a, [(x + x_r) // patch_size, patch_size, (y + y_r) //
                                     patch_size, patch_size, (z + z_r) // patch_size, patch_size,
                                     self.num_bins])
        I_a_patch = tf.transpose(I_a_patch, [0, 2, 4, 1, 3, 5, 6])
        I_a_patch = tf.reshape(I_a_patch, [-1, patch_size**3, self.num_bins])

        I_b_patch = tf.reshape(I_b, [(x + x_r) // patch_size, patch_size, (y + y_r) //
                                     patch_size, patch_size, (z + z_r) // patch_size, patch_size,
                                     self.num_bins])
        I_b_patch = tf.transpose(I_b_patch, [0, 2, 4, 1, 3, 5, 6])
        I_b_patch = tf.reshape(I_b_patch, [-1, patch_size**3, self.num_bins])

        # compute probabilities
        I_a_permute = K.permute_dimensions(I_a_patch, (0, 2, 1))
        # should be the right size now, nb_labels x nb_bins
        pab = K.batch_dot(I_a_permute, I_b_patch)
        pab /= patch_size**3
        pa = tf.reduce_mean(I_a_patch, 1, keepdims=True)
        pb = tf.reduce_mean(I_b_patch, 1, keepdims=True)

        papb = K.batch_dot(K.permute_dimensions(pa, (0, 2, 1)), pb) + K.epsilon()
        return K.mean(K.sum(K.sum(pab * K.log(pab / papb + K.epsilon()), 1), 1))

    def loss(self, y_true, y_pred):
        y_pred = K.clip(y_pred, 0, self.max_clip)
        y_true = K.clip(y_true, 0, self.max_clip)
        return -self.local_mi(y_true, y_pred)
