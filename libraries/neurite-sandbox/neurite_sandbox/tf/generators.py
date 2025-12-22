""" sandbox (experimental) generators """

# internal python imports
import sys

# third party
import numpy as np
import scipy
import tensorflow.keras.backend as K
import tensorflow as tf
try:  # tfp is tricky to install and is not required for most tools.
    import tensorflow_probability as tfp
except:
    print('Failed to import tensorflow_probability')


# local imports
import pystrum.pytools.timer as Timer
import pystrum.pynd.ndutils as nd

from . import utils


class WMHAugment(object):
    """
    generator for data augmentation for WMH hyperintensity
    """

    def __init__(self, gen, flipaxis=3, seg_blur_sigma=None, seg_vol_idx=None):
        # flipaxis default is for 3d, and flipping along third dimension
        self.gen = gen
        self.flipaxis = flipaxis
        self.sample = None
        self.flipsample = None
        self.seg_blur_sigma = seg_blur_sigma
        self.seg_vol_idx = seg_vol_idx

    def flip(self):
        while 1:
            q = next(self.gen)
            self.sample = q

            # flip with 50/50 probability
            self.flipsample = np.random.choice([True, False])
            if self.flipaxis is not None and self.flipsample:
                if isinstance(q, (tuple, list)):
                    q = tuple([np.flip(f, self.flipaxis) for f in q])
                else:
                    q = np.flip(q, self.flipaxis)

            if self.seg_blur_sigma is not None:
                if isinstance(q, (tuple, list)):
                    q = list(q)  # need to edit entry, can't be tuple
                assert self.seg_vol_idx is not None
                nd = q[self.seg_vol_idx].ndim - 2
                sigmas = [0.001, *[self.seg_blur_sigma] * nd, 0.001]
                q[self.seg_vol_idx] = scipy.ndimage.filters.gaussian_filter(
                    q[self.seg_vol_idx].astype(float), sigmas)

            if isinstance(q, (tuple, list)):
                q = tuple(q)
            yield q


class VolCount():
    """
    generate volume output
    """

    def __init__(self, gen, blur_sigma=None, prior_gen=False):
        self.gen = gen
        self.blur_sigma = blur_sigma
        self.prior_gen = prior_gen  # is this generator a prior?

    def vol_count(self):
        """ count volumes """
        while 1:
            # get new data
            self.sample = next(self.gen)
            if self.prior_gen:
                input_vol = self.sample[0]
            else:
                input_vol = self.sample
            input_shape = input_vol.shape
            ndm = input_vol.ndim - 1

            if self.blur_sigma is not None:
                shp = self.sample[1].shape

                # reshape output
                output_o = np.reshape(self.sample[1], [*input_shape, -1]).astype('float32')

                # blur volumes
                sigmas = [0.001, * [self.blur_sigma] * ndm, 0.001]
                output = scipy.ndimage.filters.gaussian_filter(output_o, sigmas)
                output = np.reshape(output, shp)

                output[:, 0] = self.sample[1][:, 0]  # keep background non-blurred

            else:
                output = self.sample[1]

            # sum amounts
            output_count = np.sum(output, 1)
            yield (self.sample[0], output_count)  # output


class ConditionalGenerator():
    """
    generate based on condition applied to sample
    """

    def __init__(self, gen, cond=None, verbose=True):
        self.g = gen
        self.cond = cond
        self.verbose = verbose

    def gen(self):
        """ generator """
        while 1:
            sample = next(self.g)
            if self.cond(sample):
                yield sample
            elif self.verbose:
                print('Condition failed. Resampling.')


class IAE_Generator():
    """
    Generator for I - AE model.
    """

    def __init__(self, g, blursigma=None):
        self.g = g
        self.blursigma = blursigma

    def train_gen(self):
        while 1:
            z = next(self.g)
            self.sample = z

            logseg = np.log(z[1] + np.finfo(float).eps)
            yield ([z[0], logseg], [logseg, logseg])

    def test_gen(self):
        while 1:
            z = next(self.g)
            self.sample = z

            logseg = np.log(z[1] + np.finfo(float).eps)
            yield (z[0], logseg)


class ChannelExtractor():

    def __init__(self, g, extract_labels=None, case_specific=None, prior=False):
        self.g = g
        self.extract_labels = extract_labels
        self.case_specific = case_specific
        self.prior = prior

    def gen(self):
        while 1:
            q = next(self.g)
            self.sample = q

            if self.case_specific is not None:
                if self.case_specific == 1:  # neuron seg
                    q = list(q)
                    if self.prior:
                        q[0][1] = np.take(q[0][1], self.extract_labels, -1)
                        q[0][1] = self._renorm(q[0][1])
                    q[1] = np.take(q[1], self.extract_labels, -1)
                    q[1] = self._renorm(q[1])

            elif self.extract_labels is not None:
                q = self._extr(q)
            yield q

    def _extr(self, q):
        if isinstance(q, (tuple, list)):
            q = [self._extr(f) for f in q]
        else:
            q = np.take(q, self.extract_labels, -1)
        return q

    def _renorm(self, q):
        """ TODO: should use something like q.put or w/e """
        if q.ndim == 5:
            q[:, :, :, :, 0] = 1 - np.sum(q[:, :, :, :, 1:], -1)
        elif q.ndim == 4:
            q[:, :, :, 0] = 1 - np.sum(q[:, :, :, 1:], -1)
        else:
            print(q.shape)
            raise ValueError("ChannelExtractor: unimplemented for low dims. Received: %d" % q.ndim)
        return q


class AE_Generator():

    def __init__(self,
                 g,  # generator
                 eps=np.finfo(float).eps,
                 extract_labels=None,
                 case_specific=None,
                 blursigma=None,
                 log_method=1,
                 do_3d=False,
                 forcefloat=False,
                 logprior_vol=None,  # logprior volume directly
                 enc_size=None,
                 do_blur_channel_norm=False,
                 do_blur_specific_channel=0,
                 do_blur_point_norm=True,
                 do_blur_point_norm_clip=1):

        # obvious params
        self.g = g
        self.eps = eps
        self.log_method = log_method
        self.enc_size = enc_size

        # label extraction - wrap passed generator into label extraction generator
        if extract_labels is not None:
            self.g = sb_gen.ChannelExtractor(g,
                                             extract_labels=extract_labels,
                                             case_specific=case_specific,
                                             prior=False).gen()

        # force float
        self.forcefloat = forcefloat
        self.do_3d = do_3d
        if not self.forcefloat:
            print('Warning: not forcing float', file=sys.stderr)

        # direct prior volume
        if logprior_vol is not None:
            msg = 'Using a manual prior, *slice*-wise generator.\n' + \
                'Are you sure you gave me a brand new generator?\n' + \
                'This seems much faster than actually using prior generator.'
            print(msg, file=sys.stderr)
            self.pi = 0
            ndims = np.ndim(logprior_vol)
            if ndims == 4 and not do_3d:
                self.reshape_logprior_vol = np.transpose(logprior_vol, [2, 0, 1, 3])
            else:
                self.reshape_logprior_vol = logprior_vol

        # blurring parameters
        self.blursigma = blursigma
        self.do_blur_channel_norm = do_blur_channel_norm  # normalize along 5th channel (channel 4)
        self.do_blur_point_norm = do_blur_point_norm  # normalize to gaussian height
        self.do_blur_point_norm_clip = do_blur_point_norm_clip
        self.do_blur_specific_channel = do_blur_specific_channel

    def gen(self, **yield_args):
        """ sample-sample auto-encoder generator """
        while 1:
            self._next_sample()
            yield self._yield(self.sample, self.sample, **yield_args)

    def log_sample_gen(self, **yield_args):
        """ logsample-sample auto-encoder generator """
        while 1:
            self._next_sample(do_log=True)
            yield self._yield(self.log_sample, self.sample, **yield_args)

    def sample_log_gen(self, **yield_args):
        """ sample-logsample auto-encoder generator """
        while 1:
            self._next_sample(do_log=True)
            yield self._yield(self.sample, self.log_sample, **yield_args)

    def prior_sample_log_gen(self, **yield_args):
        """ sample-logsample auto-encoder generator """
        while 1:
            self._next_sample(do_log=True)

            # prior
            nb_samples = self.sample.shape[0]
            if np.ndim(self.reshape_logprior_vol) == 4 and not self.do_3d:
                prior = self.reshape_logprior_vol[self.pi:(self.pi + nb_samples), :]
                self.pi = np.mod(self.pi + nb_samples, self.reshape_logprior_vol.shape[0])
            else:
                # TODO: this is probably inefficient...
                ndims = np.ndim(self.reshape_logprior_vol)
                prior = np.expand_dims(self.reshape_logprior_vol, 0)
                prior = np.tile(prior, [nb_samples] + [1] * (ndims))

            # output
            yield self._yield([self.sample, prior], self.log_sample, **yield_args)

    def prior_sample_gen(self, **yield_args):
        """ sample-logsample auto-encoder generator """
        while 1:
            self._next_sample(do_log=False)

            # prior
            nb_samples = self.sample.shape[0]
            if np.ndim(self.reshape_logprior_vol) == 4 and not self.do_3d:
                prior = self.reshape_logprior_vol[self.pi:(self.pi + nb_samples), :]
                self.pi = np.mod(self.pi + nb_samples, self.reshape_logprior_vol.shape[0])
            else:
                # TODO: this is probably inefficient...
                ndims = np.ndim(self.reshape_logprior_vol)
                prior = np.expand_dims(self.reshape_logprior_vol, 0)
                prior = np.tile(prior, [nb_samples] + [1] * (ndims))

            # yield
            yield self._yield([self.sample, prior], self.sample, **yield_args)

    def log_log_gen(self, **yield_args):
        """ logsample-logsample auto-encoder generator """
        while 1:
            self._next_sample(do_log=True)
            yield self._yield(self.log_sample, self.log_sample, **yield_args)

    def _yield(self, inp, out, nargout=None):
        """
        yield, but check for VAE/AE
        """
        if nargout is None and self.enc_size:
            nargout = 3
        elif nargout is None:
            nargout = 1

        if nargout > 1:
            assert self.enc_size is not None

        if nargout == 2:
            out = [out, self.mu]
        elif nargout == 3:
            out = [out, self.mu, self.sigma]

        return (inp, out)

    def _next_sample(self, do_log=False):
        """
        get a sample from the generator
        parameters:
            blur the sample, take the log...
            get the log of the sample.
        """

        # get sample
        self.sample = next(self.g)

        if self.enc_size is not None and all([f is not None for f in self.enc_size]):
            nb_samples = self.sample.shape[0]
            self.mu = np.zeros((nb_samples, *self.enc_size))
            self.sigma = np.zeros((nb_samples, *self.enc_size))

        # verify float
        if self.forcefloat:
            self.sample = self.sample.astype(float)

        # blur sample
        if self.blursigma is not None:
            self.orig_sample = self.sample
            self.sample = self._blur_sample(self.sample.astype(float))

        # take log
        if do_log > 0:
            self.log_sample = _log(self.sample, method=self.log_method)

            # blur the log sample
            if self.blursigma is not None:
                self.log_blur_sample = np.log(np.maximum(self.log_sample, np.finfo(float).eps))

    def _blur_sample(self, sample):
        # useful vars

        s = self.blursigma
        eps = np.finfo(float).eps
        ndims = sample.ndim - 2

        # blur
        if self.do_blur_specific_channel is not None:
            assert ndims == 3, "Only implemented this for 3d"
            c = self.do_blur_specific_channel
            q = sample
            q[:, :, :, :, c] = scipy.ndimage.filters.gaussian_filter(
                sample[:, :, :, :, c].astype(float), [eps, *[s] * ndims], truncate=6)
            q[:, :, :, :, c] = self._do_blur_point_norm(q[:, :, :, :, c])

        else:
            q = scipy.ndimage.filters.gaussian_filter(
                sample.astype(float), [eps, *[s] * ndims, eps], truncate=6)
            q = self._do_blur_point_norm(q)

        # normalize
        if self.do_blur_channel_norm:
            sumq = np.sum(q, 4, keepdims=True)
            q = q / (sumq + eps)

        return q

    def _do_blur_point_norm(self, q):
        ndims = q.ndim - 2
        if self.do_blur_point_norm:
            denom = np.sqrt(np.pi * 2 * ((self.blursigma)**ndims))
            denom = np.sqrt(np.pi * 2 * ((self.blursigma)))
            q = q * denom
            if self.do_blur_point_norm_clip is not None:
                q = np.clip(q, -np.inf, self.do_blur_point_norm_clip)
        return q


################################################################################
# These functions are legacy and need some looping over
################################################################################

def ext_data(segpath,
             volpath,
             batch_size,
             expected_files=None,
             ext='.npy',
             nb_restart_cycle=None,
             data_proc_fn=None,  # processing function that takes in one arg (the volume)
             vol_rand_seed=None,
             name='ext_data',
             verbose=False,
             yield_incomplete_final_batch=True,
             patch_stride=1,
             extract_slice=None,
             patch_size=None,
             ext_data_fields=None,
             expected_nb_files=-1):
    assert ext_data_fields is not None, "Need some external data fields"

    # get filenames at given paths
    volfiles = _get_file_list(segpath, ext, vol_rand_seed)
    nb_files = len(volfiles)
    assert nb_files > 0, "Could not find any files at %s with extension %s" % (segpath, ext)

    # compute subvolume split
    vol_data = _load_medical_volume(os.path.join(volpath, expected_files[0]), '.npz')
    # process volume
    if data_proc_fn is not None:
        vol_data = data_proc_fn(vol_data)

    # nb_patches_per_vol = 1
    # if patch_size is not None and all(f is not None for f in patch_size):
        # nb_patches_per_vol = np.prod(pl.gridsize(vol_data.shape, patch_size, patch_stride))
    if nb_restart_cycle is None:
        nb_restart_cycle = nb_files

    # assert nb_restart_cycle <= (nb_files * nb_patches_per_vol), \
        # '%s restart cycle (%s) too big (%s) in %s' % \
        # (name, nb_restart_cycle, nb_files * nb_patches_per_vol, volpath)

    # check the number of files matches expected (if passed)
    if expected_nb_files >= 0:
        assert nb_files == expected_nb_files, \
            "number of files do not match: %d, %d" % (nb_files, expected_nb_files)
    if expected_files is not None:
        if not (volfiles == expected_files):
            print('file lists did not match !!!', file=sys.stderr)

    # iterate through files
    fileidx = -1
    batch_idx = -1
    feat_idx = 0
    while 1:
        fileidx = np.mod(fileidx + 1, nb_restart_cycle)
        if verbose and fileidx == 0:
            print('starting %s cycle' % name)

        this_ext_data = np.load(os.path.join(segpath, volfiles[fileidx]))
        # print(os.path.join(segpath, volfiles[fileidx]), " was loaded")

        for _ in range(nb_patches_per_vol):
            if batch_idx == -1:
                ext_data_batch = [this_ext_data[f] for f in ext_data_fields]
            else:
                tmp_data = [this_ext_data[f] for f in ext_data_fields]
                ext_data_batch = [[*ext_data_batch[f], this_ext_data[f]]
                                  for f in range(len(tmp_data))]

            # yield patch
            batch_idx += 1
            batch_done = batch_idx == batch_size - 1
            files_done = np.mod(fileidx + 1, nb_restart_cycle) == 0
            final_batch = (yield_incomplete_final_batch and files_done)
            if verbose and final_batch:
                print('last batch in %s cycle %d' % (name, fileidx))

        if batch_done or final_batch:
            for fi, f in enumerate(ext_data_batch):
                ext_data_batch[fi] = np.array(f)

            batch_idx = -1
            yield ext_data_batch


def vol_ext_data(volpath,
                 segpath,
                 proc_vol_fn=None,
                 proc_seg_fn=None,
                 verbose=False,
                 name='vol_seg',  # name, optional
                 ext='.npz',
                 nb_restart_cycle=None,  # number of files to restart after
                 nb_labels_reshape=-1,
                 collapse_2d=None,
                 force_binary=False,
                 nb_input_feats=1,
                 relabel=None,
                 vol_rand_seed=None,
                 vol_subname='norm',  # subname of volume
                 seg_subname='norm',  # subname of segmentation
                 **kwargs):
    """
    generator with (volume, segmentation)

    verbose is passed down to the base generators.py primitive generator (e.g. vol, here)

    ** kwargs are any named arguments for vol(...),
        except verbose, data_proc_fn, ext, nb_labels_reshape and name
            (which this function will control when calling vol())
    """

    # get vol generator
    vol_gen = vol(volpath, **kwargs, ext=ext,
                  nb_restart_cycle=nb_restart_cycle, collapse_2d=collapse_2d, force_binary=False,
                  relabel=None, data_proc_fn=proc_vol_fn, nb_labels_reshape=1, name=name + ' vol',
                  verbose=verbose, nb_feats=nb_input_feats, vol_rand_seed=vol_rand_seed)

    # get seg generator, matching nb_files
    # vol_files = [f.replace('norm', 'aseg') for f in _get_file_list(volpath, ext)]
    # vol_files = [f.replace('orig', 'aseg') for f in vol_files]
    vol_files = [f.replace(vol_subname, seg_subname)
                 for f in _get_file_list(volpath, ext, vol_rand_seed)]
    seg_gen = ext_data(segpath,
                       volpath,
                       **kwargs,
                       data_proc_fn=proc_seg_fn,
                       ext='.npy',
                       nb_restart_cycle=nb_restart_cycle,
                       vol_rand_seed=vol_rand_seed,
                       expected_files=vol_files,
                       name=name + ' ext_data',
                       verbose=False)

    # on next (while):
    while 1:
        # get input and output (seg) vols
        input_vol = next(vol_gen).astype('float16')
        output_vol = next(seg_gen)  # .astype('float16')

        # output input and output
        yield (input_vol, output_vol)


def vol_ext_data_prior(*args,
                       proc_vol_fn=None,
                       proc_seg_fn=None,
                       prior_type='location',  # file-static, file-gen, location
                       prior_file=None,  # prior filename
                       prior_feed='input',  # input or output
                       patch_stride=1,
                       patch_size=None,
                       batch_size=1,
                       collapse_2d=None,
                       extract_slice=None,
                       force_binary=False,
                       nb_input_feats=1,
                       verbose=False,
                       vol_rand_seed=None,
                       **kwargs):
    """
    generator that appends prior to (volume, segmentation) depending on input
    e.g. could be ((volume, prior), segmentation)
    """

    if verbose:
        print('starting vol_seg_prior')

    # prepare the vol_seg
    gen = vol_ext_data(*args, **kwargs,
                       proc_vol_fn=None,
                       proc_seg_fn=None,
                       collapse_2d=collapse_2d,
                       extract_slice=extract_slice,
                       force_binary=force_binary,
                       verbose=verbose,
                       patch_size=patch_size,
                       patch_stride=patch_stride,
                       batch_size=batch_size,
                       vol_rand_seed=vol_rand_seed,
                       nb_input_feats=nb_input_feats)

    # get prior
    if prior_type == 'location':
        prior_vol = nd.volsize2ndgrid(vol_size)
        prior_vol = np.transpose(prior_vol, [1, 2, 3, 0])
        prior_vol = np.expand_dims(prior_vol, axis=0)  # reshape for model

    else:  # assumes a npz filename passed in prior_file
        with timer.Timer('loading prior', verbose):
            data = np.load(prior_file)
            prior_vol = data['prior'].astype('float16')

    if force_binary:
        nb_labels = prior_vol.shape[-1]
        prior_vol[:, :, :, 1] = np.sum(prior_vol[:, :, :, 1:nb_labels], 3)
        prior_vol = np.delete(prior_vol, range(2, nb_labels), 3)

    nb_channels = prior_vol.shape[-1]

    if extract_slice is not None:
        if isinstance(extract_slice, int):
            prior_vol = prior_vol[:, :, extract_slice, np.newaxis, :]
        else:  # assume slices
            prior_vol = prior_vol[:, :, extract_slice, :]

    # get the prior to have the right volume [x, y, z, nb_channels]
    assert np.ndim(prior_vol) == 4, "prior is the wrong size"

    # prior generator
    if patch_size is None:
        patch_size = prior_vol.shape[0:3]
    assert len(patch_size) == len(patch_stride)
    prior_gen = patch(prior_vol, [*patch_size, nb_channels],
                      patch_stride=[*patch_stride, nb_channels],
                      batch_size=batch_size,
                      collapse_2d=collapse_2d,
                      infinite=True,
                      variable_batch_size=True,
                      nb_labels_reshape=0)
    assert next(prior_gen) is None, "bad prior gen setup"

    # generator loop
    while 1:

        # generate input and output volumes
        input_vol, output_vol = next(gen)
        if verbose and np.all(input_vol.flat == 0):
            print("all entries are 0")

        # generate prior batch
        prior_batch = prior_gen.send(input_vol.shape[0])

        if prior_feed == 'input':
            yield ([input_vol, prior_batch], output_vol)
        else:
            assert prior_feed == 'output'
            yield (input_vol, [output_vol, prior_batch])


def _log(z, method=1, eps=np.finfo(float).eps, logeps=-36):
    """
    various log methods, sometimes used for binary map (as used in segmentations)
    """

    # log space
    if z.dtype != np.dtype(float):
        z = z.astype(float)

    # option 1: add eps then take log
    if method == 1:
        z += eps  # faster than np.maximum(z, eps)
        logseg = np.log(z)
        assert not np.any(np.isnan(logseg))
        assert not np.any(np.isinf(logseg))

    # option 2: (I think a bit slower?)
    elif method == 2:
        logseg = np.log(z)
        logseg[np.isinf(logseg)] = logeps
        assert not np.any(np.isnan(logseg))

    # option 3: (maybe slow but nice)
    elif method == 3:
        assert np.all(np.logical_or(z == 0, z == 1)), "This log method assumes z is 0 or 1"

        logseg = np.zeros(z.shape)
        logseg[z == 0] = logeps

    assert np.min(logseg) > -np.inf
    return logseg


def parallel_gen(pth, nb_restart_cycle, data):
    """
    manual parallel generator
    we should try to use the neuron generators as much as possible
    """

    # obtain numpy files
    files = [f for f in os.listdir(pth) if f.endswith('.npz')]
    fileidx = -1
    while 1:
        fileidx = np.mod(fileidx + 1, nb_restart_cycle)

        vol_data = np.load(os.path.join(pth, files[fileidx]))['vol_data']
        vol_data = vol_data.astype('uint8')

        vol_data = nrn_generators._relabel(vol_data, data.labels, forcecheck=False)

        # vol_data =
        #   nrn_generators._categorical_prep(vol_data, data.nb_labels, True, vol_data.shape)
        vol_data = nrn_generators._to_categorical(vol_data, data.nb_labels, True)

        vol_data = np.squeeze(vol_data)
        vol_data = np.transpose(vol_data, [2, 0, 1, 3])
        yield vol_data


# atrophy pair synth generator
def synth_gen_pair(label_vols, gen_model,
                   batch_size=8,
                   use_rand=True,
                   gpuid=-1,
                   use_log=False,
                   ret2=False,
                   return_labels=False):
    """
    generator for synthesizing pairs of images, one without atrophy and one with atrophy
    """
    inshape = label_vols[0].shape
    ndims = len(label_vols[0].shape)

    batch_labels = np.zeros((batch_size, *inshape, 1))
    batch_labels1 = np.zeros((batch_size, *inshape, 1))
    batch_labels2 = np.zeros((batch_size, *inshape, 1))
    batch_inputs = np.zeros((batch_size, *inshape, 2))
    batch_outputs = np.zeros((batch_size, *inshape, 2))
    if gpuid >= 0:
        device = '/gpu:' + str(gpuid)
    else:
        device = '/cpu:0'

    ind = -1
    while (True):
        for bind in range(batch_size):
            if use_rand:
                ind = np.random.randint(0, len(label_vols))
            else:
                ind = np.mod(ind + 1, len(label_vols))
            batch_labels[bind, ...] = label_vols[ind].data[..., np.newaxis]

            with (tf.device(device)):
                pred = gen_model.predict(batch_labels)

            for bind in range(batch_size):
                # randomly flip order (DISABLED since change is in t2 space!)
                if np.random.randint(0, 2) == -1:
                    fwd = False
                    batch_inputs[bind, ..., 0] = pred[1][bind, ..., 0]
                    batch_inputs[bind, ..., 1] = pred[0][bind, ..., 0]
                else:
                    fwd = True
                    batch_inputs[bind, ..., 0] = pred[0][bind, ..., 0]
                    batch_inputs[bind, ..., 1] = pred[1][bind, ..., 0]

                changed_labels = (pred[6][bind, ...] != pred[7][bind, ...])[..., 0]  # in t2 coords
                if not use_log:
                    batch_outputs[bind, ..., 0] = np.where(changed_labels == 0, 1, 0)
                    batch_outputs[bind, ..., 1] = np.where(changed_labels != 0, 1, 0)
                else:
                    batch_outputs[bind, ..., 0] = np.where(changed_labels == 0, 10, -10)
                    batch_outputs[bind, ..., 1] = np.where(changed_labels != 0, 10, -10)
                if return_labels:
                    if fwd:
                        batch_labels1[bind, ..., 0] = pred[4][bind, ..., 0]  # time1 lab
                        batch_labels2[bind, ..., 0] = pred[7][bind, ..., 0]  # time2 lab w atrophy
                    else:
                        batch_labels1[bind, ..., 0] = pred[7][bind, ..., 0]  # time1 lab
                        batch_labels2[bind, ..., 0] = pred[4][bind, ..., 0]  # time2 lab w atrophy

        inputs = [batch_inputs]
        outputs = [batch_outputs]

        if ret2:
            outputs += [batch_outputs]
        if return_labels:
            outputs += [batch_labels, batch_labels1, batch_labels2]

        yield inputs, outputs

    return 0


def _erase_labels(seg, img, erase_list):
    img = img.copy()
    img[np.isin(seg, erase_list)] = 0
    return img


def exvivo(gen_model, segs, lh_list, rh_list, bs_list,
           lh_keep_pval=.5, rh_keep_pval=.5, bs_keep_pval=1,
           insert_bag_pval=.8, batch_size=16,
           max_bag_dilations=20, use_rand=True,
           seg_dilations=0, seg_closes=0, lab_to_ind=None,
           return_bias=False, bag_label=None, mask_brain=False):
    '''
    synthesize ex vivo images, inserting a bag sometimes, and randomly erasing one hemisphere or 
    the other as well as the brainstem and the cerebellum
    '''

    if bag_label is None:
        bag_label = np.array(segs).max() + 1

    if not use_rand:
        idx0 = 0
    while 1:
        if not use_rand:
            idx = np.array([np.mod(ind, segs.shape[0]) for ind in range(idx0, idx0 + batch_size)])
            idx0 = np.mod(idx0 + batch_size, segs.shape[0])
        else:
            idx = np.random.randint(0, segs.shape[0], batch_size)

        seg_in = segs[idx, ...]
        erase_list = []
        if use_rand:
            if np.random.randint(2):  # check lh first
                if np.random.rand() > lh_keep_pval:
                    erase_list += lh_list
                elif np.random.rand() > rh_keep_pval:
                    erase_list += rh_list
            else:    # check rh first to keep it unbiased
                if np.random.rand() > rh_keep_pval:
                    erase_list += rh_list
                elif np.random.rand() > lh_keep_pval:
                    erase_list += lh_list

            if np.random.rand() > bs_keep_pval:
                erase_list += bs_list
        else:  # disable randomness for testing/debugging. Not really very effective!
            if (np.mod(idx0, 3) and lh_keep_pval < 1) or (lh_keep_pval == 0):
                erase_list += lh_list
            elif (np.mod(idx0, 5) and rh_keep_pval < 1) or (rh_keep_pval == 0):
                erase_list += rh_list
            if (np.mod(idx0, 7) and bs_keep_pval < 1) or (bs_keep_pval == 0):
                erase_list += bs_list

        seg_in = _erase_labels(seg_in, seg_in, erase_list)

        brain_mask = tf.cast(tf.where(seg_in > 0, 1, 0), tf.float32)
        if np.random.rand() < insert_bag_pval:  # put bag in sometimes
            bag_dilations = np.random.randint(1, max_bag_dilations)

            crop_lims = [26 / 27.0, 27 / 27.0]
            dil_fn1 = lambda x: utils.utils.morphology_3d(x, 1, 1,
                                                          operation='dilate',
                                                          eight_connectivity=False)
            seg_dil = tf.map_fn(dil_fn1, brain_mask, fn_output_signature=tf.bool)
            dil_fn2 = lambda x: utils.utils.morphology_3d(x, 1, bag_dilations,
                                                          operation='dilate',
                                                          rand_crop=crop_lims,
                                                          eight_connectivity=False)
            for i in range(3):
                seg_dil = tf.cast(seg_dil, dtype=brain_mask.dtype)
                seg_dil = tf.map_fn(dil_fn2, seg_dil, fn_output_signature=tf.bool)
            seg_bag = tf.cast(seg_dil, brain_mask.dtype) - brain_mask
            seg_in += tf.cast(seg_bag, seg_in.dtype) * (bag_label)  # new label

        img, seg, img_no_bias, bias = gen_model.predict(seg_in)

        if lab_to_ind is not None:
            seg_old = seg.copy()
            seg = lab_to_ind[seg]

        if not use_rand and idx0 == 1:
            idx0 *= 1

        # img_no_bias, seg, mean_img2 = gen_model_2.predict(seg_in)
        low_mask = tf.cast(seg > 0, img.dtype)
        hi_mask = tf.cast(seg < bag_label, img.dtype)
        brain_mask = low_mask * hi_mask
        if seg_closes > 0:
            dil_fn = lambda x: utils.utils.morphology_3d(x, 1, seg_closes,
                                                         operation='close',
                                                         eight_connectivity=True)
            brain_mask = tf.map_fn(dil_fn, brain_mask,
                                   fn_output_signature=tf.bool).numpy().astype('float32')

        brain_vol = brain_mask * img
        brain_vol_no_bias = img_no_bias * brain_mask

        # map the median in the unnormalized image to 0.5 in the normed one
        # median = tf.cast(tfp.stats.percentile(brain_vol[brain_vol>0], 50), img.dtype)
        # norm_val = median / .5
        # tu = tf.unique_with_counts(tf.cast(brain_vol[brain_mask>0], tf.int16))
        norm_val = tf.cast(tfp.stats.percentile(brain_vol[brain_mask > 0], 99), img.dtype)
        # norm_val = tf.cast(tu[0][0], tf.float32) / .5  # map most common brain value to .5
        img = tf.clip_by_value(tf.math.divide_no_nan(img, norm_val), 0, 3)
        img_no_bias = tf.clip_by_value(tf.math.divide_no_nan(img_no_bias, norm_val), 0, 3)

        # now make the image means the same
        brain_vol = brain_mask * img
        brain_vol_no_bias = brain_mask * img_no_bias
        img_mean = tf.reduce_mean(brain_vol[brain_mask > 0])
        img_no_bias_mean = tf.reduce_mean(brain_vol_no_bias[brain_mask > 0])

        ratio = tf.math.divide_no_nan(img_no_bias_mean, img_mean)
        img *= ratio
        if mask_brain:
            img *= brain_mask
            img_no_bias *= brain_mask
            seg *= brain_mask

        # inputs = np.concatenate([img.numpy(), seg], axis=-1)
        outputs = img_no_bias.numpy()
        if return_bias:
            outputs = [outputs, 1. / (bias + 1e-9)]
        yield [img.numpy(), seg], outputs
