

# internal python imports
import os
import glob
import pathlib
import random

# third party
import numpy as np
import nibabel as nib
from tqdm import tqdm

# local imports
import pystrum.pynd as pynd


def load_malte_brain_train_data(data_dir='/autofs/cluster/fssubjects/test/mu40-crazy/data/raw/',
                                tqdm=tqdm):
    """
    returns a dictionary <data> with keys 'train' and 'train_seg'
    """
    train_subjects = sorted(glob.glob(os.path.join(
        data_dir, 'images/training_hcp/t1_ss_??.nii.gz')))
    seg_subjects = sorted(glob.glob(os.path.join(data_dir, 'labels/training_hcp/??.nii.gz')))
    assert len(train_subjects) > 0, 'no training subjects found'
    assert len(train_subjects) == len(seg_subjects), 'images and label counts don\'t match'

    data = {}
    data['train'] = np.stack([nib.load(f).get_fdata().squeeze()
                              for f in tqdm(train_subjects)], 0)[..., np.newaxis] / 255
    data['train_seg'] = np.stack([nib.load(f).get_fdata().squeeze()
                                  for f in tqdm(seg_subjects)], 0)[..., np.newaxis]

    return data


MALTE_DATA_DIR = '/autofs/cluster/freesurfer/subjects/test/mu40-crazy/data/raw'


def load_malte_brain_validataion_data(data_dir=MALTE_DATA_DIR):

    # val data
    data = {}
    # i1, i2, l1, l2, wa, wn, aa, an = range(8)
    i1, _, l1, _, _, _, _, _ = range(8)

    def load(f, out_type='float32', normalize=True):
        if not os.path.isfile(f):
            files = sorted(glob.glob(os.path.join(data_dir, f)))
            return [load(f, out_type=out_type, normalize=normalize) for f in files]

        if f.endswith('.lta'):
            try:
                import freesurfer as fs  # pylint: disable=import-error
            except _:
                pass
            out = fs.LinearTransform.read(f).as_vox().matrix
            return np.linalg.inv(out)[None, :-1, :]  # Batch dim.

        out = np.load(f) if f.endswith('.npy') else np.asarray(nib.load(f).dataobj)
        out = np.squeeze(out)[None, ...]
        if out_type:
            out = out.astype(out_type)

        is_warp = out.ndim == 5
        if normalize and not is_warp:
            out -= out.min()
            out /= out.max()

        return out if is_warp else out[..., None]

    data['oasis_t1t1_ss'] = (
        load('images/oasis1/t1_ss_0[0-4].nii.gz'),
        load('images/oasis2/t1_ss_0[0-4].nii.gz'),
        load('labels/oasis1/0[0-4].nii.gz', normalize=False, out_type='int16'),
        load('labels/oasis2/0[0-4].nii.gz', normalize=False, out_type='int16'),
        load('ants/oasis1-oasis2/wrp_ss_0[0-4].nii.gz'),
        load('niftyreg/oasis1-oasis2/wrp_ss_0[0-4].nii.gz'),
        #     load('ants/oasis1-oasis2/aff_ss_0[0-4].lta'),
        #     load('niftyreg/oasis1-oasis2/aff_ss_0[0-4].lta'),
    )

    data['hcp_t1t1_ss'] = (
        load('images/hcp1/t1_ss_0[0-4].nii.gz'),
        load('images/hcp2/t1_ss_0[0-4].nii.gz'),
        load('labels/hcp1/0[0-4].nii.gz', normalize=False, out_type='int16'),
        load('labels/hcp2/0[0-4].nii.gz', normalize=False, out_type='int16'),
        load('ants/hcp1-hcp2/wrp_t1t1_ss_0[0-4].nii.gz'),
        load('niftyreg/hcp1-hcp2/wrp_t1t1_ss_0[0-4].nii.gz'),
        #     load('ants/hcp1-hcp2/aff_t1t1_ss_0[0-4].lta'),
        #     load('niftyreg/hcp1-hcp2/aff_t1t1_ss_0[0-4].lta'),
    )

    data['oasis/hcp_t1t1_ss'] = (
        data['oasis_t1t1_ss'][i1],
        data['hcp_t1t1_ss'][i1],
        data['oasis_t1t1_ss'][l1],
        data['hcp_t1t1_ss'][l1],
        load('ants/oasis1-hcp1/wrp_ss_0[0-4].nii.gz'),
        load('niftyreg/oasis1-hcp1/wrp_ss_0[0-4].nii.gz'),
        #     load('ants/oasis1-hcp1/aff_ss_0[0-4].lta'),
        #     load('niftyreg/oasis1-hcp1/aff_ss_0[0-4].lta'),
    )

    # Validate.
    for k, v in data.items():
        print(k, [len(x) for x in v])
        for i, ls in enumerate(v):
            assert ls, f'No data for key "{k}" index {i}'
    # vol_size = next(iter(data.values()))[0][0].shape[1:-1]

    return data


def get_malte_brain_selected_labels():
    labels = {
        2: 'Left-Cerebral-White-Matter', 41: 'Right-Cerebral-White-Matter',
        3: 'Left-Cerebral-Cortex', 42: 'Right-Cerebral-Cortex',
        4: 'Left-Lateral-Ventricle', 43: 'Right-Lateral-Ventricle',
        7: 'Left-Cerebellum-White-Matter', 46: 'Right-Cerebellum-White-Matter',
        8: 'Left-Cerebellum-Cortex', 47: 'Right-Cerebellum-Cortex',
        10: 'Left-Thalamus', 49: 'Right-Thalamus',
        11: 'Left-Caudate', 50: 'Right-Caudate',
        12: 'Left-Putamen', 51: 'Right-Putamen',
        13: 'Left-Pallidum', 52: 'Right-Pallidum',
        17: 'Left-Hippocampus', 53: 'Right-Hippocampus',
        18: 'Left-Amygdala', 54: 'Right-Amygdala',
        28: 'Left-VentralDC', 60: 'Right-VentralDC',
        16: 'Brain-Stem',
        14: '3rd-Ventricle', 15: '4th-Ventricle',
        31: 'Left-choroid-plexus', 63: 'Right-choroid-plexus',
    }
    return labels


SET_PATH = '/autofs/space/topaz_001/users/ah221/test/hyper/fullset'


def load_t1mix_brain_data(data_dir='/cluster/vxmdata1/FS_Slim/proc/cleaned/',
                          train_subj_list=SET_PATH + '/train',
                          train_max_load=1000,
                          val_subj_list=SET_PATH + '/validate',
                          val_max_load=10,
                          crop_to_support=0.001,
                          crop_mult_factor=32,
                          crop=None,
                          extract_slice_no=None,
                          extract_axis_no=-1,
                          vol_file='norm.mgz',  # 'norm_talairach.mgz',
                          seg_file='aseg.mgz',  # 'aseg_23_talairach.mgz',
                          tqdm=tqdm):
    """
    load t1 mix data from Andrew's cleaned data source 

    parameters:
        crop_to_support: crops volumes to a size that is a multiple of <crop_mult_factor> 
            and includes a population overage intensity of > crop_to_support.
            set to None to not do crop
    """
    print('TODO: move loader to NeuriteSandbox')

    main_path = pathlib.Path(data_dir)

    with open(train_subj_list) as f:
        train_subjects = f.readlines()
        train_subjects = [f.strip() for f in train_subjects]

    with open(val_subj_list) as f:
        val_subjects = f.readlines()
        val_subjects = [f.strip() for f in val_subjects]

    def load_data(main_path, subj_list, max_load=100):
        vols = []
        segs = []
        for f in tqdm(subj_list[:max_load]):
            volfile = os.path.join(str(main_path), f, vol_file)
            segfile = os.path.join(str(main_path), f, seg_file)

            assert os.path.isfile(volfile), volfile
            if os.path.isfile(volfile) and os.path.isfile(segfile):
                vols.append(nib.load(volfile).get_fdata().squeeze())
                segs.append(nib.load(segfile).get_fdata().squeeze())

                if extract_slice_no is not None:
                    vols[-1] = np.take(vols[-1], extract_slice_no, extract_axis_no)
                    segs[-1] = np.take(segs[-1], extract_slice_no, extract_axis_no)

        vols = np.stack(vols, 0)[..., np.newaxis]
        segs = np.stack(segs, 0)[..., np.newaxis]
        return vols, segs

    data = {}
    data['train'], data['train_seg'] = load_data(main_path, train_subjects, max_load=train_max_load)
    data['val'], data['val_seg'] = load_data(main_path, val_subjects, max_load=val_max_load)

    # rescale intensities
    data['train'] = data['train'] / 255
    data['val'] = data['val'] / 255

    # some bounding box
    if crop_to_support is not None:
        assert crop_mult_factor is not None
        assert crop is None
        crop = _crop_to_multiple(data['train'].mean(0).squeeze() >
                                 crop_to_support, crop_mult_factor)

    if crop is not None:
        print('crop', crop)
        bb_k_format = [(0, 0)] + crop + [(0, 0)]
        data['train'] = pynd.ndutils.volcrop(data['train'], crop=bb_k_format)
        data['train_seg'] = pynd.ndutils.volcrop(data['train_seg'], crop=bb_k_format)
        data['val'] = pynd.ndutils.volcrop(data['val'], crop=bb_k_format)
        data['val_seg'] = pynd.ndutils.volcrop(data['val_seg'], crop=bb_k_format)

    return data


def _crop_to_multiple(bw_vol, mult=32):
    """
    return crop matrix (in the form [(crop_start, crop_end), ...], given binary volume vol, 
    that is a multiple of 'mult'
    """

    ndims = len(bw_vol.shape)
    bb = pynd.ndutils.boundingbox(bw_vol)
    if not isinstance(bb[0], (list, tuple)):
        bb = [(bb[f], bb[f + ndims]) for f in range(ndims)]

    crop = []
    for d in range(ndims):
        df = bb[d][1] - bb[d][0]
        df = int(np.ceil(df / mult) * mult - df)
        st, en = df // 2, df - df // 2
        crop.append((bb[d][0] - st, bw_vol.shape[d] - (bb[d][1] + en)))
        assert crop[d][0] >= 0
        assert crop[d][1] >= 0

    return crop


class StuartFlairAffineData():
    def __init__(self, data_path):
        self.data_path = data_path
        self.data_dct = None

    def _keras_vol_prep(self, v, clip_pct=97, ds=2):
        """normalize by percentile, clip, downsample, and keras-format

        Args:
            v ([type]): 3D volume
            p (int, optional): percentile. Defaults to 97.
            ds (int, optional): downsampling. Defaults to 2.

        Returns:
            [type]: [description]
        """
        v = v[::ds, ::ds, ::ds]
        v = v - v.min()
        v = v / np.percentile(v, clip_pct)
        v = np.clip(v, 0, 1)
        v = v[np.newaxis, ..., np.newaxis]
        return v

    # quick data loader

    def load_all_data(self, ds=2, tqdm=tqdm, clip_pct=97):
        """load all data into a dictionary

        Args:
            ds (int, optional): [description]. Defaults to 2.

        Returns:
            [type]: [description]
        """
        # get all the subjects
        data_path = self.data_path
        folders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]

        # go through subjects and
        dct = {}
        for subj in tqdm(folders):
            files = glob.glob(os.path.join(data_path, subj, 'conf.flair.time?.nii.gz'))
            files.sort()

            if len(files) == 0:
                continue

            # go through files in subject
            dct[subj] = {'full': [], 'ss': [], 'filename': []}
            for file in files:
                ssfile = file.replace('.nii.gz', '.masked.mgz')
                vol = nib.load(file).get_fdata()
                dct[subj]['full'].append(self._keras_vol_prep(vol, ds=ds, clip_pct=clip_pct))
                vol = nib.load(ssfile).get_fdata()
                dct[subj]['ss'].append(self._keras_vol_prep(vol, ds=ds, clip_pct=clip_pct))
                dct[subj]['filename'].append(file)

        self.data_dct = dct
        return dct

    def affine_generator(self,
                         batch_size=1,
                         bidir=False,
                         ds=None,
                         do_aug=False,
                         use_midspace=False,
                         return_affine=False,
                         return_warp=False,
                         separate_masked_loss=True,
                         verbose=0):
        """generator for affine networks

        uses self.data_dct if it exists, or else self.data_path

        Args:
            batch_size (int, optional): batch size. Defaults to 1.
            bidir (bool, optional): whether do yield bools. Defaults to False.
            ds ([type], optional): [description]. Defaults to None.
            do_aug (bool, optional): [description]. Defaults to False.
            separate_masked_loss (bool, optional): whether to yield masked data for separate loss. 
                Defaults to True.

        Yields:
            [type]: [inputs, outputs] where inputs are original images and skull stripped data
        """

        is_dct = self.data_dct is not None
        if not is_dct:
            folders = [f for f in os.listdir(self.data_path) if os.path.isdir(
                os.path.join(self.data_path, f))]
            if ds is None:
                ds = 2
        else:
            folders = list(self.data_dct.keys())
            assert ds is None

        if return_affine:
            zero_affine = np.zeros((batch_size, 3, 4))

        while True:
            x1batch = []
            x2batch = []
            x1batch_masked = []
            x2batch_masked = []

            for _ in range(batch_size):
                subj = random.choice(folders)
                if verbose > 0:
                    print(subj, flush=True)

                # choose a subject
                if not is_dct:
                    files = glob.glob(os.path.join(self.data_path, subj, 'conf.flair.time?.nii.gz'))

                    idx = np.random.choice(files, size=2, replace=False)
                    file = idx[0]
                    x1 = self._keras_vol_prep(nib.load(file).get_fdata(), ds=ds)
                    if separate_masked_loss:
                        ssfile = file.replace('.nii.gz', '.masked.mgz')
                        x1ss = self._keras_vol_prep(nib.load(ssfile).get_fdata(), ds=ds)

                    file = idx[1]
                    x2 = self._keras_vol_prep(nib.load(file).get_fdata(), ds=ds)
                    if separate_masked_loss:
                        ssfile = file.replace('.nii.gz', '.masked.mgz')
                        x2ss = self._keras_vol_prep(nib.load(ssfile).get_fdata(), ds=ds)

                else:
                    idx = np.random.choice(len(self.data_dct[subj]['full']), size=2, replace=False)
                    x1 = self.data_dct[subj]['full'][idx[0]]
                    x1ss = self.data_dct[subj]['ss'][idx[0]]
                    x2 = self.data_dct[subj]['full'][idx[1]]
                    x2ss = self.data_dct[subj]['ss'][idx[1]]

                # some data augmentation: flipping and axis shuffling
                if do_aug:
                    for d in range(1, 4):
                        if np.random.uniform() < 0.5:
                            x1 = np.flip(x1, d)
                            x2 = np.flip(x2, d)
                            if separate_masked_loss:
                                x1ss = np.flip(x1ss, d)
                                x2ss = np.flip(x2ss, d)

                    sh = list(range(1, 4))
                    random.shuffle(sh)

                    sh = [0] + sh + [4]
                    x1 = np.transpose(x1, sh)
                    x2 = np.transpose(x2, sh)
                    if separate_masked_loss:
                        x1ss = np.transpose(x1ss, sh)
                        x2ss = np.transpose(x2ss, sh)

                x1batch.append(x1)
                x2batch.append(x2)
                if separate_masked_loss:
                    x1batch_masked.append(x1ss)
                    x2batch_masked.append(x2ss)

            # gather into single matrices
            x1batch = np.concatenate(x1batch, 0)
            x2batch = np.concatenate(x2batch, 0)
            if separate_masked_loss:
                x1batch_masked = np.concatenate(x1batch_masked, 0)
                x2batch_masked = np.concatenate(x2batch_masked, 0)

            if return_warp:
                zero_warp = np.zeros(x1batch.shape[:-1] + (3,))

            if use_midspace:
                zero_stack = np.zeros(x1batch.shape[:-1] + (2,))

            # prepare inputs and outputs
            if separate_masked_loss:
                if bidir:
                    inputs = [x1batch, x2batch, x1batch_masked, x2batch_masked]
                    outputs = [x2batch_masked, x1batch_masked]
                else:
                    inputs = [x1batch, x2batch, x1batch_masked]
                    outputs = [x2batch_masked]
            else:
                if bidir:
                    inputs = [x1batch, x2batch]
                    outputs = [x2batch, x1batch]
                else:
                    inputs = [x1batch, x2batch]
                    outputs = [x2batch]

            if use_midspace:
                outputs += [zero_stack]
            if return_affine:
                outputs += [zero_affine]
            if return_warp:
                outputs += [zero_warp]

            yield (inputs, outputs)

# class NeuriteData():
#     def __init__(self, path='/cluster/vxmdata1/FS_Slim/proc/cleaned/', subj_file=None):
#         self.path = path
#         self.subj_file = subj_file

#     def load_dataset_in_memory():
#         if self.subj_file is not None:
#             subj_list =
