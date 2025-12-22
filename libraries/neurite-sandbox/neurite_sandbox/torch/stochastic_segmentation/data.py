import pathlib
import subprocess
import sys

import torchvision.datasets
import torchvision.transforms
import torch.nn.functional as F
import torch
import nibabel as nib


class MNIST():

    def __init__(self):
        self.data = {}

        mnist_trainset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True)
        self.data['train'], self.data['train_blur'] = self.prep_dataset(
            mnist_trainset.data)

        mnist_testset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True)
        self.data['test'], self.data['test_blur'] = self.prep_dataset(
            mnist_testset.data)

    def prep_dataset(self, data, blur_kernel=[5, 5], blur_sigma=[1, 1]):

        # clean and pad MNIST data
        data = data.float().cuda() / 255
        data = F.pad(data, (2, 2, 2, 2, 0, 0))

        # have a blurred version, which we'll use to help us simulate stochasticity
        trf = torchvision.transforms.GaussianBlur(
            kernel_size=blur_kernel, sigma=blur_sigma)
        data_blur = trf(data)
        data_blur = data_blur / data_blur.amax(dim=[1, 2], keepdim=True)

        return data[:, None, ...], data_blur[:, None, ...]


class OASIS2D():
    def __init__(self,
                 dst="/tmp/universeg_oasis/",
                 tar_url="https://surfer.nmr.mgh.harvard.edu/ftp/data/neurite/data/neurite-oasis.2d.v1.0.tar",  # noqa
                 fields=['norm', 'orig', 'seg4', 'seg24'],
                 resize=None,
                 device='cuda'):
        dest_folder = pathlib.Path(dst)

        if not dest_folder.exists():

            subprocess.run(
                ["curl", tar_url, "--create-dirs", "-o",
                    str(dest_folder / 'neurite-oasis.2d.v1.0.tar')],
                stderr=subprocess.DEVNULL,
                check=True,
            )

            subprocess.run(
                ["tar", 'xf', str(
                    dest_folder / 'neurite-oasis.2d.v1.0.tar'), '-C',
                    str(dest_folder)],
                stderr=subprocess.DEVNULL,
                check=True,
            )

        self.data = {}
        for field in fields:
            self.data[field] = []

        subj_folders = [f for f in sorted(
            dest_folder.glob("*/")) if f.is_dir()]
        for subj_folder in subj_folders:
            for field in fields:
                file_name = subj_folder / '{}.nii.gz'.format('slice_' + field)
                im = nib.load(file_name).get_fdata().squeeze()
                im = torch.from_numpy(im).float().to(device)

                if resize is not None:
                    interp = 'nearest' if field.startswith(
                        'seg') else 'bilinear'
                    im = F.interpolate(
                        im[None, None, ...], size=resize, mode=interp)
                    im = im[0, 0, ...]

                self.data[field].append(im)

        for field in fields:
            self.data[field] = torch.stack(
                self.data[field], axis=0)[:, None, ...]

        print('warning: OASIS2D data is not split into train/val/test',
              file=sys.stderr)
