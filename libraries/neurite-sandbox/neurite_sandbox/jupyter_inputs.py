# main python imports
import os
import shutil
import copy
import sys
from imp import reload
from tempfile import NamedTemporaryFile
import pathlib

# usual third parties
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
import nibabel as nib
import tqdm.keras
from tqdm.keras import TqdmCallback
from tqdm import tqdm_notebook as tqdm
from IPython.display import display, Image
import h5py

# local imports
import pystrum
import voxelmorph as vxm
import voxelmorph_sandbox as vxms

# useful vars
eps = np.finfo('float').eps
