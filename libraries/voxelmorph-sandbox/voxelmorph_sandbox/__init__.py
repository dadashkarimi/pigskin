# import general python utilities
from . import py

# get backend
import voxelmorph as vxm
import voxelmorph.py.utils
backend = vxm.py.utils.get_backend()

# imports based on backend
if backend == 'pytorch':
    # the pytorch backend can be enabled by setting the VXM_BACKEND
    # environment var to "pytorch"
    from . import torch
    from .torch import *

else:
    # tensorflow is default backend
    from . import tf
    from .tf import layers
    from .tf import networks
    from .tf import utils
