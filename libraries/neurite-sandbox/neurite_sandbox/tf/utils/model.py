""" model utilities """

# internal imports

# third party imports
from tqdm import tqdm
import numpy as np

# local imports


def transfer_partial_conv_weights_3d(model, wts, old_nb_feat=64, new_nb_feats=48):
    """
    transfer saved weights from an old conv model to a new conv model
    with different number of features

    **only works in special cases**
    """

    for wi, w in enumerate(wts):

        if len(wts[wi].shape) == 1 and wts[wi].shape[-1] == old_nb_feat:
            wts[wi] = wts[wi][0:new_nb_feats]
        elif len(wts[wi].shape) == 5:
            if wts[wi].shape[-2] == old_nb_feat and wts[wi].shape[-1] == old_nb_feat:
                wts[wi] = wts[wi][:, :, :, 0:new_nb_feats, 0:new_nb_feats]
            elif wts[wi].shape[-1] == old_nb_feat:
                wts[wi] = wts[wi][:, :, :, :, 0:new_nb_feats]
            elif wts[wi].shape[-2] == old_nb_feat:
                wts[wi] = wts[wi][:, :, :, 0:new_nb_feats, :]

    model.set_weights(wts)


def check_equal_weights_of_two_models(model1, model2, tqdm=tqdm):
    """
    checks if two models have the same weight values
    """
    for layer in tqdm(model1.layers):
        if layer.name in [f.name for f in model2.layers]:
            wts = layer.get_weights()
            wts_chk = model2.get_layer(layer.name).get_weights()
            for f in range(len(wts)):
                assert np.max(wts[f] - wts_chk[f]) < 1e-5, \
                    "weight check failed: err %f" % np.max(wts[f] - wts_chk[f])
