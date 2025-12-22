# third party
import time
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import numpy as np
import random
import warnings


class ReduceLRWithModelCheckpointAndRecovery(keras.callbacks.ReduceLROnPlateau):
    '''
       keras callback that will write the best model into a file, 
       then revert to it if the loss excedes a threshold (or is worse
       than the worst loss during the burn_in period). 
       Subclass of keras.callbacks.ReduceLROnPlateau 
       and supports all its functionality.

       fname - filename to write model to (if contains %d will add epoch)
       thresh - use absolute threshold instead of worst during burn_in
                this threshold is only considered if the loss is increasing
       thresh_increase_factor - use relative threshold based on increase from
                     previous epoch. This works somewhat differently for positive losses
                     like MSE than negative ones like Dice. For positive, the relative thresh is:
                        inc_thresh = prev_loss + prev_loss * self.thresh_inc_factor
                        so an increase from .05 to 1 would exceed a thresh_inc_factor of 2
                     for negative it is:
                        inc_thresh = prev_loss - prev_loss * self.thresh_inc_factor
                        so an increase from -.1 to -0.05 would exceed a thresh_inc_factor of .5

       save_weights_only - all model.save_weights() instead of model.save()
       increase_epoch - epoch at which to apply increase_factor to lr
       increase_factor - amount to increase lr by at increase_epoch
       decrease_factor - amount to decrease lr by (if not specified use factor)
       recovery_decrease_factor - opional factor to decrease by only when recovering,
       burn_in - # of epochs after which saving is started (defaults to 0)
       nloss - # of epochs to average over when computing previous loss
       save_prev - boolean. If true will save (and revert to) previous. Best will still be saved, 
                   but not used to recover from a bad step
       loss_func - an alternative function to compute the epoch-wide loss. Takes current model
                   as a parameter
       avg_loss_num_epochs=1, - number of epochs to average the loss over before deciding whether the
                   current epoch is a new best
    '''
    def __init__(self,
                 fname='saved.%3.3d.h5', 
                 lr_log_name='lr',
                 save_weights_only=True,
                 thresh=None,
                 thresh_inc_factor=None,
                 increase_epoch=None,
                 increase_factor=10,
                 decrease_factor=None,
                 recovery_decrease_factor=None,
                 warm_restart_epoch=None,
                 warm_restart_lr=1e-3,
                 warm_restart_thresh=None,
                 warm_restarts=None,   
                 burn_in=0,
                 restart=False,
                 loss_func=None,
                 save_prev=False,
                 avg_loss_num_epochs=None,
                 use_median=False,
                 nloss=5,
                 **kwargs):
        super(ReduceLRWithModelCheckpointAndRecovery, self).__init__(**kwargs)
        if self.verbose > 0:
            self._supports_tf_logs = True

        self.use_median = use_median
        self.restart = restart

        if save_prev:
            self.prev_fname = 'prev.' + fname
        else:
            self.prev_fname = None

        if isinstance(warm_restart_epoch, list):
            assert len(warm_restart_epoch) == len(warm_restart_lr), \
                'ReduceLRCallback, len of restart_epoch and lr must be the same'
            warm_restarts = [warm_restart_epoch, warm_restart_lr]
            self.warm_restart_epoch = warm_restarts[0][0]
            self.warm_restart_lr = warm_restarts[1][0]
            if len(warm_restarts[0]) > 1:
                self.warm_restarts = [warm_restarts[0][1:], warm_restarts[1][1:]]
            else:
                self.warm_restarts = None   # only one restart
        else:
            self.warm_restart_epoch = warm_restart_epoch
            self.warm_restart_lr = warm_restart_lr
            self.warm_restarts = warm_restarts

        if avg_loss_num_epochs is not None:
            warnings.warn('avg_loss_num_epochs is deprecated - use nloss instead')

        self.loss_list = []
        self.nloss = nloss
        self.loss_func = loss_func
        self.warm_restart_thresh = warm_restart_thresh
        self.lr_log_name = lr_log_name
        self.save_weights_only = save_weights_only
        self.thresh = thresh
        self.thresh_inc_factor = thresh_inc_factor

        self.fname = fname
        self.epoch = 0
        self.saved_best = np.Inf
        self.worst = -np.Inf
        self.prev_loss = np.Inf
        self.first_call = True
        self.burn_in = burn_in
        self.recovery_decrease_factor = recovery_decrease_factor
        self.increase_epoch = increase_epoch
        self.increase_factor = increase_factor
        if self.verbose > 1:
            print(f'lr callback created with thresh {thresh}')
        if decrease_factor is None:
            self.decrease_factor = self.factor
        else:
            self.decrease_factor = decrease_factor

    def on_batch_end(self, batch, logs=None):
        super(ReduceLRWithModelCheckpointAndRecovery, self).on_batch_end(batch, logs)
        if self.first_call:
            if self.restart:
                print('restarting from old weights and opt...')
                self.best_fname = self.fname
                self._load_best()
                self.restart = False
                self.first_call = False
                self.init_limits(logs)

    #    if self.burn_in is None and self.epoch == 0:  # init worst/best value
    #        self.init_limits(logs)

    def on_batch_start(self, batch, logs=None):
        super(ReduceLRWithModelCheckpointAndRecovery, self).on_batch_start(batch, logs)
        current = K.eval(logs["loss"])
        if self.epoch <= 1 and self.batch <= 1:
            print(f'first batch {self.epoch}:{self.batch} - loss is {current}')

    def init_limits(self, logs):
        if self.verbose > 1:
            print('LR callback: initializing limits')

        current = logs.get(self.monitor) if self.loss_func is None else self.loss_func(self.model)
        if current is None:
            current = np.Inf

        # in some cases the loss at the end of the first batch is
        # pretty good, so only use it as worst if it is worse
        # than the threshold (assuming one is provided). 
        # Also check all batches in the first epoch and use the worst
        # as the threshold if thresh is None
        if self.thresh is None:
            if current > self.worst:
                if self.verbose > 1:
                    print(f'\n- lr_callback resetting worst to {current}')
                self.worst = current
        else:
            if current < self.thresh:
                self.worst = self.thresh
            elif current > self.worst:
                self.worst = current

        if self.restart:
            print('restarting from old weights and opt...')
            self.best_fname = self.fname
            self._load_best()
            self.restart = False
            self.first_call = False
        elif self.first_call and self.monitor_op(current, self.saved_best):
            self.first_call = False
            self._save_best(current, 0)

    def _save_best(self, current, epoch):
        if self.verbose > 1:
            print(f'\n - lr_callback: saving new best %f in epoch {epoch}' % current)
        self.saved_best = current
        self.best_epoch = epoch + 1
        if self.fname.find('%') >= 0:  # format string included in fname
            fname = self.fname % epoch  
        else:
            fname = self.fname

        self.best_fname = fname
        if self.save_weights_only:
            self.model.save_weights(fname)
        else:
            self.model.save(fname)
        self.opt_weights = self.model.optimizer.get_weights()
        np.savez(
            'opt.' + self.best_fname + '.npz', 
            opt=np.array(self.opt_weights, dtype=object), 
            lr=K.get_value(self.model.optimizer.lr), 
            saved_best=self.saved_best, 
            worst=self.worst, 
            best_epoch=self.best_epoch,
            loss_list=self.loss_list)

        if self.verbose > 1:
            print(f'\n - saving weights to {fname}')

    def _load_best(self):
        if self.verbose > 1:
            print(f'loading weights and optimizer from {self.best_fname}')
        self.model.load_weights(self.best_fname)
        dct = np.load('opt.' + self.best_fname + '.npz', allow_pickle=True)
        self.opt_weights = dct['opt']
        K.set_value(self.model.optimizer.lr, dct['lr'])
        self.model.optimizer.set_weights(self.opt_weights)
        self.worst = dct['worst']
        self.saved_best = dct['saved_best']
        self.best_epoch = dct['best_epoch']
        self.loss_list = dct['loss_list'].tolist()
        if self.verbose:
            print(f'restoring best_epoch {self.best_epoch}, saved_best {self.saved_best}, ' +
                  f'worst {self.worst}, lr {K.get_value(self.model.optimizer.lr)}')

    def on_epoch_start(self, epoch, logs=None):
        super(ReduceLRWithModelCheckpointAndRecovery, self).on_epoch_start(batch, logs)
        self.epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        super(ReduceLRWithModelCheckpointAndRecovery, self).on_epoch_end(epoch, logs)

        old_lr = K.get_value(self.model.optimizer.lr)
        logs[self.lr_log_name] = K.get_value(self.model.optimizer.lr)
        self.epoch = epoch

        current = logs.get(self.monitor) if self.loss_func is None else self.loss_func(self.model)
        is_nan = tf.math.is_nan(current)
        if not is_nan:
            self.loss_list.append(current)

        if self.nloss > 1:
            if self.use_median:
                current = np.median(np.array(self.loss_list[-self.nloss:]))
            else:
                current = np.array(self.loss_list[-self.nloss:]).mean()

        if len(self.loss_list) > 0 and self.nloss > 1:
            prev_loss = np.median(self.loss_list[-self.nloss:])
        else:
            prev_loss = self.prev_loss

        new_best = self.monitor_op(current, self.saved_best) if epoch > 1 else True
        if new_best:
            if self.loss_func is not None:
                print(f'\n\tnew best external loss {current}, saving weights')

            self._save_best(current, epoch)


        if self.verbose > 1:
            print('\nepoch %d: best %2.3e (epoch %d), worst %2.3e, current %2.3e, prev %2.3f' 
                  % (epoch, self.saved_best, self.best_epoch, self.worst, current, prev_loss))

        bad_step = False
        if self.epoch > self.burn_in or is_nan:
            if not new_best:   # not a new best - check to see if worst
                # revert to prev weights if now at worst ever or
                # if a threshold has been set and we excede it and
                # are not improving
                increasing = current > prev_loss
                thresh = self.thresh if self.thresh is not None else self.worst

                # if self.thresh_incr_factor is set (not None) set the
                # threshold using it instead
                if thresh is not None:
                    bad_step = current > thresh
                    if self.verbose > 0 and bad_step:
                        print(f'\nabsolute threshold exceeded, thresh is {self.thresh}' + 
                              f' worst is {self.worst}')
                else:
                    bad_step = False

                if self.thresh_inc_factor is not None and not bad_step:
                    if current > 0:
                        inc_thresh = prev_loss + prev_loss * self.thresh_inc_factor
                    else:
                        inc_thresh = prev_loss - prev_loss * self.thresh_inc_factor

                    bad_step = current > inc_thresh
                    if self.verbose > 0 and bad_step:
                        print('\nrelative threshold exceeded')
                else:
                    inc_thresh = 0  # not used

                # revert to previous best and reduce lr
                if is_nan or (bad_step and increasing):
                    if self.prev_fname:
                        self.model.load_weights(self.prev_fname)
                    else:
                        self._load_best()

                    if self.recovery_decrease_factor is None:
                        decrease_factor = self.decrease_factor
                    else:
                        decrease_factor = self.recovery_decrease_factor

                    new_lr = max(old_lr * decrease_factor, self.min_lr)
                    K.set_value(self.model.optimizer.lr, new_lr)
                    if self.verbose > 0:
                        print(f'\nlr_callback: loss function too large ')
                        print(f'(%2.3e > %2.3e/%2.3e) - reverting to %s @ %2.3e at epoch %d'
                              % (current, thresh, inc_thresh, self.best_fname, self.saved_best, 
                                 self.best_epoch))
                        print('Epoch %05d: ReduceLRWithModelCheckpointAndRecovery reducing'
                              ' learning rate to %2.3e.' % (epoch + 1, new_lr))

                    current = self.saved_best  # otherwise won't catch second adjacent divergent 

        if self.increase_epoch == epoch:
            new_lr = old_lr * self.increase_factor
            K.set_value(self.model.optimizer.lr, new_lr)
            if self.verbose > 0:
                print(' - increasing lr to %2.3e ' % new_lr)

        # only keep track of the worst for thresholding during the burn in period
        # otherwise when things diverge the threshold keeps getting worse also
        if (self.burn_in is None or epoch <= self.burn_in) and \
           (current > self.worst or not np.isfinite(self.worst)):
            if self.verbose > 0:
                print(f' - new worst {self.worst} <-- {current}')
                self.worst = current

        if self.prev_fname and not bad_step:
            self.model.save_weights(self.prev_fname)

        self.prev_loss = current

        try_restart = (self.warm_restart_epoch is not None) or \
                      (self.warm_restart_thresh is not None)
        if try_restart:  # see if it is time to bump up lr
            if self.warm_restart_epoch is not None:
                warm_restart = (epoch + 1) == self.warm_restart_epoch
            else:
                warm_restart = current < self.warm_restart_thresh
            if warm_restart:
                print(f'\ntrying warm restart: setting lr to {self.warm_restart_lr}')
                K.set_value(self.model.optimizer.lr, self.warm_restart_lr)
                self.model.compile(optimizer=self.model.optimizer, 
                                   loss=self.model.loss)
                self.warm_restart_thresh = None
                self.warm_restart_epoch = None  # only restart once
                if self.warm_restarts is not None:
                    self.warm_restart_epoch = self.warm_restarts[0][0]
                    self.warm_restart_lr = self.warm_restarts[1][0]
                    if len(self.warm_restarts[0]) == 1:
                        self.warm_restarts = None
                    else:
                        self.warm_restarts = [self.warm_restarts[0][1:], 
                                              self.warm_restarts[1][1:]]


class ReserveCapacity(tf.keras.callbacks.Callback):
    '''
    keras callback that set some pct of the models conv layer to be non-trainable
    then gradually add them back in.

    wt_scale         - amount to scale the untrainable weights by (not used yet)
    reserve_capacity - the pct of conv layers to set as untrainable initially
    reserve_epochs   - how many epochs to wait before adding layers back in
    reserve_pct      - the pct of the reserve capacity to add in every reserve_epochs
    reserve_num      - if specified will override reserve_pct (and recompute it)
    order            - order in which layers should be added back in. Options are
                       'random' and 'fixed'. If fixed, layers will be added by
                       spatial scale. So 1 at every scale before going to the second at 
                       each scale
    verbose          - whether to print messages or not
    '''
    def __init__(self,
                 wt_scale=1e-2,
                 reserve_capacity=0.5,   # 50%
                 reserve_epochs=10,     # increase capacity every 10th epoch
                 reserve_pct=.1,        # add back 10% of capacity
                 reserve_num=None,      # if specified use this instead of pct
                 verbose=0,
                 order='fixed',         # could be 'random' also
                 layer_name_substr=None,
                 reset_lr=True,
                 min_lr=None,
                 reset_optimizer=False,
                 **kwargs):
        super(ReserveCapacity, self).__init__(**kwargs)
        self.verbose = verbose
        self.epoch = 0
        self.order = order
        self.wt_scale = wt_scale
        self.reset_optimizer = reset_optimizer
        self.reserve_epochs = reserve_epochs
        self.reserve_capacity = reserve_capacity
        self.min_lr = min_lr
        self.reset_lr = reset_lr
        self.layer_name_substr = layer_name_substr
        self.reserve_num = reserve_num
        self.reserve_pct = reserve_pct

    def on_train_begin(self, logs={}):
        ndims = len(self.model.inputs[0].get_shape()) - 2
        if ndims == 3:
            self.layer_type = tf.python.keras.layers.convolutional.Conv3D
        elif ndims == 2:
            self.layer_type = tf.python.keras.layers.convolutional.Conv2D
        elif ndims == 1:
            self.layer_type = tf.python.keras.layers.convolutional.Conv1D
        else:
            assert 0, f'ReserveCapacity: ndims {ndims} not supported'

        # build a list of all conv layers, then randomize them and pick
        # out the percent specified by the caller
        if self.layer_name_substr is not None:
            layer_list = nes.utils.find_layers_with_substring(vxm_model, self.layer_name_substr)
        else:
            layer_list = self._build_layer_list(self.model, self.layer_type)

        self.nreserve = int(len(layer_list) * self.reserve_capacity)
        self.reserve_layer_list = layer_list.copy()

        if self.order == 'random':
            random.shuffle(self.reserve_layer_list)
        else:  # fixed order 
            self.reserve_layer_list = self._order_list(self.reserve_layer_list)

        self.reserve_layer_list = self.reserve_layer_list[:self.nreserve]
        self.reserve_increase_num = max(1, int(self.reserve_pct * self.nreserve)) \
            if self.reserve_num is None else self.reserve_num
        for lno in range(self.nreserve):
            self.reserve_layer_list[lno].trainable = False
            w = self.reserve_layer_list[lno].get_weights()
            wscale = [wl * self.wt_scale for wl in w]
            self.reserve_layer_list[lno].set_weights(wscale) 

        if self.verbose:
            print(f'ReserveCapacity: {len(layer_list)} found, {self.nreserve} in reserve,'
                  f' {self.reserve_increase_num} to be added every {self.reserve_epochs} epochs')

    def _order_list(self, llist):
        prev_shape = None
        added = len(llist) * [False]
        ordered_list = []
        for lno in range(len(llist)):
            if not (llist[lno].input_shape[1:-1] == prev_shape):
                ordered_list.append(llist[lno])
                added[lno] = True

            prev_shape = llist[lno].input_shape[1:-1]

        while len(ordered_list) < len(llist):
            new_added = added.copy()
            for lno in range(1, len(llist)):
                # add this one if the prev was added last round
                if added[lno - 1] and not added[lno]:  
                    new_added[lno] = 0
                    ordered_list.append(llist[lno])

        return ordered_list

    def _build_layer_list(self, model, layer_type):
        # build a list of all layers of a given type in a model (recursively if needed)
        layer_list = []
        for layer in model.layers:
            if hasattr(layer, 'layers'):
                layer_list += self._build_layer_list(layer, layer_type)
            elif type(layer) == layer_type:
                layer_list.append(layer)

        return layer_list

    def on_batch_end(self, batch, logs=None):
        super(ReserveCapacity, self).on_batch_end(batch, logs)

    def on_epoch_end(self, epoch, logs=None):
        super(ReserveCapacity, self).on_epoch_end(epoch, logs)
        if epoch == 0 and self.reset_optimizer:
            self.optimizer = self.model.optimizer
            self.opt_weights = self.model.optimizer.get_weights()
            self.lr = self.model.optimizer.lr

        if np.mod(epoch + 1, self.reserve_epochs) == 0:   # add some layers back in
            if self.min_lr is not None and self.model.optimizer.lr > self.min_lr:
                print(f'\nreserve not added yet as current learning rate ' +
                      f'{self.model.optimizer.lr} > {self.min_lr}')
                return  # if min_lr is set, don't add any reserve until lr is below it

            if len(self.reserve_layer_list) > 0:
                for lno in range(self.reserve_increase_num):
                    if len(self.reserve_layer_list) > 0:
                        layer = self.reserve_layer_list.pop()
                        layer.trainable = True

                if self.verbose > 0:
                    print(f'\nepoch {epoch+1}: adding {self.reserve_increase_num} reserve layers '
                          f'({len(self.reserve_layer_list)} remaining)')

            if self.reset_optimizer:
                print(f'resetting optimizer {self.model.optimizer._name}')
                optimizer = self.optimizer
                optimizer.set_weights(self.opt_weights)
                if self.reset_lr:
                    print(f'resetting learning rate to {self.lr}')
                    K.set_value(optimizer.lr, self.lr)
                else:
                    K.set_value(optimizer.lr, self.model.optimizer.lr)
                self.model.compile(optimizer=optimizer, loss=self.model.loss)
            else:
                self.model.compile(optimizer=self.model.optimizer, loss=self.model.loss)


class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


class WriteHist(tf.keras.callbacks.Callback):
    '''
       keras callback that will write losses and learning rates
       into a log file at the end of every epoch
    parameters:
       fname - filename to write into
       mode  - mode to use in opening file ('a' for appending)
    '''

    def __init__(self, fname, mode='w'):
        super(WriteHist, self).__init__()
        self._supports_tf_logs = True
        self.fname = fname
        self.inited = True if mode == 'a' else False
        self.fp = None
        self.mode = mode

    def on_epoch_end(self, epoch, logs=None):
        key_list = list(logs.keys())
        val_list = list(logs.values())

        # write header row of column names into file once
        if self.inited is False or not os.path.exists(self.fname):
            self.inited = True
            self.fp = open(self.fname, self.mode)
            key_str = '# epoch'
            for key in key_list:
                key_str += ' ' + key
            self.fp.write('%s\n' % key_str)
        elif self.fp is None:  # reopen file that we previously opened
            self.fp = open(self.fname, "a")

        # now write the losses and lr and such
        self.fp.write('%d ' % epoch)
        for key in key_list:
            val_index = key_list.index(key)
            val = val_list[val_index]
            if hasattr(val, 'numpy'):
                val = K.eval(val)
            self.fp.write('%s ' % (str(val)))

        self.fp.write('\n')
        self.fp.close()
        self.fp = None   # mark it as closed so we will reopen it

    def on_train_end(self, logs=None):
        if self.fp is not None:
            self.fp.close()
        self.fp = None
