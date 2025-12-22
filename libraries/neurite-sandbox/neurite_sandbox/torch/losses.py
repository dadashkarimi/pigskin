class NCCGuhaExperimental:
    """
    Local (over window) normalized cross correlation loss.

    This is a small "fix" to the NCC in vxm.

    Messages from Guha on slack, 10/18/2020:
    i feel like it shouldnt be necessary to compute so many volumes since we are dealing with 
    averages and stuff, and there should be some simplification somewhere
    but i dont see it right now
    equation 6 in our paper is what i was looking at: https://arxiv.org/pdf/1809.05231.pdf
    ask your guy to expand out the quadratics and make sure what Ii did looks right
    oh this is the torch, i can do the keras one
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        Ii2 = Ii * Ii
        Ji2 = Ji * Ji
        IiJi = Ii * Ji

        Ii_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        Ji_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        Ii2_sum = conv_fn(Ii2, sum_filt, stride=stride, padding=padding)
        Ji2_sum = conv_fn(Ji2, sum_filt, stride=stride, padding=padding)
        IiJi_sum = conv_fn(IiJi, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_Ii = Ii_sum / win_size
        u_Ji = Ji_sum / win_size

        # New: Compute these additional volumes so that the NCC loss exactly
        # matches paper's formula.
        Ii_u_Ii_sum = conv_fn(Ii * u_Ii, sum_filt, stride=stride, padding=padding)
        Ji_u_Ji_sum = conv_fn(Ji * u_Ji, sum_filt, stride=stride, padding=padding)
        Ii_u_Ji_sum = conv_fn(Ii * u_Ji, sum_filt, stride=stride, padding=padding)
        Ji_u_Ii_sum = conv_fn(Ji * u_Ii, sum_filt, stride=stride, padding=padding)
        u_Ii_u_Ji_sum = conv_fn(u_Ii * u_Ji, sum_filt, stride=stride, padding=padding)
        u_Ii_u_Ii_sum = conv_fn(u_Ii * u_Ii, sum_filt, stride=stride, padding=padding)
        u_Ji_u_Ji_sum = conv_fn(u_Ji * u_Ji, sum_filt, stride=stride, padding=padding)

        cross = IiJi_sum - Ii_u_Ji_sum - Ji_u_Ii_sum + u_Ii_u_Ji_sum
        Ii_var = Ii2_sum - 2 * Ii_u_Ii_sum + u_Ii_u_Ii_sum
        Ji_var = Ji2_sum - 2 * Ji_u_Ji_sum + u_Ji_u_Ji_sum

        cc = cross * cross / (Ii_var * Ji_var + 1e-5)

        return -torch.mean(cc)
