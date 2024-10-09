import torch
import numpy as np

num_inputs, number_hidden, num_output = 128, 256, 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_params():
    def _one(shape):
        ts = torch.tensor(np.random.normal(0, 1, size=shape))
        return torch.nn.Parameter(ts, requires_grad=True)

    def _three(shape):
        return (_one((num_inputs, num_output)),
                _one((number_hidden, number_hidden)),
                torch.nn.Parameter(torch.zeros(number_hidden, requires_grad=True))
                )


def lstm(inputs, state, params):
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f,
     W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q] = params
    (H, C) = state

    outputs = []

    for x in inputs:
        I = torch.sigmoid(torch.matmul(x.float(), W_xi) + torch.matmul(H.float(), W_hi) + b_i.float())
        F = torch.sigmoid(torch.matmul(x.float(), W_xf) + torch.matmul(H.float(), W_hf) + b_f.float())
        O = torch.sigmoid(torch.matmul(x.float(), W_xo) + torch.matmul(H.float(), W_ho) + b_o.float())
        C_tilda = torch.tanh(torch.matmul(x.float(), W_xc) + torch.matmul(H.float(), W_hc) + b_c.float())

        C = F * C + I * C_tilda

        H = O * torch.tanh(C)
        Y = torch.matmul(H.float(), W_hq) + b_q.float()

        outputs.append(Y)

    return outputs, (H, C)
