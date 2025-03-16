import torch
import torch.nn.functional as F


def build_r_coeffs_hook(y_batch: list[int],
                        partition: list[int]):
    features = None  # dict
    
    def _r_coeffs_hook(module, input, output):
        _r_coeffs = output[0].squeeze()[0, :].detach()
        _r_coeffs = torch.stack(torch.split(_r_coeffs, partition[1]))  # TEMP
        _r_coeffs = torch.abs(_r_coeffs)
        _r_hot = torch.linalg.norm(_r_coeffs, dim=1)

        y = y_batch[0].item()
        _r_mask = torch.zeros_like(_r_hot)
        _r_mask[y] = 1
        print("Heat of _r coefficients: {}\nLabel mask: {}".format(_r_hot, _r_mask))
            
    return features, _r_coeffs_hook  # common hook builder return signature


def build_softmax_hook():
    features = None  # dict

    def softmax_hook(module, input, output):
        prob = F.softmax(output,dim=1)
        print("Batch p-distributions: {}".format(prob))

    return features, softmax_hook
