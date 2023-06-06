import torch
import torch.nn as nn


def init_lora_weights(layer_in, layer_out):
    nn.init.zeros_(layer_in.weight)
    nn.init.normal_(layer_out.weight)

def linear_forward_hook(module, intsr, outtsr):
    module.input = intsr[0]

def linear_backward_hook(layer, grad_input, grad_output):
    grad_output = grad_output[0] # len, n, outdim
    grad_input = layer.input #len, n, indim


    A, B = grad_input.half(), grad_output.half()


    if len(A.shape) == 3:
        # by default, linear is not batch first
        layer_batch_dim =  0
    else:
        layer_batch_dim =  0

    if layer_batch_dim == 1:
        # k: tokens-per-sample
        # n: batch size
        gs = torch.einsum("kn...i,kn...j->nij", B, A)
        if layer.bias is not None:
            gs_bias = torch.einsum("kn...i->ni", B)
    else:
        gs = torch.einsum("n...i,n...j->nij", B, A)
        if layer.bias is not None:
            gs_bias = torch.einsum("n...k->nk", B)

    layer.weight.grad_sample = gs.float()
    if layer.bias is not None:
        layer.bias.grad_sample = gs_bias.float()

    

def register_hooks(layers):
    for layer in layers:
        if(isinstance(layer, nn.Linear)):
            layer.register_forward_hook(linear_forward_hook)
            layer.register_backward_hook(linear_backward_hook)
        else:
            raise 'not implemented'

 

# layer = nn.Linear(256, 128).cuda()
# layer.register_forward_hook(linear_forward_hook)
# layer.register_backward_hook(linear_backward_hook)   


# noise = torch.normal(0, 1, size=(32, 64, 256)).cuda()

# loss = torch.mean(torch.sum(layer(noise), dim=[1,2]))

# loss.backward()

# print(layer.weight.grad.norm(), torch.sum(layer.weight.grad_sample, dim=0).norm())

