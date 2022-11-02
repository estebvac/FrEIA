# These imports and declarations apply to all examples
import numpy as np
import torch
import torch.nn as nn

import FrEIA.framework as Ff
import FrEIA.modules as Fm

print("Numpy version: ", np.__version__)
print("Torch version: ", torch.__version__)


def subnet_fc(c_in, c_out):
    return nn.Sequential(nn.Linear(c_in, 512), nn.ReLU(),
                         nn.Linear(512, c_out))


def subnet_conv(c_in, c_out):
    return nn.Sequential(nn.Conv2d(c_in, 256, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(256, c_out, 3, padding=1))


def subnet_conv_1x1(c_in, c_out):
    return nn.Sequential(nn.Conv2d(c_in, 256, 1), nn.ReLU(),
                         nn.Conv2d(256, c_out, 1))


### THE BASIC 2D EXAMPLE: ###
print(100 * "*", "\n Basic 2D example \n", 100 * "*")

inn = Ff.SequenceINN(2)
for k in range(2):  # 8
    inn.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, permute_soft=True)

x = torch.rand(4, 2)
y = inn(x)
print(f"""
inn = {inn}
x.shape = {x.shape}, \t  x = {x} ,
y.shape = {y[0].shape}, jacobian.shape={y[1].shape} \t y = {y}

""")

# Conditional INN for MNIST
print(100 * "*", "\n Conditional INN for MNIST \n", 100 * "*")

in_channels = 28 * 28
conditional = 10
feat = (28 * 28) / 2 + conditional

print(f"in_channels = {in_channels}, conditional= {conditional},  feat = in_channels/2 + cond =  {feat}")

cinn = Ff.SequenceINN(28 * 28)

for k in range(3):  # 12
    cinn.append(Fm.AllInOneBlock, cond=0, cond_shape=(10,), subnet_constructor=subnet_fc)

x = torch.rand(2, in_channels)
c = torch.rand(2, conditional)
y = cinn(x, c=[c])
print(f"""
inn = {cinn}
x.shape = {x.shape}, \t  x = {x} ,
y.shape = {y[0].shape}, jacobian.shape={y[1].shape} \t y = {y}

""")

# Convolutional INN with invertible downsampling
print(100 * "*", "\n Convolutional INN with invertible downsampling \n", 100 * "*")

nodes = [Ff.InputNode(3, 32, 32, name='input')]
ndim_x = 3 * 32 * 32

# Higher resolution convolutional part
for k in range(1):  # 4
    nodes.append(Ff.Node(nodes[-1],
                         Fm.GLOWCouplingBlock,
                         {'subnet_constructor': subnet_conv, 'clamp': 1.2},
                         name=F'conv_high_res_{k}'))
    nodes.append(Ff.Node(nodes[-1],
                         Fm.PermuteRandom,
                         {'seed': k},
                         name=F'permute_high_res_{k}'))

nodes.append(Ff.Node(nodes[-1], Fm.IRevNetDownsampling, {}))

# Lower resolution convolutional part
for k in range(2):  # 12
    if k % 2 == 0:
        subnet = subnet_conv_1x1
    else:
        subnet = subnet_conv

    nodes.append(Ff.Node(nodes[-1],
                         Fm.GLOWCouplingBlock,
                         {'subnet_constructor': subnet, 'clamp': 1.2},
                         name=F'conv_low_res_{k}'))
    nodes.append(Ff.Node(nodes[-1],
                         Fm.PermuteRandom,
                         {'seed': k},
                         name=F'permute_low_res_{k}'))

# Make the outputs into a vector, then split off 1/4 of the outputs for the
# fully connected part
nodes.append(Ff.Node(nodes[-1], Fm.Flatten, {}, name='flatten'))
split_node = Ff.Node(nodes[-1],
                     Fm.Split,
                     {'section_sizes': (ndim_x // 4, 3 * ndim_x // 4), 'dim': 0},
                     name='split')
nodes.append(split_node)

# Fully connected part
for k in range(3):  # 12
    nodes.append(Ff.Node(nodes[-1],
                         Fm.GLOWCouplingBlock,
                         {'subnet_constructor': subnet_fc, 'clamp': 2.0},
                         name=F'fully_connected_{k}'))
    nodes.append(Ff.Node(nodes[-1],
                         Fm.PermuteRandom,
                         {'seed': k},
                         name=F'permute_{k}'))

# Concatenate the fully connected part and the skip connection to get a single output
nodes.append(Ff.Node([nodes[-1].out0, split_node.out1],
                     Fm.Concat1d, {'dim': 0}, name='concat'))
nodes.append(Ff.OutputNode(nodes[-1], name='output'))

conv_inn = Ff.GraphINN(nodes)

x = torch.zeros((4, 3, 32, 32))
y = conv_inn(x)

print(f"""
inn = {conv_inn}
x.shape = {x.shape}, \t  x = {x} ,
y.shape = {y[0].shape}, jacobian.shape={y[1].shape} \t y = {y}
""")

# define a graph INN
print(100 * "*", "\n define a graph INN \n", 100 * "*")

BATCHSIZE = 10
DIMS_IN = 2

input_1 = Ff.InputNode(DIMS_IN, name='input_1')
input_2 = Ff.InputNode(DIMS_IN, name='input_2')

print(input_1)

channels = 16
input_dim = (channels, 128, 128)
in_node = Ff.InputNode(*input_dim, name='input')
n_el = channels // 2 + 2
split_node = Ff.Node(in_node,
                     Fm.Split,
                     {'section_sizes': (n_el, channels - n_el), 'dim': 0},
                     name='split')

out_node = Ff.OutputNode(split_node.out0, name='output')
out_node1 = Ff.OutputNode(split_node.out1, name='output')
conv_inn = Ff.GraphINN([in_node, split_node, out_node, out_node1])

batch = torch.ones(tuple([2, ] + list(input_dim)))
result = conv_inn(batch)
part1 = result[0][0]
part2 = result[0][1]
# glow = Ff.Node(in_node, Fm.GLOWCouplingBlock,
#                {'subnet_constructor': subnet_fc, 'clamp': 2.0},
#                name=F'coupling_{k}'
#                )  # Can be also used: in_node.out0
#
# Fm.Split()
