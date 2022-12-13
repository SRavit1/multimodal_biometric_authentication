import torch
import models.resnet
import compile_utils
import binarized_modules

xy = 4
z = 32
kz = 32
kxy = 1
pd = 0
pl = 2

conv_layer = binarized_modules.BinarizeConv2d(1, 1, 1, z, kz, kxy, stride=1, padding=pd, bias=False)
bn_layer = torch.nn.BatchNorm2d(kz).eval()
with torch.no_grad():
    for p in [bn_layer.running_mean, bn_layer.running_var, bn_layer.weight, bn_layer.bias]:
        p.copy_(torch.rand(p.shape))
pool_layer = torch.nn.MaxPool2d(pl, stride=pl)
model = torch.nn.Sequential(
    conv_layer,
    pool_layer,
    bn_layer,
    #torch.nn.Hardtanh(),
)
model.eval()

layer = "test"
layer_input = (torch.randint(low=0, high=2, size=(1, z, xy, xy)).float()-0.5)
compile_utils.compile_conv_block(conv_layer, bn_layer, layer_input, pool_layer=pool_layer, label=layer, save_to="/home/sravit/3pxnet/3pxnet-inference/examples/conv" + layer + ".h", print_=False)

exit(0)

prec_config = {
        "conv1": {"q_scheme": "bwn", "bias": False, "weight_bw": 1, "activation_type": "relu", "leaky_relu_slope": 0.},
        "layer1": {"q_scheme": "xnor", "bias": False, "act_bw": 1, "weight_bw": 1, "activation_type": "relu", "leaky_relu_slope": 0.},
        "layer2": {"q_scheme": "xnor", "bias": False, "act_bw": 1, "weight_bw": 1, "activation_type": "relu", "leaky_relu_slope": 0.},
        "layer3": {"q_scheme": "xnor", "bias": False, "act_bw": 1, "weight_bw": 1, "activation_type": "relu", "leaky_relu_slope": 0.},
        "layer4": {"q_scheme": "xnor", "bias": False, "act_bw": 1, "weight_bw": 1, "activation_type": "relu", "leaky_relu_slope": 0.},
        "fc": {"q_scheme": "xnor", "bias": False, "act_bw": 1, "weight_bw": 1, "activation_type": "relu", "leaky_relu_slope": 0.}
    }

resnet10_conv_layers = ["2_1", "2_2", "3_1", "3_2", "4_1", "4_2", "4_d", "5_1", "5_2"]
resnet10_layer_index_map = {
    "conv1": 1,
    "bn1": 2,
    "conv2_1": 7,
    "bn2_1": 8,
    "conv2_2": 10,
    "bn2_2": 11,
    "conv3_1": 14,
    "bn3_1": 15,
    "conv3_2": 17,
    "bn3_2": 18,
    "conv4_1": 22,
    "bn4_1": 23,
    "conv4_2": 25,
    "bn4_2": 26,
    "conv4_d": 28,
    "bn4_d": 29,
    "conv5_1": 32,
    "bn5_1": 33,
    "conv5_2": 35,
    "bn5_2": 36
}
resnet10_layer_input_shape_map = {
    "conv1": (1, 3, 224, 224),
    "conv2_1": (1, 64, 56, 56),
    "conv2_2": (1, 64, 56, 56),
    "conv3_1": (1, 64, 56, 56),
    "conv3_2": (1, 64, 56, 56),
    "conv4_1": (1, 64, 56, 56),
    "conv4_2": (1, 128, 28, 28),
    "conv4_d": (1, 64, 56, 56),
    "conv5_1": (1, 128, 28, 28),
    "conv5_2": (1, 128, 28, 28)
}

model = models.resnet.resnetCustomLayers(layers=[2, 2], prec_config=prec_config)
model.load_state_dict(torch.load("/home/sravit/models/resnet10.pth"))
model.eval()

x = torch.ones((1, 3, 224, 224))
model.forward(x)

layers_list = list(model.modules())
for layer in ["2_1"]: #resnet10_conv_layers:
    layer_input = torch.full(resnet10_layer_input_shape_map["conv" + layer], -1).float()
    compile_utils.compile_conv_block(layers_list[resnet10_layer_index_map["conv" + layer]], layers_list[resnet10_layer_index_map["bn" + layer]], layer_input, label=layer, save_to="/home/sravit/3pxnet/3pxnet-inference/examples/conv" + layer + ".h", print_=False)

#torch.save(model.state_dict(), "/home/sravit/models/resnet10.pth")
#torch.onnx.export(model, torch.zeros((1, 3, 224, 224)), "/home/sravit/models/resnet10.onnx")