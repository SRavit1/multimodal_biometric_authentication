import torch
import models.resnet
import compile_utils
import binarized_modules

prec_config = {
        "conv1": {"q_scheme": "bwn", "bias": False, "weight_bw": 1, "activation_type": "relu", "leaky_relu_slope": 0.},
        "layer1": {"q_scheme": "xnor", "bias": False, "act_bw": 1, "weight_bw": 1, "activation_type": "relu", "leaky_relu_slope": 0.},
        "layer2": {"q_scheme": "xnor", "bias": False, "act_bw": 1, "weight_bw": 1, "activation_type": "relu", "leaky_relu_slope": 0.},
        "layer3": {"q_scheme": "xnor", "bias": False, "act_bw": 1, "weight_bw": 1, "activation_type": "relu", "leaky_relu_slope": 0.},
        "layer4": {"q_scheme": "xnor", "bias": False, "act_bw": 1, "weight_bw": 1, "activation_type": "relu", "leaky_relu_slope": 0.},
        "fc": {"q_scheme": "bwn", "bias": False, "act_bw": 1, "weight_bw": 1, "activation_type": "relu", "leaky_relu_slope": 0.}
    }
resnet10_conv_layers = ["2_1", "2_2", "3_1", "3_2", "4_1", "4_2", "4_d", "5_1", "5_2"]
resnet10_layer_index_map = {
    "conv1": 1,
    "bn1": 2,
    "pool1": 4,
    "conv2_1": 7,
    "bn2_1": 8,
    "conv2_2": 10,
    "bn2_2": 11,
    "conv3_1": 14,
    "bn3_1": 15,
    "conv3_2": 17,
    "bn3_2": 18,
    "conv4_1": 22,
    "pool4_1": 23,
    "bn4_1": 24,
    "conv4_2": 26,
    "bn4_2": 27,
    "conv4_d": 29,
    "pool4_d": 30,
    "bn4_d": 31,
    "conv5_1": 34,
    "bn5_1": 35,
    "conv5_2": 37,
    "bn5_2": 38,
    "fc": 45
}
resnet10_layer_input_shape_map = {
    "conv1": (1, 3, 224, 224),
    "conv2_1": (1, 64, 112, 112),
    "conv2_2": (1, 64, 112, 112),
    "conv3_1": (1, 64, 112, 112),
    "conv3_2": (1, 64, 112, 112),
    "conv4_1": (1, 64, 112, 112),
    "conv4_2": (1, 128, 56, 56),
    "conv4_d": (1, 64, 112, 112),
    "conv5_1": (1, 128, 56, 56),
    "conv5_2": (1, 128, 56, 56)
}

model = models.resnet.resnetCustomLayers(layers=[2, 2], prec_config=prec_config)
#model.load_state_dict(torch.load("/home/sravit/models/resnet10.pth"))
model.eval()

layers_list = list(model.modules())

#model_input = torch.rand(resnet10_layer_input_shape_map["conv1"])-0.5
#model.forward(model_input)

layer_input = torch.rand(resnet10_layer_input_shape_map["conv2_1"])-0.5

expected_output = layer_input.clone()
for block_index in [6, 13, 21, 33]:
    block = layers_list[block_index]
    expected_output = block(expected_output)
#print(compile_utils.convert_conv_act(expected_output, binarize=False)[-20:])


layer_input = compile_utils.compile_identity_block(layer_input, ["2_1", "2_2"], layers_list, resnet10_layer_index_map)
layer_input = compile_utils.compile_identity_block(layer_input, ["3_1", "3_2"], layers_list, resnet10_layer_index_map)
layer_input = compile_utils.compile_residual_block(layer_input, ["4_1", "4_2"], ["4_d"], layers_list, resnet10_layer_index_map)
layer_input = compile_utils.compile_identity_block(layer_input, ["5_1", "5_2"], layers_list, resnet10_layer_index_map)
avgpool_output = torch.nn.AdaptiveAvgPool2d((1, 1))(layer_input)
avgpool_output = avgpool_output.flatten(1)
fc_output = compile_utils.compile_bwn_fc(layers_list[resnet10_layer_index_map["fc"]], avgpool_output, "", False)

x = fc_output
x_norm = torch.sqrt(torch.sum(torch.mul(x,x), dim=1) + 1e-3)  #torch.linalg.norm(x)
x_norm = torch.unsqueeze(x_norm, 1)
x = torch.div(x, x_norm)

print(x.flatten())

#print(compile_utils.convert_conv_act(layer_input, binarize=False))

#print(compile_utils.convert_conv_act(layer_input, binarize=False)[-20:])
assert(torch.allclose(layer_input, expected_output))

#torch.save(model.state_dict(), "/home/sravit/models/resnet10.pth")
#torch.onnx.export(model, torch.zeros((1, 3, 224, 224)), "/home/sravit/models/resnet10.onnx", opset_version=11)c