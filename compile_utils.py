import torch

def pack(arr, pckWdt=32):
    if not len(arr)%pckWdt == 0:
        return None
    packs = []
    for i in range(int(len(arr)/pckWdt)):
        pack_dec = 0
        for j in range(pckWdt):
            pack_dec += arr[i*pckWdt + j]*pow(2, pckWdt-1-j) if arr[i*pckWdt + j] == 1 else 0
        pack = str(hex(int(pack_dec)))
        packs.append(pack)
    return ", ".join(packs)

def convert_fc_act(x, binarize=True):
    x = x[0].flatten().tolist()
    if binarize:
        x = [0 if elem<0 else 1 for elem in x]
    return x

def convert_conv_act(x, binarize=True):
    x = x[0]
    x = x.permute((2, 1, 0)) #CHW -> WHC, BWN
    x = x.flatten().tolist()
    if binarize:
        x = [0 if elem<0 else 1 for elem in x]
    return x

def convert_fc_weight(w):
    w = w.flatten().tolist()
    w = [0 if elem<0 else 1 for elem in w]
    return w

def convert_conv_weight(w):
    w = w.permute((0, 3, 2, 1)) #NCHW -> NWHC
    w = w.flatten().tolist()
    w = [0 if elem<0 else 1 for elem in w]
    return w

def convert_bn(mu, sigma, gamma, beta, binarize_input=True):
    mult = 4 if binarize_input else 2 # weights always binarized, input sometimes binarized
    #thr = ((-1*beta*torch.sqrt(sigma+1e-5) / torch.sqrt(torch.pow(gamma, 2) + 1e-4)) + mu).tolist()
    thr = ((-1*beta*torch.sqrt(sigma+1e-5) / torch.sqrt(torch.pow(gamma, 2) + 1e-5)) + mu).tolist()
    sign = [0 if elem<0 else 1 for elem in (torch.sign(gamma)).tolist()]
    return [e*mult for e in thr], sign

def convert_bn_float(mu, sigma, gamma, beta):
    # second return value is not really sigma but sigma+1e-5
    return (mu*4).tolist(), (torch.sqrt(sigma + 1e-5)*4).tolist(), gamma.tolist(), beta.tolist()

def compile_conv_block(conv_layer, bn_layer, x, label, print_=True, save_to=None, pool_layer=None):
    #y = conv_block.forward(x)
    
    conv1_out = conv_layer(x)
    if pool_layer:
        conv1_out = pool_layer(conv1_out)
    conv1_out_bn = bn_layer(conv1_out)
    y = conv1_out_bn

    x_no_bin_inf = convert_conv_act(x, binarize=False)
    x_inf = convert_conv_act(x, binarize=True)
    w_inf = convert_conv_weight(conv_layer.weight)
    mu, sigma, gamma, beta = bn_layer.running_mean.detach().clone(), bn_layer.running_var.detach().clone(), bn_layer.weight.detach().clone(), bn_layer.bias.detach().clone()
    mean_inf, sigma_inf, gamma_inf, beta_inf = convert_bn_float(mu, sigma, gamma, beta)
    bn_th_inf, bn_sign_inf = convert_bn(mu, sigma, gamma, beta, True)
    #bn_th_inf, bn_sign_inf = convert_bn(bn_layer.running_mean, bn_layer.running_var, bn_layer.weight, bn_layer.bias, conv_layer.binarize_input)
    bn_sign_inf_pack = pack(bn_sign_inf)
    y_no_bin_inf = convert_conv_act(y, binarize=False)
    y_inf = convert_conv_act(y)

    x_inf_pack = pack(x_inf)
    w_inf_pack = pack(w_inf)
    y_inf_pack = pack(y_inf)

    if save_to is not None:
        with open(save_to, 'w') as f:
            f.write("#define C" + label + "XY " + str(x.shape[2]) + "\n")
            f.write("#define C" + label + "Z " + str(x.shape[1]) + "\n")
            f.write("#define C" + label + "KXY " + str(list(conv_layer.parameters())[0].shape[2]) + "\n")
            f.write("#define C" + label + "KZ " + str(list(conv_layer.parameters())[0].shape[0]) + "\n")
            f.write("#define C" + label + "OXY " + str(y.shape[2]) + "\n")
            f.write("#define C" + label + "PD " + str(int(conv_layer.padding[0])) + "\n")
            f.write("#define C" + label + "PL " + str(int(pool_layer.kernel_size) if pool_layer is not None else 1) + "\n\n")

            f.write("#define weight_" + label + " {\\\n\t" + str(w_inf)[1:-1] + "\\\n};\n")
            f.write("#define weight_" + label + "_pack {\\\n\t" + str(w_inf_pack) + "\\\n};\n")
            f.write("#define thr_" + label + " {\\\n\t" + str(bn_th_inf)[1:-1] + "\\\n};\n")
            f.write("#define sign_" + label + " {\\\n\t" + str(bn_sign_inf)[1:-1] + "\\\n};\n")
            f.write("#define sign_" + label + "_pack {\\\n\t" + str(bn_sign_inf_pack) + "\\\n};\n")

            f.write("#define mu_" + label + " {\\\n\t" + str(mean_inf)[1:-1] + "\\\n};\n")
            f.write("#define sigma_" + label + " {\\\n\t" + str(sigma_inf)[1:-1] + "\\\n};\n")
            f.write("#define gamma_" + label + " {\\\n\t" + str(gamma_inf)[1:-1] + "\\\n};\n")
            f.write("#define beta_" + label + " {\\\n\t" + str(beta_inf)[1:-1] + "\\\n};\n")
            
            f.write("#define input_" + label + " {\\\n\t" + str(x_no_bin_inf)[1:-1] + "\\\n};\n")
            f.write("#define input_" + label + "_pack {\\\n\t" + str(x_inf_pack) + "\\\n};\n")
            f.write("#define output_" + label + " {\\\n\t" + str([float("{:.2f}".format(e)) for e in y_no_bin_inf])[1:-1] + "\\\n};\n")
            f.write("#define output_" + label + "_pack {\\\n\t" + str(y_inf_pack) + "\\\n};\n")

    if print_:
        print("x", x_inf_pack, x_inf)
        print("w", w_inf_pack, w_inf)
        print("thr", bn_th_inf)
        print("sign", bn_sign_inf_pack, bn_sign_inf)
        print("y", y_inf_pack, y_inf)

    return x_inf, w_inf, bn_th_inf, bn_sign_inf, bn_sign_inf_pack, y_inf

#inputs: layer_input, layer_input_clone, save_path, resnet10_layer_index_map
#outputs: layer_input_clone
def compile_identity_block(layer_input, block_layers, layers_list, resnet10_layer_index_map, save_path="/home/sravit/3pxnet/3pxnet-inference/examples/conv"):
    layer_input_clone = layer_input.clone()
    for layer in block_layers: #resnet10_conv_layers:
        conv_layer = layers_list[resnet10_layer_index_map["conv" + layer]]
        bn_layer = layers_list[resnet10_layer_index_map["bn" + layer]]
        pool_layer = layers_list[resnet10_layer_index_map["pool" + layer]] if ("pool" + layer) in list(resnet10_layer_index_map.keys()) else None
        compile_conv_block(conv_layer, bn_layer, layer_input, label=layer, save_to=save_path + layer + ".h", print_=False)
        layer_input = conv_layer(layer_input)
        layer_input = bn_layer(layer_input)
        if layer[-2:]=="_1":
            layer_input = torch.nn.functional.relu(layer_input)
    layer_input += layer_input_clone
    layer_input = torch.nn.functional.relu(layer_input)
    return layer_input

def compile_residual_block(layer_input, identity_layers, residual_layers, layers_list, resnet10_layer_index_map, save_path="/home/sravit/3pxnet/3pxnet-inference/examples/conv"):
    layer_input_clone = layer_input.clone()
    for layer in identity_layers: #resnet10_conv_layers:
        conv_layer = layers_list[resnet10_layer_index_map["conv" + layer]]
        bn_layer = layers_list[resnet10_layer_index_map["bn" + layer]]
        pool_layer = layers_list[resnet10_layer_index_map["pool" + layer]] if ("pool" + layer) in list(resnet10_layer_index_map.keys()) else None
        compile_conv_block(conv_layer, bn_layer, layer_input, pool_layer=pool_layer, label=layer, save_to=save_path + layer + ".h", print_=False)
        layer_input = conv_layer(layer_input)
        if pool_layer:
            layer_input = pool_layer(layer_input)
        layer_input = bn_layer(layer_input)
        if layer[-2:]=="_1":
            layer_input = torch.nn.functional.relu(layer_input)
    for layer in residual_layers:
        conv_layer = layers_list[resnet10_layer_index_map["conv" + layer]]
        bn_layer = layers_list[resnet10_layer_index_map["bn" + layer]]
        pool_layer = layers_list[resnet10_layer_index_map["pool" + layer]] if ("pool" + layer) in list(resnet10_layer_index_map.keys()) else None
        compile_conv_block(conv_layer, bn_layer, layer_input_clone, pool_layer=pool_layer, label=layer, save_to=save_path + layer + ".h", print_=False)
        layer_input_clone = conv_layer(layer_input_clone)
        if pool_layer:
            layer_input_clone = pool_layer(layer_input_clone)
        layer_input_clone = bn_layer(layer_input_clone)
    #print(compile_utils.convert_conv_act(layer_input_clone, binarize=False)[-20:])
    layer_input += layer_input_clone
    layer_input = torch.nn.functional.relu(layer_input)
    return layer_input

def compile_fc_block(fc_layer, bn_layer, x, print_=True, binarize_output=True):
    y = fc_layer.forward(x)
    y = bn_layer(y) if bn_layer is not None else y

    x_inf = convert_fc_act(x)
    w_inf = convert_fc_weight(fc_layer.weight)
    mu, sigma, gamma, beta = convert_bn_float(bn_layer.running_mean, bn_layer.running_var, bn_layer.weight, bn_layer.bias)
    bn_th_inf, bn_sign_inf = convert_bn(bn_layer.running_mean, bn_layer.running_var, bn_layer.weight, bn_layer.bias)
    bn_sign_inf_pack = pack(bn_sign_inf)
    y_inf = convert_fc_act(y, binarize=binarize_output)

    x_inf_pack = pack(x_inf)
    w_inf_pack = pack(w_inf)
    y_inf_pack = pack(y_inf)

    if print_:
        print("x", x_inf_pack, x_inf)
        print("w", w_inf_pack, w_inf)
        print("mu", mu)
        print("sigma", sigma)
        print("gamma", gamma)
        print("beta", beta)
        print("thr", bn_th_inf)
        print("sign", bn_sign_inf_pack, bn_sign_inf)
        print("y", y_inf_pack, y_inf)

    return mu, sigma, gamma, beta, bn_th_inf, bn_sign_inf, bn_sign_inf_pack, y_inf, y_inf_pack, x_inf, x_inf_pack, w_inf, w_inf_pack

def compile_bwn_fc(fc_layer, x, label, print_=True, save_to="/home/sravit/3pxnet/3pxnet-inference/examples/"+"fc.h"):
    y = fc_layer.forward(x)

    x_inf = convert_fc_act(x, binarize=False)
    w_inf = convert_fc_weight(fc_layer.weight)
    y_inf = convert_fc_act(y, binarize=False)

    w_inf_pack = pack(w_inf)

    if print_:
        print("x", x_inf)
        print("w", w_inf_pack, w_inf)
        print("y", y_inf)
    
    if save_to is not None:
        with open(save_to, 'w') as f:
            f.write("#define F" + label + "I " + str(x.shape[1]) + "\n")
            f.write("#define F" + label + "O " + str(y.shape[1]) + "\n")
            
            f.write("#define weight_" + label + " {\\\n\t" + str(w_inf)[1:-1] + "\\\n};\n")
            f.write("#define weight_" + label + "_pack {\\\n\t" + str(w_inf_pack) + "\\\n};\n")
            
            f.write("#define input_" + label + " {\\\n\t" + str(x_inf)[1:-1] + "\\\n};\n")
            f.write("#define output_" + label + " {\\\n\t" + str([float("{:.2f}".format(e)) for e in y_inf])[1:-1] + "\\\n};\n")
            
    return y