import kaldi_io
import argparse
import configparser
from tqdm import tqdm
import os
import torch
from models.fusion import System
from utils_py.utils_common import load_mean_vec


def write_ark(model, args):
    model.eval()

    # read config
    read_stream = 'copy-vector scp:{} ark:- |'

    # write config
    ark_path = args.ark_path
    scp_path = ark_path[:-3] + 'scp'
    ark_scp_output = 'ark:| copy-vector ark:- ark,scp:{},{}'.format(ark_path, scp_path)

    if args.face_mean_vec is not None:
        face_mean_vec = load_mean_vec(args.face_mean_vec)
        utt_mean_vec = load_mean_vec(args.utt_mean_vec)
    else:
        face_mean_vec = None
        utt_mean_vec = None

    with open(args.utt_scp, 'r') as f:
        lines = f.readlines()
        length = len(lines)

    face_generator = kaldi_io.read_vec_flt_ark(read_stream.format(args.face_scp))
    utt_generator = kaldi_io.read_vec_flt_ark(read_stream.format(args.utt_scp))
    with torch.no_grad(), kaldi_io.open_or_fd(ark_scp_output, 'wb') as f_w, \
            tqdm(total=length) as pbar:
        for i in range(length):
            utt_face, face_emb = next(face_generator)
            utt_utt, utt_emb = next(utt_generator)
            assert (utt_face == utt_utt)

            if face_mean_vec is not None:
                face_emb = face_emb - face_mean_vec
                utt_emb = utt_emb - utt_mean_vec

            input_face = torch.from_numpy(face_emb).float().unsqueeze(0)
            input_utt = torch.from_numpy(utt_emb).float().unsqueeze(0)

            if torch.cuda.is_available() and not args.cpu:
                input_face = input_face.cuda()
                input_utt = input_utt.cuda()

            out, _, _ = model(input_face, input_utt)
            out = out.squeeze().detach().data.cpu().numpy()
            kaldi_io.write_vec_flt(f_w, out, utt_face)

            pbar.update()
    print('Success!!!')


def get_attention(model, args):
    model.eval()

    # read config
    read_stream = 'copy-vector scp:{} ark:- |'

    with open(args.utt_scp, 'r') as f:
        lines = f.readlines()
        length = len(lines)

    if args.face_mean_vec is not None:
        face_mean_vec = load_mean_vec(args.face_mean_vec)
        utt_mean_vec = load_mean_vec(args.utt_mean_vec)
    else:
        face_mean_vec = None
        utt_mean_vec = None

    face_generator = kaldi_io.read_vec_flt_ark(read_stream.format(args.face_scp))
    utt_generator = kaldi_io.read_vec_flt_ark(read_stream.format(args.utt_scp))
    with torch.no_grad(), open(args.att_path, 'w') as f_w, tqdm(total=length) as pbar:
        for i in range(length):
            utt_face, face_emb = next(face_generator)
            utt_utt, utt_emb = next(utt_generator)
            assert (utt_face == utt_utt)

            if face_mean_vec is not None:
                face_emb = face_emb - face_mean_vec
                utt_emb = utt_emb - utt_mean_vec

            input_face = torch.from_numpy(face_emb).float().unsqueeze(0)
            input_utt = torch.from_numpy(utt_emb).float().unsqueeze(0)

            if torch.cuda.is_available() and not args.cpu:
                input_face = input_face.cuda()
                input_utt = input_utt.cuda()

            attention = model.get_attention(input_face, input_utt)
            face_att = attention[0][0].item()
            utt_att = attention[0][1].item()

            write_line = ' '.join([utt_utt, str(face_att), str(utt_att)])
            f_w.write(write_line + '\n')

            pbar.update()
    print('Success!!!')


def get_face_emb(model, args):
    model.eval()

    # read config
    read_stream = 'copy-vector scp:{} ark:- |'

    # write config
    ark_path = args.ark_path
    scp_path = ark_path[:-3] + 'scp'
    ark_scp_output = 'ark:| copy-vector ark:- ark,scp:{},{}'.format(ark_path, scp_path)

    if args.face_mean_vec is not None:
        face_mean_vec = load_mean_vec(args.face_mean_vec)
    else:
        face_mean_vec = None

    with open(args.face_scp, 'r') as f:
        lines = f.readlines()
        length = len(lines)

    face_generator = kaldi_io.read_vec_flt_ark(read_stream.format(args.face_scp))
    with torch.no_grad(), kaldi_io.open_or_fd(ark_scp_output, 'wb') as f_w, \
            tqdm(total=length) as pbar:
        for i in range(length):
            utt_face, face_emb = next(face_generator)
            if face_mean_vec is not None:
                face_emb = face_emb - face_mean_vec
            input_face = torch.from_numpy(face_emb).float().unsqueeze(0)

            if torch.cuda.is_available() and not args.cpu:
                input_face = input_face.cuda()

            out = model.get_face_emb(input_face).squeeze().detach().data.cpu().numpy()
            kaldi_io.write_vec_flt(f_w, out, utt_face)

            pbar.update()
    print('Success!!!')


def get_utt_emb(model, args):
    model.eval()

    # read config
    read_stream = 'copy-vector scp:{} ark:- |'

    # write config
    ark_path = args.ark_path
    scp_path = ark_path[:-3] + 'scp'
    ark_scp_output = 'ark:| copy-vector ark:- ark,scp:{},{}'.format(ark_path, scp_path)

    with open(args.utt_scp, 'r') as f:
        lines = f.readlines()
        length = len(lines)

    if args.utt_mean_vec is not None:
        utt_mean_vec = load_mean_vec(args.utt_mean_vec)
    else:
        utt_mean_vec = None

    utt_generator = kaldi_io.read_vec_flt_ark(read_stream.format(args.utt_scp))
    with torch.no_grad(), kaldi_io.open_or_fd(ark_scp_output, 'wb') as f_w, \
            tqdm(total=length) as pbar:
        for i in range(length):
            utt_utt, utt_emb = next(utt_generator)
            if utt_mean_vec is not None:
                utt_emb = utt_emb - utt_mean_vec
            input_utt = torch.from_numpy(utt_emb).float().unsqueeze(0)

            if torch.cuda.is_available() and not args.cpu:
                input_utt = input_utt.cuda()

            out = model.get_utt_emb(input_utt).squeeze().detach().data.cpu().numpy()
            kaldi_io.write_vec_flt(f_w, out, utt_utt)

            pbar.update()
    print('Success!!!')


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--expdir', type=str, default=None,
                        help='the config file')
    parser.add_argument("--model-init", type=str, default='Nothing', help='The model path')
    parser.add_argument("--face-scp", type=str, default=None, help='face embedding scp path')
    parser.add_argument("--utt-scp", type=str, default=None, help='utt embedding scp path')
    parser.add_argument("--ark-path", type=str, default=None, help='the path to store ark')
    parser.add_argument("--cpu", action='store_true', default=False,
                        help='use cpu not gpu')
    parser.add_argument("--best", type=int, default=0, help='whether to use best model')
    parser.add_argument("--attention", action='store_true', default=False,
                        help='get the attention of face and utt')
    parser.add_argument("--att-path", type=str, default=None, help='the path to store attention')
    parser.add_argument("--only-face", type=int, default=0, help='extract using only face')
    parser.add_argument("--only-utt", type=int, default=0, help='extract using only utt')

    args = parser.parse_args()
    # *********************** process config ***********************
    config_path = os.path.join(args.expdir, 'train.conf')
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.optionxform = str
    config.read(config_path)

    if 'fusion_model' in config:
        model_section = config['fusion_model']
    else:
        model_section = config['model']

    data_section = config['data']
    optim_section = config['optimization']

    model_params = {k: eval(v) for k, v in model_section.items()}
    data_params = {k: eval(v) for k, v in data_section.items()}
    optim_params = {k: eval(v) for k, v in optim_section.items()}

    # init the model
    model = System(**model_params)
    if os.path.exists(args.model_init):
        state_dict = torch.load(args.model_init, map_location=lambda storage, loc: storage)
    elif args.best == 1:
        model_path = os.path.join(args.expdir, 'models', 'epoch-Best.th')
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    else:
        model_path = None
        for i in range(optim_params['max_epoch'] - 1, -1, -1):
            model_path = os.path.join(args.expdir, 'models', 'epoch-{}.th'.format(str(i)))
            if os.path.exists(model_path):
                break
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)

    if 'fusion_model' in state_dict:
        model.load_state_dict(state_dict['fusion_model'])
    elif 'model' in list(state_dict.keys()):
        model.load_state_dict(state_dict['model'])
    else:
        model.load_state_dict(state_dict)

    if torch.cuda.is_available() and not args.cpu:
        model = model.cuda()

    if 'face_mean_vec' in data_params:
        args.face_mean_vec = data_params['face_mean_vec']
        args.utt_mean_vec = data_params['utt_mean_vec']
    else:
        args.face_mean_vec = None
        args.utt_mean_vec = None

    if args.only_face == 1:
        get_face_emb(model, args)
    elif args.only_utt == 1:
        get_utt_emb(model, args)
    elif args.attention:
        get_attention(model, args)
    else:
        write_ark(model, args)


if __name__ == '__main__':
    main()
