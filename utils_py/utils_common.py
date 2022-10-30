import os
import random
import fire
import logging
from tqdm import tqdm
import kaldi_io
import torch
import numpy as np

def spk2id(spk_file):
    spk2id_dic = {}
    with open(spk_file,'r') as f:
        lines = f.readlines()
        for i,line in enumerate(lines):
            spk2id_dic[line.strip()] = i
    return spk2id_dic


def generate_training_scp_contrastive(storedir, store_id, face_spk2utt_dic, utt_spk2utt_dic, batch_pair_num, num_batch):
    spk_list = face_spk2utt_dic.keys()

    face_train_scp = os.path.join(storedir,'face_train.' + str(store_id) + '.scp')
    utt_train_scp = os.path.join(storedir, 'utt_train.' + str(store_id) + '.scp')

    with open(face_train_scp,'w') as face_f, open(utt_train_scp,'w') as utt_f:
        for i in range(num_batch):
            sample_spks = random.sample(spk_list, batch_pair_num)
            for spk in sample_spks:
                face_f.write(random.sample(face_spk2utt_dic[spk], 1)[0] + '\n')
                utt_f.write(random.sample(utt_spk2utt_dic[spk] ,1)[0] + '\n')
            for spk in sample_spks: # do again to generate positive pair
                face_f.write(random.sample(face_spk2utt_dic[spk], 1)[0] + '\n')
                utt_f.write(random.sample(utt_spk2utt_dic[spk], 1)[0] + '\n')


def generate_training_scp_classify(storedir, store_id, face_spk2utt_dic, utt_spk2utt_dic, num_items):
    spk_list = face_spk2utt_dic.keys()

    face_train_scp = os.path.join(storedir,'face_train.' + str(store_id) + '.scp')
    utt_train_scp = os.path.join(storedir, 'utt_train.' + str(store_id) + '.scp')

    with open(face_train_scp,'w') as face_f, open(utt_train_scp,'w') as utt_f:
        for i in range(num_items):
            spk = random.sample(spk_list,1)[0]
            face_f.write(random.sample(face_spk2utt_dic[spk], 1)[0] + '\n')
            utt_f.write(random.sample(utt_spk2utt_dic[spk], 1)[0] + '\n')


def process_before_generate_scp(face_scp, utt_scp):
    with open(face_scp,'r') as f:
        face_list = f.readlines()

    with open(utt_scp,'r') as f:
        utt_list = f.readlines()

    face_spk2utt_dic = {}
    for line in face_list:
        line = line.strip()
        spk = line[:7]
        if spk not in face_spk2utt_dic:
            face_spk2utt_dic[spk] = []
            face_spk2utt_dic[spk].append(line)
        else:
            face_spk2utt_dic[spk].append(line)

    utt_spk2utt_dic = {}
    for line in utt_list:
        line = line.strip()
        spk = line[:7]
        if spk not in utt_spk2utt_dic:
            utt_spk2utt_dic[spk] = []
            utt_spk2utt_dic[spk].append(line)
        else:
            utt_spk2utt_dic[spk].append(line)

    return utt_spk2utt_dic, face_spk2utt_dic


# log to console or a file
def get_logger(
        name,
        format_str="%(asctime)s [%(pathname)s:%(lineno)s - %(levelname)s ] %(message)s",
        date_format="%Y-%m-%d %H:%M:%S",
        file=False):
    """
    Get python logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # file or console
    handler = logging.StreamHandler() if not file else logging.FileHandler(
        name)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt=format_str, datefmt=date_format)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


# log to concole and file at the same time
def get_logger_2(
        name,
        format_str="%(asctime)s [%(pathname)s:%(lineno)s - %(levelname)s ] %(message)s",
        date_format="%Y-%m-%d %H:%M:%S"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(name)
    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.INFO)

    # Create formatters and add it to handlers
    c_format = logging.Formatter(fmt=format_str, datefmt=date_format)
    f_format = logging.Formatter(fmt=format_str, datefmt=date_format)
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger

def check_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

"""
def store_to_h5(topdir, type='face', scp_num=100):
    stream = 'copy-vector scp:{} ark:- |'

    scps_dir = os.path.join(topdir,type + '_scp')
    h5_store_dir = os.path.join(topdir,type + '_h5')

    for i in range(scp_num):
        scp_idx = str(i).zfill(3)
        scp_path = os.path.join(scps_dir, 'scp_' + scp_idx)

        h5_path = os.path.join(h5_store_dir,'data_' + scp_idx + '.h5')
        h5_index = os.path.join(h5_store_dir,'data_' + scp_idx + '.index')

        with h5py.File(h5_path, 'w') as store_h5, open(h5_index,'w') as store_index, tqdm(total=10913) as pbar:

            for utt, vec in kaldi_io.read_vec_flt_ark(stream.format(scp_path)):
                spk, utt_id = utt[0:7], utt[-5:]
                h5_index_name = spk + '/' + utt_id
                store_h5[h5_index_name] = vec
                store_index.write(h5_index_name + '\n')
                pbar.update()
"""


def save_state_dict(state_dict, checkpoint_format, epoch, iters, is_best=False):
    filepath = checkpoint_format.format('Epoch-{}-Iter-{}'.format(epoch,iters))
    torch.save(state_dict, filepath)

    if is_best:
        filepath = checkpoint_format.format('Best')
        torch.save(state_dict, filepath)


def write_conf(read_path, store_path):
    '''
    store the config file in exp dir
    :param read_path: the conf path to read
    :param store_path: the conf path to write
    '''
    with open(read_path,'r') as f_r, open(store_path,'w') as f_w:
        lines = f_r.readlines()
        f_w.writelines(lines)


def normalize_np_vec(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def add_noise_to_embedding(emb, prob):
    if np.random.random() > prob:
        return emb
    else:
        choice = np.random.randint(0, 3)
        if choice == 0:  # set to zero embedding
            return np.zeros(emb.shape[0])
        elif choice == 1:  # set to random noise embedding
            return np.random.randn(emb.shape[0])
        else:  # add random noise
            std = np.std(emb)
            emb += 0.5 * std * np.random.randn(emb.shape[0])
            return emb


def add_noise_to_embedding_v2(emb, noise_mean_list, noise_std_list, prob):
    if np.random.random() > prob:
        return emb
    else:
        length = emb.shape[0]
        choice = np.random.randint(0, 4)
        noise = np.random.randn(length) * noise_std_list[choice] + noise_mean_list[choice]
        # if choice == 3:  # change here
        #     return np.random.randn(length)
        return emb + noise


def vox1_zero_random(read_scp, store_ark, random=True, emb_dim=512):
    '''
    Write all zeros embeddings or random embeddings as vox1 extracted embedding
    '''
    ark_path = store_ark
    scp_path = ark_path[:-3] + 'scp'
    ark_scp_output = 'ark:| copy-vector ark:- ark,scp:{},{}'.format(ark_path, scp_path)

    data = np.zeros(emb_dim)
    with open(read_scp,'r') as f, kaldi_io.open_or_fd(ark_scp_output, 'wb') as f_w:
        lines = f.readlines()
        for line in tqdm(lines):
            if random:
                data = np.random.randn(emb_dim)
            utt = line.strip().split()[0]
            kaldi_io.write_vec_flt(f_w, data, utt)


def generate_polluted_scp(scp, store_scp, prob=0.6, noise_num=3):
    '''
    The output of every line is: "utt label_face label_utt"
    label_face, label_uut are in [0, 1, 2, 3]
    0 means the original emb, 1 set emb to zero, 2 set emb to random noise
    3 add emb random noise
    '''
    with open(scp, 'r') as r_f, open(store_scp, 'w') as w_f:
        lines = r_f.readlines()
        for line in tqdm(lines):
            utt = line.strip().split()[0]

            label_utt = '0'
            label_face = '0'

            if np.random.random() > prob:
                line = ' '.join([utt, label_face, label_utt])
                w_f.write(line + '\n')
            else:
                modality_choice = np.random.randint(0, 2)
                if modality_choice == 0:
                    label_face = str(np.random.randint(1, 1+noise_num))
                else:
                    label_utt = str(np.random.randint(1, 1+noise_num))
                line = ' '.join([utt, label_face, label_utt])
                w_f.write(line + '\n')


def pollute_emb(read_scp, polluted_scp, store_ark, emb_type='face'):
    '''
    Using polluted_scp generated above to store new ark
    '''
    read_stream = 'copy-vector scp:{} ark:- |'

    # write config
    ark_path = store_ark
    scp_path = ark_path[:-3] + 'scp'
    ark_scp_output = 'ark:| copy-vector ark:- ark,scp:{},{}'.format(ark_path, scp_path)

    with open(polluted_scp, 'r') as f:
        lines = f.readlines()

    with kaldi_io.open_or_fd(ark_scp_output, 'wb') as f_w, tqdm(total=len(lines)) as pbar:
        for index, (utt, data) in enumerate(kaldi_io.read_vec_flt_ark(read_stream.format(read_scp))):
            seg = lines[index].strip().split()
            if emb_type == 'face':
                label = int(seg[1])
            else:
                label = int(seg[2])

            if label == 1:
                data = np.zeros(data.shape[0])
            elif label == 2:
                data = np.random.randn(data.shape[0])
            elif label == 3:
                std = np.std(data)
                data = data + 0.5 * std * np.random.randn(data.shape[0])

            kaldi_io.write_vec_flt(f_w, data, utt)
            pbar.update()


def pollute_emb_v2(scp0, scp1, scp2, scp3, scp4, polluted_scp, new_scp, emb_type='face'):
    '''
    :param scp0: original embedding
    :param scp1: embedding with type 1 noise
    :param scp2: embedding with type 2 noise
    :param scp3: embedding with type 3 noise
    :param scp2: zero embedding
    '''
    scp_dict = {}
    with open(scp0, 'r') as f:
        scp_dict['0'] = f.readlines()
    with open(scp1, 'r') as f:
        scp_dict['1'] = f.readlines()
    with open(scp2, 'r') as f:
        scp_dict['2'] = f.readlines()
    with open(scp3, 'r') as f:
        scp_dict['3'] = f.readlines()
    with open(scp4, 'r') as f:
        scp_dict['4'] = f.readlines()

    with open(polluted_scp, 'r') as f:
        polluted_scp_lines = f.readlines()
    with open(new_scp, 'w') as f_w:
        for i, line in enumerate(polluted_scp_lines):
            seg = line.strip().split()
            if emb_type == 'face':
                label = seg[1]
            else:
                label = seg[2]
            write_line = scp_dict[label][i].strip()
            f_w.write(write_line + '\n')


def pollute_fbank_v2(scp0, scp1, scp2, scp3, scp4, polluted_scp, new_scp, emb_type='utt'):
    '''
    :param scp0: original embedding
    :param scp1: embedding with type 1 noise
    :param scp2: embedding with type 2 noise
    :param scp3: embedding with type 3 noise
    :param scp2: "zero embedding", here I also use the original fbank, in extract.py I will use random noise to replace this
    '''
    scp_dict = {}
    with open(scp0, 'r') as f:
        scp_dict['0'] = f.readlines()
    with open(scp1, 'r') as f:
        scp_dict['1'] = f.readlines()
    with open(scp2, 'r') as f:
        scp_dict['2'] = f.readlines()
    with open(scp3, 'r') as f:
        scp_dict['3'] = f.readlines()
    with open(scp4, 'r') as f:
        scp_dict['4'] = f.readlines()

    scp_dict['4'] = scp_dict['0']

    with open(polluted_scp, 'r') as f:
        polluted_scp_lines = f.readlines()
    with open(new_scp, 'w') as f_w:
        for i, line in enumerate(polluted_scp_lines):
            seg = line.strip().split()
            if emb_type == 'face':
                label = seg[1]
            else:
                label = seg[2]
            write_line = scp_dict[label][i].strip()
            f_w.write(write_line + '\n')


def load_mean_vec(filepath):
    # transform kaldi mean.vec to numpy array
    with open(filepath,'r') as f:
        line = f.readline()
    line = line.strip().strip('[').strip(']').strip()
    line = ','.join(line.split())
    line = '[' + line + ']'
    line = eval(line)
    mean_vec = np.array(line)
    return mean_vec


def param2args(args, params_dict):
    exec_format = 'args.{} = {}'
    exec_format_str = 'args.{} = "{}"'
    for key, value in params_dict.items():
        if isinstance(value, str):
            exec(exec_format_str.format(key, value))
        else:
            exec(exec_format.format(key, value))


def concat_kaldi_vec(scp1, scp2, store_ark, mean1=None, mean2=None, norm=True):
    read_stream = 'copy-vector scp:{} ark:- |'
    write_steam = 'ark:| copy-vector ark:- ark,scp:{},{}'
    store_scp = store_ark[:-3] + 'scp'
    ark_scp_output = write_steam.format(store_ark, store_scp)

    if mean1 is not None:
        mean1 = load_mean_vec(mean1)
        mean2 = load_mean_vec(mean2)

    with open(scp1, 'r') as f:
        lines = f.readlines()
        length = len(lines)

    read_generator1 = kaldi_io.read_vec_flt_ark(read_stream.format(scp1))
    read_generator2 = kaldi_io.read_vec_flt_ark(read_stream.format(scp2))
    with kaldi_io.open_or_fd(ark_scp_output, 'wb') as f_w, tqdm(total=length) as pbar:
        for i in range(length):
            utt_id1, vec1 = next(read_generator1)
            utt_id2, vec2 = next(read_generator2)
            assert (utt_id1 == utt_id2)

            if mean1 is not None:
                vec1 = vec1 - mean1
                vec2 = vec2 - mean2

            if norm:
                vec1 = normalize_np_vec(vec1)
                vec2 = normalize_np_vec(vec2)
            store_vec = np.concatenate((vec1, vec2))

            kaldi_io.write_vec_flt(f_w, store_vec, utt_id1)
            pbar.update()






def main(storedir):
    face_scp = '/Users/chenzhengyang/PycharmProjects/Multi-Model/exp/train_data/face_vec.scp'
    utt_scp = '/Users/chenzhengyang/PycharmProjects/Multi-Model/exp/train_data/utt_vec.scp'
    utt_spk2utt_dic, face_spk2utt_dic = process_before_generate_scp(face_scp, utt_scp)
    generate_training_scp_classify(storedir, 1, face_spk2utt_dic, utt_spk2utt_dic, 60)


if __name__ == '__main__':
    #fire.Fire(store_to_h5)
    #fire.Fire(vox1_zero_random)
    #fire.Fire(generate_polluted_scp)
    fire.Fire(pollute_emb_v2)
    #fire.Fire(pollute_emb)
    # fire.Fire(concat_kaldi_vec)

