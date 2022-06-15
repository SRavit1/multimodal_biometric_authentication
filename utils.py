import os
import random
import fire
import logging
import h5py
from tqdm import tqdm
import kaldi_io
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
                # spk, utt_id = utt[0:7], utt[-5:]
                # spk, utt_id = utt[0:7], utt[-9:]
                spk, utt_id = utt[0:7], utt[8:]
                h5_index_name = spk + '/' + utt_id
                store_h5[h5_index_name] = vec
                store_index.write(h5_index_name + '\n')
                pbar.update()


def kaldi_to_h5(scp_path, h5_path):
    stream = 'copy-vector scp:{} ark:- |'

    with open(scp_path, 'r') as f:
        lines = f.readlines()
        length = len(lines)

    h5_index = h5_path[:-2] + 'txt'
    with h5py.File(h5_path, 'w') as store_h5, open(h5_index, 'w') as store_index, tqdm(total=length) as pbar:
        for utt, vec in kaldi_io.read_vec_flt_ark(stream.format(scp_path)):
            spk, utt_id = utt[0:7], utt[8:]
            h5_index_name = spk + '/' + utt_id
            store_h5[h5_index_name] = vec
            store_index.write(h5_index_name + '\n')
            pbar.update()


def statistic(exp_dir, noise_type):
    store_dir = os.path.join(exp_dir, 'noise_statistic')
    if not os.path.exists(store_dir):
        os.mkdir(store_dir)

    read_stream = 'copy-vector scp:{} ark:- |'

    clean_scp = os.path.join(exp_dir, 'xvector_nnet_1a/xvectors_train/xvector_10w.scp')
    noise_scp = os.path.join(exp_dir, 'vox2_{}/xvector_nnet_1a/xvectors_train/xvector.scp'.format(noise_type))

    with open(noise_scp, 'r') as f:
        lines = f.readlines()
        length = len(lines)

    clean_generator = kaldi_io.read_vec_flt_ark(read_stream.format(clean_scp))
    noise_generator = kaldi_io.read_vec_flt_ark(read_stream.format(noise_scp))

    diff_list = []
    with tqdm(total=length) as pbar:
        for i in range(length):
            clean_utt, clean_vec = next(clean_generator)
            noise_utt, noise_vec = next(noise_generator)
            assert clean_utt == noise_utt
            diff_list.append(noise_vec - clean_vec)
            pbar.update()

    all_diff = np.stack(diff_list)
    mean = np.mean(all_diff, axis=0)
    std = np.std(all_diff, axis=0)

    np.save(os.path.join(store_dir, noise_type + '_mean.npy'), mean)
    np.save(os.path.join(store_dir, noise_type + '_std.npy'), std)


def main(storedir):
    face_scp = '/Users/chenzhengyang/PycharmProjects/Multi-Model/exp/train_data/face_vec.scp'
    utt_scp = '/Users/chenzhengyang/PycharmProjects/Multi-Model/exp/train_data/utt_vec.scp'
    utt_spk2utt_dic, face_spk2utt_dic = process_before_generate_scp(face_scp, utt_scp)
    generate_training_scp_classify(storedir, 1, face_spk2utt_dic, utt_spk2utt_dic, 60)


if __name__ == '__main__':
    fire.Fire(statistic)

