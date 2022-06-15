#!/bin/bash

# some default settings
exp_dir=''
face_scp=''
utt_scp=''
model_init='Nothing'
vox=1
mode=0  # mode 1 to get attention
only_face=0
only_utt=0
use_best=0  # whether to use the best model

nj=10               # number of subjobs to extract the embeddings
usegpu=0

. utils/parse_options.sh

PYTHON=/mnt/lustre/sjtu/home/czy97/.conda/envs/pytorch1_6/bin/python
tmpdir=tmpdata  # tmpdir to store splits


ark_dir_vox1=$exp_dir/xvector_nnet_1a/xvectors_voxceleb1
ark_dir_vox2=$exp_dir/xvector_nnet_1a/xvectors_train
log_dir1=$exp_dir/xvector_nnet_1a/xvectors_voxceleb1/extract_log
log_dir2=$exp_dir/xvector_nnet_1a/xvectors_train/extract_log
if [[ ! -d $log_dir1 ]];then
    mkdir -p $log_dir1
fi
if [[ ! -d $log_dir2 ]];then
    mkdir -p $log_dir2
fi

if [[ $vox == 1 ]];then
  tmpdir=$log_dir1
else
  tmpdir=$log_dir2
fi


if [[ $mode == 0 ]];then
    # split the input scp to some sub_scps
    file_len=`wc -l $face_scp | awk '{print $1}'`
    subfile_len=$[$file_len / $nj + 1]
    #prefix=`uuidgen`
    prefix='split'
    split -l $subfile_len -d -a 3 $face_scp $tmpdir/${prefix}_face_
    split -l $subfile_len -d -a 3 $utt_scp $tmpdir/${prefix}_utt_

    # extract embeddings
    if [[ $vox == 1 ]];then
      ark_path=$ark_dir_vox1/xvector.ark
    else
      ark_path=$ark_dir_vox2/xvector.ark
    fi

    ark_path_prefix=${ark_path%.*}
    for suffix in `seq 0 $[$nj-1]`;do
        suffix=`printf '%03d' $suffix`
        face_subfile=$tmpdir/${prefix}_face_${suffix}
        utt_subfile=$tmpdir/${prefix}_utt_${suffix}
        ark_path_subfile=${ark_path_prefix}.${suffix}.ark
        if [[ $usegpu == 1 ]];then
            srun -p gpu -n 1 -c 2 --gres=gpu:1 $PYTHON extract_fusion.py --expdir $exp_dir \
                                                        --face-scp $face_subfile \
                                                        --utt-scp $utt_subfile \
                                                        --ark-path $ark_path_subfile \
                                                        --model-init $model_init \
                                                        --only-face $only_face \
                                                        --only-utt $only_utt \
                                                        --best $use_best \
                                                        > ${tmpdir}/${prefix}.${suffix}.log 2>&1 &
        else
            srun -p cpu -n 1 $PYTHON extract_fusion.py --expdir $exp_dir \
                                                        --face-scp $face_subfile \
                                                        --utt-scp $utt_subfile \
                                                        --ark-path $ark_path_subfile \
                                                        --model-init $model_init \
                                                        --only-face $only_face \
                                                        --only-utt $only_utt \
                                                        --best $use_best \
                                                        --cpu \
                                                        > ${tmpdir}/${prefix}.${suffix}.log 2>&1 &

        fi
    done
else
    # split the input scp to some sub_scps
    file_len=`wc -l $face_scp | awk '{print $1}'`
    subfile_len=$[$file_len / $nj + 1]
    #prefix=`uuidgen`
    prefix='split'
    #split -l $subfile_len -d -a 3 $face_scp $tmpdir/${prefix}_face_
    #split -l $subfile_len -d -a 3 $utt_scp $tmpdir/${prefix}_utt_

    att_path=$exp_dir/xvector_nnet_1a/attention
    for suffix in `seq 0 $[$nj-1]`;do
        suffix=`printf '%03d' $suffix`
        face_subfile=$tmpdir/${prefix}_face_${suffix}
        utt_subfile=$tmpdir/${prefix}_utt_${suffix}
        att_path_subfile=$tmpdir/${prefix}_att_${suffix}

        srun -p cpu -n 1 -x cqxx-01-002 $PYTHON extract_fusion.py --expdir $exp_dir \
                                                --face-scp $face_subfile \
                                                --utt-scp $utt_subfile \
                                                --attention \
                                                --model-init $model_init \
                                                --att-path $att_path_subfile \
                                                --best $use_best \
                                                --cpu \
                                                > ${tmpdir}/${prefix}.attention.log 2>&1 &

    done

    wait
    cat $tmpdir/${prefix}_att_* > $att_path
fi


