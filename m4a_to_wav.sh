#!/bin/bash
cd ~/datasets/VoxCeleb-multimodal/VoxCeleb2/dev
mkdir utt

print_freq=1000

total=$(find utt-m4a/ -type f | wc -l)
i=1
for m4a_file in $(find utt-m4a/ -type f)
do
    wav_file=${m4a_file%.m4a}.wav
    wav_file=${wav_file/utt-m4a/utt}
    mkdir -p $(dirname $wav_file)
    #echo $m4a_file $wav_file
    ffmpeg -y -loglevel quiet -i $m4a_file $wav_file
    
    if [ 0 == $(expr $i % $print_freq) ]
    then
        echo Finished converting $i/$total files.
    fi
    i=$((i+1))
done