size=100000
print_freq=100
#train_dir="/home/sravit/datasets/VoxCeleb-multimodal/VoxCeleb1/dev"
train_dir="/home/sravit/datasets/VoxCeleb-multimodal/VoxCeleb2/dev"
small_face_dir=$train_dir/face-small
list_file="../../face_small.txt"

#rm -rf $small_face_dir
#rm $train_dir/../face_small.txt

echo "Number of images in large face directory:" $(find $train_dir/face -type f | wc -l)
echo "Making smaller face directory with" $size "images."

mkdir $small_face_dir
cd $train_dir/face
find -type f | shuf | head -n $size > $list_file

index=0
while read line;
do
cp --parents $line $small_face_dir
index=$((index + 1))
if [ $(expr $index % $print_freq) = 0 ]
then
echo "Finished copying image #" $index "/" $size
fi
done < $list_file

small_dataset_size=$(find $small_face_dir -type f | wc -l)
if [ $small_dataset_size = $size ]
then
echo "Successfully created dataset of size" $small_dataset_size
exit 0
fi
echo "Error: Created dataset of size" $small_dataset_size "instead of" $size
exit 1
