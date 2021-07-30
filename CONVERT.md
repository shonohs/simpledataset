# Dataset conversion

## OpenImages Boxes to SIMPLE dataset.

Here is an example to download the dataset and convert it to SIMPLE dataset format.

```zsh
# Download training dataset images.
# See https://storage.googleapis.com/openimages/web/download.html for details.
mkdir images && cd images/
for i in 0 1 2 3 4 5 6 7 8 9 a b c d e f; do aws s3 --no-sign-request cp s3://openimages-dataset/tar/train_$i.tar.gz . && tar zxvf train_$i.tar.gz && rm train_$i.tar.gz; done
cd ../

# Download bounding box annotations.
wget https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv

# Convert to a SIMPLE dataset: converted.txt
# In this example --skip-images flag is used because the dataset is too large.
dataset_convert_from openimages_od oidv6-train-annotations-bbox.csv images/ converted.txt --skip-images

# Check the summary
dataset_summary converted.txt
```

## OpenImages Visual Relationships to SIMPLE Visual Relationship dataset.

```zsh
# Download training dataset images.
# See https://storage.googleapis.com/openimages/web/download.html for details.
mkdir images && cd images/
for i in 0 1 2 3 4 5 6 7 8 9 a b c d e f; do aws s3 --no-sign-request cp s3://openimages-dataset/tar/train_$i.tar.gz . && tar zxvf train_$i.tar.gz && rm train_$i.tar.gz; done
cd ../

# Download bounding box annotations.
wget https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-vrd.csv

# Convert to a SIMPLE dataset: converted.txt
dataset_convert_from openimages_od oidv6-train-annotations-vrd.csv images/ converted.txt

# Check the summary
dataset_summary converted.txt
```

## COCO Object Detection to SIMPLE Object Detection dataset

```zsh
# Download COCO dataset from https://cocodataset.org/#download
unzip annotations_trainval2017.zip
unzip val2017.zip

# Convert to SIMPLE format.
# dataset_convert_from coco <coco's json filepath> <coco's image directory> <output filename>
dataset_convert_from coco annotations/instances_val2017.json val2017/ converted.txt

# Check the summary
dataset_summary converted.txt
```

## HICO-DET to SIMPLE Visual Relationship dataset

```zsh
tar zxvf hico_20160224_det.tar.gz
cd hico_20160224_det

# Convert to SIMPLE format.
# dataset_convert_from hicodet <annotation filepath> <image directory> <bbox_train|bbox_test> <output filepath>
# Invisible relationship will be ignored.
dataset_convert_from hicodet anno_box.mat images/train2015 bbox_train train/images.txt
dataset_convert_from hicodet anno_box.mat images/test2015 bbox_test test/images.txt

# Check the summary
dataset_summary train/images.txt
dataset_summary test/images.txt
```
