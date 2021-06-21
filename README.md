# simpledataset

Utility tools for simple vision image dataset format. WORK IN PROGRESS.

## Features
* See the summary of a dataset
* Convert from/to various dataset formats.
* Web UI to look into a dataset
* CUI tools to split and concat datasets.
* CUI tools to modify labels.

## Install
```
pip install simpledataset==0.1.0
```

## Usage
```
# Show summary
dataset_summary <input_dataset>

# For Classification dataset, extract only the images that have the specified labels.
# For Detection dataset, extract only the boxes that have the specified labels.
dataset_filter <input_dataset> <output_dataset> [--include_class <class_id> [<class_id> ...]] [--exclude_class <class_id> [<class_id> ...]]

# Update class labels
dataset_map <input_dataset> <output_dataset> --map <src_class_id> <dst_class_id> [--map <src_class_id> <dst_class_id> [--map...]]

dataset_split # NYI

dataset_concat # NYI

dataset_shuffle # NYI

dataset_sample # NYI

dataset_pack # NYI

# Convert from/to other dataset types. COCO format is supported now.
dataset_convert_from <input_dataset> <input_format> <output_filepath>
dataset_convert_to <input_dataset> <output_format> <output_filepath>
```


## Examples
### Change class ids
For example, if you would like to change MNIST to odd or even classification dataset, you can use dataset_map command. In this example, we use class_id=0 for even numbers, and class_id=1 for odd numbers.
```bash
dataset_map mnist.txt new_dataset.txt --map 2 0 --map 3 1 --map 4 0 --map 5 1 --map 6 0 --map 7 1 --map 8 0 --map 9 1
```


## SIMPLE Dataset format
Currently there are 2 dataset formats, Image Classification and Object Detection. Both datasets have a single txt file, image files and an optional list of label names (labels.txt). In addition to that, Object Detection datasets has label files that contains bbox info.

### Image Classification
The main txt format is:
```
<file> ::= <txt_line> ('\n' <txt_line>)*
<txt_line> ::= <image_filepath> ' ' <labels>
<image_filepath> ::= <filepath> | <zip_filepath> '@' <entry_name>
<labels> ::= <class_id> (',' <class_id>)*
```

Here is an example txt file.
```
train_images.zip@0.jpg 0
train_images2.zip@1.jpg 1
image.png 0,1
image2.bmp 0,1,2,3
```

### Object Detection
The main txt format is:
```
<file> ::= <txt_line> ('\n' <txt_line>)*
<txt_line> ::= <image_filepath> ' ' <label_filepath>
<image_filepath> ::= <filepath> | <zip_filepath> '@' <entry_name>
<label_filepath> ::= <filepath> | <zip_filepath> '@' <entry_name>
```

The format of a label file is:
```
<file> ::= <label_line> ('\n' <label_line>)*
<label_line> ::= <class_id> ' ' <bbox_x_min> ' ' <bbox_y_min> ' ' <bbox_x_max> ' ' <bbox_y_max>
<class_id> ::= <int>
<bbox_x_min> ::= <int>      ; 0 <= <bbox_x_min> < <bbox_x_max> <= <image_width>
<bbox_y_min> ::= <int>      ; 0 <= <bbox_y_min> < <bbox_y_max> <= <image_height>
<bbox_x_max> ::= <int>
<bbox_y_max> ::= <int>
```


## Usage for remote datasets
NYI.
This tool allows you to use datasets on Azure Blob Storage. You can update a dataset on the storage efficiently.
