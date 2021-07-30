import argparse
import pathlib
from simpledataset.common import SimpleDatasetFactory, ImageClassificationDataset, ObjectDetectionDataset, VisualRelationshipDataset, DatasetWriter


def pack(main_txt_filepath, output_filepath, images_directory, keep_empty_images):
    dataset = SimpleDatasetFactory().load(main_txt_filepath.read_text(), main_txt_filepath.parent, images_directory=images_directory)

    data = [(image, labels) for image, labels in dataset if labels or keep_empty_images]

    DATASET_CLASSES = {'image_classification': ImageClassificationDataset,
                       'object_detection': ObjectDetectionDataset,
                       'visual_relationship': VisualRelationshipDataset}
    dataset = DATASET_CLASSES[dataset.type](data, main_txt_filepath.parent, label_names=dataset.labels, images_directory=images_directory)

    DatasetWriter().write(dataset, output_filepath, copy_images=True)
    print(f"Successfully saved {output_filepath}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('main_txt_filepath', type=pathlib.Path)
    parser.add_argument('output_filepath', type=pathlib.Path)
    parser.add_argument('--images_dir', type=pathlib.Path, help="Directory from which load images.")
    parser.add_argument('--keep_empty_images', action='store_true', help="Keep images that don't have annotations.")

    args = parser.parse_args()
    images_dir = args.images_dir or args.main_txt_filepath.parent
    pack(args.main_txt_filepath, args.output_filepath, images_dir, args.keep_empty_images)


if __name__ == '__main__':
    main()
