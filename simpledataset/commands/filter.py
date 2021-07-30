import argparse
import logging
import pathlib
from simpledataset.common import SimpleDatasetFactory, ImageClassificationDataset, ObjectDetectionDataset, VisualRelationshipDataset, DatasetWriter

logger = logging.getLogger(__name__)


def filter_dataset(main_txt_filepath, output_filepath, include_class_ids, exclude_class_ids):
    dataset = SimpleDatasetFactory().load(main_txt_filepath.read_text(), main_txt_filepath.parent)
    max_class_id = dataset.get_max_class_id()
    include_class_ids = set(include_class_ids or [i for i in range(max_class_id + 1) if i not in exclude_class_ids])

    for c in include_class_ids:
        if c > max_class_id:
            logger.warning(f"The class {c} is not in the dataset.")

    if dataset.type == 'image_classification':
        data = [(image, [i for i in labels if i in include_class_ids]) for image, labels in dataset]
        # Remove images that have no labels.
        data = [d for d in data if d[1]]
    elif dataset.type == 'object_detection':
        data = [(image, [x for x in labels if x[0] in include_class_ids]) for image, labels in dataset]
        # Remove images that have no labels
        data = [d for d in data if d[1]]
    elif dataset.type == 'visual_relationship':
        data = [(image, [x for x in labels if x[10] in include_class_ids]) for image, labels in dataset]
    else:
        raise RuntimeError

    DATASET_CLASSES = {'image_classification': ImageClassificationDataset,
                       'object_detection': ObjectDetectionDataset,
                       'visual_relationship': VisualRelationshipDataset}
    dataset = DATASET_CLASSES[dataset.type](data, main_txt_filepath.parent, label_names=dataset.labels)

    copy_images = main_txt_filepath != output_filepath.parent
    DatasetWriter().write(dataset, output_filepath, copy_images=copy_images)
    print(f"Successfully saved {output_filepath}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('main_txt_filepath', type=pathlib.Path)
    parser.add_argument('output_filepath', type=pathlib.Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--include_class', nargs='*', default=[], metavar='CLASS_ID')
    group.add_argument('--exclude_class', nargs='*', default=[], metavar='CLASS_ID')
    args = parser.parse_args()

    if args.output_filepath.exists():
        parser.error(f"{args.output_filepath} already exists.")

    if (args.include_class and args.exclude_class) or not (args.include_class or args.exclude_class):
        parser.error("--include_class or --exclude_class must be specified.")

    include_class_ids = [int(c) for c in args.include_class]
    exclude_class_ids = [int(c) for c in args.exclude_class]
    filter_dataset(args.main_txt_filepath, args.output_filepath, include_class_ids, exclude_class_ids)


if __name__ == '__main__':
    main()
