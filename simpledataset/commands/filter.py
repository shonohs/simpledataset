import argparse
import logging
import pathlib
from simpledataset.common import SimpleDatasetFactory, ImageClassificationDataset, ObjectDetectionDataset, DatasetWriter

logger = logging.getLogger(__name__)


def filter_dataset(main_txt, directory, output_filepath, include_class_ids, exclude_class_ids):
    dataset = SimpleDatasetFactory().load(main_txt, directory)
    max_class_id = dataset.get_max_class_id()
    include_class_ids = set(include_class_ids or [i for i in range(max_class_id + 1) if i not in exclude_class_ids])

    for c in include_class_ids:
        if c > max_class_id:
            logger.warning(f"The class {c} is not in the dataset.")

    if dataset.type == 'image_classification':
        data = [(image, [i for i in labels if i in include_class_ids]) for image, labels in dataset]
        # Remove images that have no labels.
        data = [d for d in data if d[1]]
        dataset = ImageClassificationDataset(data, directory)
    elif dataset.type == 'object_detection':
        data = [(image, [x for x in labels if x[0] in include_class_ids]) for image, labels in dataset]
        # Remove images that have no labels
        data = [d for d in data if d[1]]
        dataset = ObjectDetectionDataset(data, directory)
    else:
        raise RuntimeError

    DatasetWriter(directory).write(dataset, output_filepath)
    print(f"Successfully saved {output_filepath}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('main_txt_filepath', type=pathlib.Path)
    parser.add_argument('output_filename')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--include_class', nargs='*', default=[], metavar='CLASS_ID')
    group.add_argument('--exclude_class', nargs='*', default=[], metavar='CLASS_ID')
    args = parser.parse_args()

    if '/' in args.output_filename:
        parser.error("The output file must be in the same directory.")

    main_txt = args.main_txt_filepath.read_text()
    directory = args.main_txt_filepath.parent

    if (directory / args.output_filename).exists():
        parser.error(f"{args.output_filename} already exists.")

    if (args.include_class and args.exclude_class) or not (args.include_class or args.exclude_class):
        parser.error("--include_class or --exclude_class must be specified.")

    include_class_ids = [int(c) for c in args.include_class]
    exclude_class_ids = [int(c) for c in args.exclude_class]
    filter_dataset(main_txt, directory, args.output_filename, include_class_ids, exclude_class_ids)


if __name__ == '__main__':
    main()
