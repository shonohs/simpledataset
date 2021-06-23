import argparse
import collections
import pathlib
from simpledataset.common import SimpleDatasetFactory, ImageClassificationDataset, ObjectDetectionDataset, DatasetWriter


def concat_datasets(main_txt_filepaths, output_filepath):
    datasets = [SimpleDatasetFactory().load(f.read_text(), f.parent) for f in main_txt_filepaths]

    dataset_types = [d.type for d in datasets]
    if len(set(dataset_types)) != 1:
        raise ValueError(f"Cannot concatenate different kind of datasets: {dataset_types}")

    dataset_type = dataset_types[0]

    directory = output_filepath.parent

    data = []
    labels = []
    label_id_offset = 0
    annotations = collections.defaultdict(list)
    for dataset in datasets:
        print(f"Concatenating {len(dataset)} images and {len(dataset.labels)} classes.")
        for new_label in dataset.labels:
            if new_label in labels:
                print(f"Class name conflict: {new_label}")
            while new_label in labels:
                new_label = new_label + '.'
            labels.append(new_label)

        relative_path = dataset.base_directory.relative_to(directory)
        for d in dataset:
            new_path = str(relative_path / d[0])
            if dataset_type == 'image_classification':
                new_labels = [i + label_id_offset for i in d[1]]
            elif dataset_type == 'object_detection':
                new_labels = [(i[0] + label_id_offset, *i[1:]) for i in d[1]]
            annotations[new_path].extend(new_labels)
        label_id_offset += len(dataset.labels)

    data = [(key, annotations[key]) for key in annotations]

    if dataset_type == 'image_classification':
        dataset = ImageClassificationDataset(data, directory, label_names=labels)
    elif dataset_type == 'object_detection':
        dataset = ObjectDetectionDataset(data, directory, label_names=labels)

    DatasetWriter(directory).write(dataset, output_filepath)
    print(f"Successfully saved {output_filepath}")


def _is_relative_to(p, p2):
    """Return True if p is relative path to p2.

        Notes:
            PurePath.is_relative_to() is added to Python 3.9.
    """
    try:
        p.relative_to(p2)
        return True
    except ValueError:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('main_txt_filepath', nargs='+', type=pathlib.Path)
    parser.add_argument('output_filepath', type=pathlib.Path)

    args = parser.parse_args()

    if len(args.main_txt_filepath) < 2:
        parser.error("Needs multiple input txt files.")

    if not all(_is_relative_to(p, args.output_filepath.parent) for p in args.main_txt_filepath):
        parser.error("All input directories must be relative to the output directory.")

    if args.output_filepath.exists():
        parser.error(f"{args.output_filepath} already exists.")

    concat_datasets(args.main_txt_filepath, args.output_filepath)


if __name__ == '__main__':
    main()
