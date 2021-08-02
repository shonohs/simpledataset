import argparse
import collections
import hashlib
import pathlib
from simpledataset.common import SimpleDatasetFactory, ImageClassificationDataset, ObjectDetectionDataset, VisualRelationshipDataset, DatasetWriter


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
    image_md5s = {}
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
            elif dataset_type == 'visual_relationship':
                new_labels = [(i[0] + label_id_offset, *i[1:5], i[5] + label_id_offset, *i[6:10], i[10] + label_id_offset) for i in d[1]]
            annotations[new_path].extend(new_labels)
            if new_path not in image_md5s:
                image_md5s[new_path] = hashlib.md5(dataset.read_image_binary(d[0])).hexdigest()
        label_id_offset += len(dataset.labels)

    annotations = _dedup_images(annotations, image_md5s)
    data = [(key, annotations[key]) for key in annotations]

    if dataset_type == 'image_classification':
        dataset = ImageClassificationDataset(data, directory, label_names=labels)
    elif dataset_type == 'object_detection':
        dataset = ObjectDetectionDataset(data, directory, label_names=labels)
    elif dataset_type == 'visual_relationship':
        dataset = VisualRelationshipDataset(data, directory, label_names=labels)

    copy_images = any(f.parent != output_filepath.parent for f in main_txt_filepaths)
    DatasetWriter().write(dataset, output_filepath, copy_images=copy_images)
    print(f"Successfully saved {output_filepath}")


def _dedup_images(annotations, image_hashs):
    """Find duplicated entries and merge them.
    Args:
        annotations: dict. image_filepath => labels.
        image_hashs: dict. image_filepath => hash.
    """

    hash_to_path = {}
    for image_path, image_hash in image_hashs.items():
        if image_hash not in hash_to_path:
            hash_to_path[image_hash] = image_path

    new_annotations = {}
    for image_filepath, labels in annotations.items():
        image_hash = image_hashs[image_filepath]
        new_path = hash_to_path[image_hash]

        if new_path in new_annotations:
            labels = new_annotations[new_path] + labels

        new_annotations[new_path] = labels

    return new_annotations


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
