import argparse
import collections
import pathlib
from simpledataset.common import SimpleDatasetFactory


def print_summary(main_txt_filepath):
    dataset = SimpleDatasetFactory().load(main_txt_filepath)
    num_class_samples = collections.Counter()
    additional_summaries = []

    if dataset.type == 'image_classification':
        for image, labels in dataset:
            num_class_samples.update(labels)
    elif dataset.type == 'object_detection':
        max_num_boxes_per_image = 0
        for image, labels in dataset:
            num_class_samples.update(x[0] for x in labels)
            max_num_boxes_per_image = max(max_num_boxes_per_image, len(labels))
        additional_summaries.append(f"The max number of boxes per image: {max_num_boxes_per_image}")
    elif dataset.type == 'visual_relationship':
        for image, labels in dataset:
            num_class_samples.update(x[0] for x in labels)
            num_class_samples.update(x[5] for x in labels)
            num_class_samples.update(x[10] for x in labels)
    else:
        raise RuntimeError

    print(f"The dataset type: {dataset.type}")
    print(f"The number of images: {len(dataset)}")
    print(f"The number of classes: {len(num_class_samples)}")
    print(f"Max class id: {dataset.get_max_class_id()}")

    for s in additional_summaries:
        print(s)

    print("Class distribution:")
    labels = dataset.labels
    for class_id in sorted(num_class_samples.keys()):
        print(f"    Class {class_id} ({labels[class_id]}): {num_class_samples[class_id]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('main_txt_filepath', type=pathlib.Path)

    args = parser.parse_args()
    if not args.main_txt_filepath.exists():
        parser.error(f"Cannot find {args.main_txt_filepath}")

    print_summary(args.main_txt_filepath)


if __name__ == '__main__':
    main()
