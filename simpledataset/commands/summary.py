import argparse
import collections
import pathlib
from simpledataset.common import SimpleDatasetFactory


def print_summary(main_txt, directory):
    dataset = SimpleDatasetFactory().load(main_txt, directory)
    num_class_samples = collections.Counter()

    if dataset.type == 'image_classification':
        for image, labels in dataset:
            num_class_samples.update(labels)
    elif dataset.type == 'object_detection':
        for image, labels in dataset:
            num_class_samples.update(x[0] for x in labels)
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
    print("Class distribution:")
    labels = dataset.labels
    for class_id in sorted(num_class_samples.keys()):
        print(f"    Class {class_id} ({labels[class_id]}): {num_class_samples[class_id]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('main_txt_filepath', type=pathlib.Path)

    args = parser.parse_args()

    main_txt = args.main_txt_filepath.read_text()
    directory = args.main_txt_filepath.parent

    print_summary(main_txt, directory)


if __name__ == '__main__':
    main()
