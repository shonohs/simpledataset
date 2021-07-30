import argparse
import pathlib
from simpledataset.common import SimpleDatasetFactory, ImageClassificationDataset, ObjectDetectionDataset, VisualRelationshipDataset, DatasetWriter


def defrag(main_txt_filepath, output_filepath):
    dataset = SimpleDatasetFactory().load(main_txt_filepath.read_text(), main_txt_filepath.parent)
    used_labels = set()

    if dataset.type == 'image_classification':
        for image, labels in dataset:
            used_labels.update(labels)
    elif dataset.type == 'object_detection':
        for image, labels in dataset:
            used_labels.update([x[0] for x in labels])
    elif dataset.type == 'visual_relationship':
        for image, labels in dataset:
            used_labels.update([x[0] for x in labels])
            used_labels.update([x[5] for x in labels])
            used_labels.update([x[10] for x in labels])
    else:
        raise RuntimeError

    sorted_used_labels = sorted(list(used_labels))
    mapping = {i: n for n, i in enumerate(sorted_used_labels)}
    print("Here is the plan:")
    for old_index, new_index in mapping.items():
        print(f"{old_index} => {new_index} ({dataset.labels[old_index]})")

    new_label_names = [dataset.labels[sorted_used_labels[i]] for i in range(len(mapping))]
    if dataset.type == 'image_classification':
        data = [(image, [mapping.get(x, x) for x in labels]) for image, labels in dataset]
        dataset = ImageClassificationDataset(data, main_txt_filepath.parent, label_names=new_label_names)
    elif dataset.type == 'object_detection':
        data = [(image, [(mapping.get(x[0], x[0]), *x[1:]) for x in labels]) for image, labels in dataset]
        dataset = ObjectDetectionDataset(data, main_txt_filepath.parent, label_names=new_label_names)
    elif dataset.type == 'visual_relationship':
        data = [(image, [(mapping.get(x[0], x[0]), *x[1:5], mapping.get(x[5], x[5]), *x[5:10], mapping.get(x[10], x[10])) for x in labels]) for image, labels in dataset]
        dataset = VisualRelationshipDataset(data, main_txt_filepath.parent, label_names=new_label_names)
    else:
        raise RuntimeError

    DatasetWriter().write(dataset, output_filepath)
    print(f"Successfully saved {output_filepath}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('main_txt_filepath', type=pathlib.Path)
    parser.add_argument('output_filepath', type=pathlib.Path)

    args = parser.parse_args()

    if args.output_filepath.exists():
        parser.error(f"{args.output_filepath} already exists.")

    defrag(args.main_txt_filepath, args.output_filepath)


if __name__ == '__main__':
    main()
