import argparse
import pathlib
from simpledataset.common import SimpleDatasetFactory, ImageClassificationDataset, ObjectDetectionDataset, VisualRelationshipDataset, DatasetWriter


def map_dataset(main_txt_filepath, output_filepath, mappings_list):
    dataset = SimpleDatasetFactory().load(main_txt_filepath.read_text(), main_txt_filepath.parent)
    mappings = {int(src): int(dst) for src, dst in mappings_list}

    if dataset.type == 'image_classification':
        data = [(image, [mappings.get(x, x) for x in labels if mappings.get(x, x) >= 0]) for image, labels in dataset]
        dataset = ImageClassificationDataset(data, main_txt_filepath.parent)
    elif dataset.type == 'object_detection':
        data = [(image, [(mappings.get(x[0], x[0]), *x[1:]) for x in labels if mappings.get(x[0], x[0]) >= 0]) for image, labels in dataset]
        dataset = ObjectDetectionDataset(data, main_txt_filepath.parent)
    elif dataset.type == 'visual_relationship':
        data = [(image, [(mappings.get(x[0], x[0]), *x[1:5], mappings.get(x[5], x[5]), *x[6:10], mappings.get(x[10], x[10])) for x in labels]) for image, labels in dataset]
        data = [(image, [x for x in labels if x[0] >= 0 and x[5] >= 0 and x[10] >= 0]) for image, labels in data]
        dataset = VisualRelationshipDataset(data, main_txt_filepath.parent)
    else:
        raise RuntimeError

    DatasetWriter().write(dataset, output_filepath, skip_labels_txt=True)
    print(f"Successfully saved {output_filepath}")


def generate_mapping(src_labels_filepath, dst_labels_filepath):
    print(f"Getting mappings from {src_labels_filepath} to {dst_labels_filepath}")
    src_list = [n for n in src_labels_filepath.read_text().splitlines() if n]
    dst_list = [n for n in dst_labels_filepath.read_text().splitlines() if n]
    dst_map = {n: i for i, n in enumerate(dst_list)}

    mappings_list = []
    for i, name in enumerate(src_list):
        if name not in dst_map:
            print(f"Label {name} is not in the destination labels.txt. This label will be removed.")
            dst_index = -1
        else:
            dst_index = dst_map[name]

        if i != dst_index:
            mappings_list.append((i, dst_index))
            print(f"Detected label mappings: {i} => {dst_index}")

    return mappings_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('main_txt_filepath', type=pathlib.Path)
    parser.add_argument('output_filepath', type=pathlib.Path)
    parser.add_argument('--map', nargs=2, action='append', metavar=('src_class_id', 'dst_class_id'))
    parser.add_argument('--map_all', nargs=2, type=pathlib.Path, help="Given 2 labels.txt files, update the annotations so that it align with the second labels.txt.")
    args = parser.parse_args()

    if args.main_txt_filepath.parent != args.output_filepath.parent:
        parser.error("The output file must be in the same directory.")

    if args.output_filepath.exists():
        parser.error(f"{args.output_filepath} already exists.")

    if (not (args.map or args.map_all)) or (args.map and args.map_all):
        parser.error("You must specify either --map or --map_all.")

    mappings_list = generate_mapping(args.map_all[0], args.map_all[1]) if args.map_all else args.map

    map_dataset(args.main_txt_filepath, args.output_filepath, mappings_list)


if __name__ == '__main__':
    main()
