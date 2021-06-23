import argparse
import pathlib
from simpledataset.common import SimpleDatasetFactory, ImageClassificationDataset, ObjectDetectionDataset, DatasetWriter


def map_dataset(main_txt, directory, output_filepath, mappings_list):
    dataset = SimpleDatasetFactory().load(main_txt, directory)
    mappings = {int(src): int(dst) for src, dst in mappings_list}

    if dataset.type == 'image_classification':
        data = [(image, [mappings.get(x, x) for x in labels]) for image, labels in dataset]
        dataset = ImageClassificationDataset(data, directory)
    elif dataset.type == 'object_detection':
        data = [(image, [(mappings.get(x[0], x[0]), *x[1:]) for x in labels]) for image, labels in dataset]
        dataset = ObjectDetectionDataset(data, directory)
    else:
        raise RuntimeError

    DatasetWriter(directory).write(dataset, output_filepath)
    print(f"Successfully saved {output_filepath}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('main_txt_filepath', type=pathlib.Path)
    parser.add_argument('output_filename')
    parser.add_argument('--map', nargs=2, required=True, action='append', metavar=('src_class_id', 'dst_class_id'))
    args = parser.parse_args()

    if '/' in args.output_filename:
        parser.error("The output file must be in the same directory.")

    main_txt = args.main_txt_filepath.read_text()
    directory = args.main_txt_filepath.parent

    if (directory / args.output_filename).exists():
        parser.error(f"{args.output_filename} already exists.")

    map_dataset(main_txt, directory, args.output_filename, args.map)


if __name__ == '__main__':
    main()
