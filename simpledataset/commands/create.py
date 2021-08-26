import argparse
import pathlib
from simpledataset.common import SimpleDatasetFactory, DatasetWriter


def create(image_filepaths, output_filepath):
    print("Creating a dataset with empty labels.")
    data = [(str(i), []) for i in image_filepaths]

    # Create an image_classificaiton dataset. Since there is no labels any type of dataset is ok.
    dataset = SimpleDatasetFactory().create('image_classification', data, pathlib.Path.cwd(), label_names=[])
    DatasetWriter().write(dataset, output_filepath, copy_images=True, skip_labels_txt=True)
    print(f"Successfully created {output_filepath}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_filepaths', nargs='+', type=pathlib.Path)
    parser.add_argument('--output_filepath', '-o', required=True, type=pathlib.Path)

    args = parser.parse_args()

    if args.output_filepath.exists():
        parser.error(f"{args.output_filepath} already exists.")

    create(args.image_filepaths, args.output_filepath)


if __name__ == '__main__':
    main()
