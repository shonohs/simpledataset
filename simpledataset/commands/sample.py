import argparse
import pathlib
import random
from simpledataset.common import SimpleDatasetFactory, DatasetWriter


def sample(main_txt_filepath, output_filepath, num_images):
    dataset = SimpleDatasetFactory().load(main_txt_filepath)

    if num_images >= len(dataset):
        print(f"Request {num_images} samples, but the dataset has only {len(dataset)} samples.")
        return

    data = []
    sampled_indexes = random.sample(range(len(dataset)), num_images)
    for i, d in enumerate(dataset):
        if i in sampled_indexes:
            data.append(d)

    new_dataset = SimpleDatasetFactory().create(dataset.type, data, main_txt_filepath.parent, dataset.labels)
    copy_images = main_txt_filepath.parent != output_filepath.parent
    DatasetWriter().write(new_dataset, output_filepath, copy_images=copy_images)
    print(f"Successfully saved {output_filepath}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('main_txt_filepath', type=pathlib.Path)
    parser.add_argument('output_filepath', type=pathlib.Path)
    parser.add_argument('--num_images', '-n', default=100, type=int)

    args = parser.parse_args()

    if args.output_filepath.exists():
        parser.error(f"{args.output_filepath} already exists.")

    sample(args.main_txt_filepath, args.output_filepath, args.num_images)


if __name__ == '__main__':
    main()
