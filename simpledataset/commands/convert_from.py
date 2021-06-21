import argparse
import pathlib
from simpledataset.common import DatasetWriter
from simpledataset.converters import CocoReader


def convert_from(input_filepath, source_format, output_filepath):
    if source_format == 'coco':
        dataset = CocoReader().read(input_filepath, input_filepath.parent)
    else:
        raise RuntimeError(f"Unsupported format: {source_format}")

    directory = output_filepath.parent
    DatasetWriter(directory).write(dataset, output_filepath)
    print(f"Successfully saved {output_filepath}.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_filepath', type=pathlib.Path)
    parser.add_argument('source_format', choices=['coco'])
    parser.add_argument('output_filepath', type=pathlib.Path)

    args = parser.parse_args()

    if args.output_filepath.exists():
        parser.error(f"{args.output_filepath} already exists.")

    convert_from(args.input_filepath, args.source_format, args.output_filepath)


if __name__ == '__main__':
    main()
