import argparse
import pathlib
from simpledataset.common import DatasetWriter
from simpledataset.converters import CocoReader, OpenImagesODReader, OpenImagesVRReader


_READERS = {'coco': CocoReader,
            'openimages_od': OpenImagesODReader,
            'openimages_vr': OpenImagesVRReader}


def convert_from(source_format, output_filepath, args):
    reader = _READERS[source_format]()
    dataset = reader.read(**vars(args))

    directory = output_filepath.parent
    DatasetWriter(directory).write(dataset, output_filepath)
    print(f"Successfully saved {output_filepath}.")


def main():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest='source_format')
    for name, reader in _READERS.items():
        p = subparsers.add_parser(name)
        reader.add_arguments(p)

    parser.add_argument('output_filepath', type=pathlib.Path)

    args = parser.parse_args()

    if args.output_filepath.exists():
        parser.error(f"{args.output_filepath} already exists.")

    convert_from(args.source_format, args.output_filepath, args)


if __name__ == '__main__':
    main()
