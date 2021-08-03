import argparse
import pathlib
from simpledataset.common import DatasetWriter
from simpledataset.converters import CocoReader, HicoDetReader, OpenImagesODReader, OpenImagesVRReader, VisualGenomeVRReader


_READERS = {'coco': CocoReader,
            'hicodet': HicoDetReader,
            'openimages_od': OpenImagesODReader,
            'openimages_vr': OpenImagesVRReader,
            'visualgenome_vr': VisualGenomeVRReader}


def convert_from(source_format, output_filepath, skip_images, args):
    reader = _READERS[source_format]()
    dataset = reader.read(**vars(args))

    DatasetWriter().write(dataset, output_filepath, copy_images=not skip_images)
    print(f"Successfully saved {output_filepath}.")


def main():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest='source_format')
    for name, reader in _READERS.items():
        p = subparsers.add_parser(name)
        reader.add_arguments(p)

    parser.add_argument('output_filepath', type=pathlib.Path)
    parser.add_argument('--skip_images', action='store_true', help="Do not copy images. Useful when the dataset is too large.")

    args = parser.parse_args()

    if args.output_filepath.exists():
        parser.error(f"{args.output_filepath} already exists.")

    convert_from(args.source_format, args.output_filepath, args.skip_images, args)


if __name__ == '__main__':
    main()
