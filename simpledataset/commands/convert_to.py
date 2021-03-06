import argparse
import pathlib
from simpledataset.common import SimpleDatasetFactory, DatasetWriter
from simpledataset.converters import CocoWriter, GoogleAutoMLODWriter, TaskConverter

_WRITERS = {'coco': CocoWriter,
            'google_automl_od': GoogleAutoMLODWriter}


def convert_to(main_txt_filepath, target_format, output_filepath, args):
    dataset = SimpleDatasetFactory().load(main_txt_filepath)

    if target_format in ['image_classification', 'object_detection', 'visual_relationship']:
        dataset = TaskConverter().convert(dataset, target_format, output_filepath.parent)
        DatasetWriter().write(dataset, output_filepath)
        print(f"Successfully saved to {output_filepath}")
    elif target_format in _WRITERS:
        writer = _WRITERS[target_format]()
        writer.write(dataset, images_dir=output_filepath.parent, **vars(args))
        print(f"Successfully saved to {output_filepath}")
    else:
        raise RuntimeError(f"Unsupported format: {target_format}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('main_txt_filepath', type=pathlib.Path)

    subparsers = parser.add_subparsers(dest='target_format')
    for name, writer in _WRITERS.items():
        p = subparsers.add_parser(name)
        writer.add_arguments(p)
    for c in ['image_classification', 'object_detection', 'visual_relationship']:
        subparsers.add_parser(c)

    parser.add_argument('output_filepath', type=pathlib.Path)

    args = parser.parse_args()

    if args.output_filepath.exists():
        parser.error(f"{args.output_filepath} already exists.")

    args.output_filepath.parent.mkdir(parents=True, exist_ok=True)
    convert_to(args.main_txt_filepath, args.target_format, args.output_filepath, args)


if __name__ == '__main__':
    main()
