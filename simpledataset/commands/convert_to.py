import argparse
import pathlib
from simpledataset.common import SimpleDatasetFactory
from simpledataset.converters import CocoWriter


def convert_to(main_txt, directory, target_format, output_filepath):
    dataset = SimpleDatasetFactory().load(main_txt, directory)

    if target_format == 'coco':
        CocoWriter().write(dataset, output_filepath, output_filepath.parent)
        print(f"Successfully saved to {output_filepath}")
    else:
        raise RuntimeError(f"Unsupported format: {target_format}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('main_txt_filepath', type=pathlib.Path)
    parser.add_argument('target_format', choices=['coco'])
    parser.add_argument('output_filepath', type=pathlib.Path)

    args = parser.parse_args()

    main_txt = args.main_txt_filepath.read_text()
    directory = args.main_txt_filepath.parent

    if args.output_filepath.exists():
        parser.error(f"{args.output_filepath} already exists.")

    convert_to(main_txt, directory, args.target_format, args.output_filepath)


if __name__ == '__main__':
    main()
