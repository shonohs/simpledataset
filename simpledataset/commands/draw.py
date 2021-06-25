import argparse
import logging
import pathlib
import PIL.ImageDraw
import tqdm
from simpledataset.common import SimpleDatasetFactory


COLOR_CODES = ["black", "brown", "red", "orange", "yellow", "green", "blue", "violet", "grey", "white"]
logger = logging.getLogger(__name__)


def _draw_od_labels(image, annotations, label_names):
    draw = PIL.ImageDraw.Draw(image)
    for class_id, x, y, x2, y2 in annotations:
        color = COLOR_CODES[class_id % len(COLOR_CODES)]
        draw.rectangle(((x, y), (x2, y2)), outline=color)
        draw.text((x, y), label_names[class_id])


def _draw_ic_labels(image, annotations, label_names):
    draw = PIL.ImageDraw.Draw(image)
    for i, class_id in enumerate(annotations):
        draw.text((0, i * 10), label_names[class_id], color='red')


def draw_dataset(main_txt_filepath, output_dir):
    dataset = SimpleDatasetFactory().load(main_txt_filepath.read_text(), main_txt_filepath.parent)

    drawer = {'image_classificaiton': _draw_ic_labels,
              'object_detection': _draw_od_labels}[dataset.type]

    for image_filename, annotations in tqdm.tqdm(dataset):
        image = dataset.load_image(image_filename)
        image_filename = image_filename.split('@')[-1]
        output_filepath = output_dir / image_filename
        if output_filepath.exists():
            logger.warning(f"{output_filepath} already exists. skipping the image...")
            continue
        output_filepath.parent.mkdir(parents=True, exist_ok=True)
        drawer(image, annotations, dataset.labels)
        image.save(output_filepath)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('main_txt_filepath', type=pathlib.Path)
    parser.add_argument('output_dir', type=pathlib.Path)

    args = parser.parse_args()

    if not args.main_txt_filepath.exists():
        parser.error(f"{args.main_txt_filepath} is not found.")

    draw_dataset(args.main_txt_filepath, args.output_dir)


if __name__ == '__main__':
    main()
