import collections
import csv
import logging
import pathlib
import PIL.Image
import tqdm
from simpledataset.common import ObjectDetectionDataset


logger = logging.getLogger(__name__)


class OpenImagesODReader:
    def read(self, bbox_csv_filepath, openimages_images_dir, include_occluded=True, include_depiction=False, include_inside=False, **args):
        """
        Args:
             bbox_csv_filepath (pathlib.Path): Filepath to annotation-bbox.csv.
             images_dir (pathlib.Path): The directory contains image files.
        """
        annotations = collections.defaultdict(list)
        image_sizes = {}
        label_name_set = set()
        total_count = 0
        box_count = 0
        with open(bbox_csv_filepath, newline='') as f:
            reader = csv.DictReader(f)
            for row in tqdm.tqdm(reader):
                total_count += 1
                image_id = row['ImageID']
                label_name = row['LabelName']
                label_name_set.add(label_name)
                if image_id not in image_sizes:
                    image_sizes[image_id] = self._get_image_size(openimages_images_dir, image_id)
                w, h = image_sizes[image_id]
                if not include_occluded and row.get('IsOccluded') == '1':
                    continue
                if not include_depiction and row.get('IsDepiction') == '1':
                    continue
                if not include_inside and row.get('IsInside') == '1':
                    continue

                box_count += 1
                annotations[image_id].append((label_name,
                                              int(round(float(row['XMin']) * w)), int(round(float(row['YMin']) * h)),
                                              int(round(float(row['XMax']) * w)), int(round(float(row['YMax']) * h))))

        logger.info(f"Total number of boxes is {total_count}. Skipped {total_count - box_count}.")

        label_names = sorted(label_name_set)
        label_id_map = {n: i for i, n in enumerate(label_names)}

        dataset_data = []
        for image_id in annotations:
            image_filename = self._resolve_image_filename(openimages_images_dir, image_id)
            labels = [(label_id_map[a[0]], *a[1:]) for a in annotations[image_id]]
            dataset_data.append((image_filename, labels))

        return ObjectDetectionDataset(dataset_data, openimages_images_dir, label_names=label_names)

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('bbox_csv_filepath', type=pathlib.Path)
        parser.add_argument('openimages_images_dir', type=pathlib.Path)
        parser.add_argument('--exclude_occluded', action='store_false', dest='include_occluded')
        parser.add_argument('--include_depiction', action='store_true')
        parser.add_argument('--include_inside', action='store_true')

    @staticmethod
    def _resolve_image_filename(images_dir, image_id):
        """Get a filepath for the image."""
        filename = image_id + '.jpg'
        if (images_dir / filename).exists():
            return filename
        elif (images_dir / f'train_{image_id[0]}' / filename).exists():
            # If user downloaded train_x.tar.gz and extrated images, the images will be in train_x directories.
            return f'train_{image_id[0]}/{filename}'

        return None

    @staticmethod
    def _get_image_size(images_dir, image_id):
        filename = OpenImagesODReader._resolve_image_filename(images_dir, image_id)
        if not filename:
            raise RuntimeError(f"Image is not found: {image_id}")

        with PIL.Image.open(images_dir / filename) as image:
            return (image.width, image.height)


class OpenImagesODWriter:
    def write(self, dataset, output_filepath, images_dir):
        raise NotImplementedError("Export to OpenImages OD format is not supported.")
