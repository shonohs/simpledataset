import collections
import csv
import logging
import pathlib
import tqdm
from simpledataset.common import VisualRelationshipDataset
from simpledataset.converters.openimages_od import OpenImagesODReader


logger = logging.getLogger(__name__)


class OpenImagesVRReader:
    def read(self, bbox_csv_filepath, openimages_images_dir, **args):
        """
        Args:
             bbox_csv_filepath (pathlib.Path): Filepath to annotation-bbox.csv.
             images_dir (pathlib.Path): The directory contains image files.
        """
        logger.warning("Note that this converter doesn't copy image files.")

        annotations = collections.defaultdict(list)
        image_sizes = {}
        label_name_set = set()
        relationship_set = set()
        with open(bbox_csv_filepath, newline='') as f:
            reader = csv.DictReader(f)
            for row in tqdm.tqdm(reader):
                image_id = row['ImageID']
                label_name1 = row['LabelName1']
                label_name2 = row['LabelName2']
                relationship_label = row['RelationshipLabel']

                label_name_set.add(label_name1)
                label_name_set.add(label_name2)
                relationship_set.add(relationship_label)

                if image_id not in image_sizes:
                    image_sizes[image_id] = OpenImagesODReader._get_image_size(openimages_images_dir, image_id)
                w, h = image_sizes[image_id]

                annotations[image_id].append((label_name1,
                                              int(round(float(row['XMin1']) * w)), int(round(float(row['YMin1']) * h)),
                                              int(round(float(row['XMax1']) * w)), int(round(float(row['YMax1']) * h)),
                                              label_name2,
                                              int(round(float(row['XMin2']) * w)), int(round(float(row['YMin2']) * h)),
                                              int(round(float(row['XMax2']) * w)), int(round(float(row['YMax2']) * h)),
                                              relationship_label))

        logger.info(f"Loaded {len(annotations)} annotations and {len(image_sizes)} images.")

        label_names = sorted(label_name_set) + sorted(relationship_set)
        label_id_map = {n: i for i, n in enumerate(label_names)}

        dataset_data = []
        for image_id in annotations:
            image_filename = OpenImagesODReader._resolve_image_filename(openimages_images_dir, image_id)
            labels = [(label_id_map[a[0]], *a[1:5], label_id_map[a[5]], *a[6:10], label_id_map[a[10]]) for a in annotations[image_id]]
            dataset_data.append((image_filename, labels))

        return VisualRelationshipDataset(dataset_data, openimages_images_dir, label_names=label_names)

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('bbox_csv_filepath', type=pathlib.Path)
        parser.add_argument('openimages_images_dir', type=pathlib.Path)


class OpenImagesVRWriter:
    def write(self, dataset, output_filepath, images_dir):
        raise NotImplementedError("Export to OpenImages Visual Relation format is not supported.")
