import collections
import json
import logging
import pathlib
import tqdm
from simpledataset.common import ObjectDetectionDataset


logger = logging.getLogger(__name__)


class CocoReader:
    def read(self, input_json_filepath, input_images_dir, **args):
        data = json.loads(input_json_filepath.read_text())

        image_map = {}
        for image in data['images']:
            image_map[image['id']] = image['file_name']

        dataset_data = []
        annotations = collections.defaultdict(list)
        for annotation in data['annotations']:
            bbox = annotation['bbox']
            new_label = (annotation['category_id'], int(bbox[0]), int(bbox[1]), int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3]))
            if new_label[1] == new_label[3] or new_label[2] == new_label[4]:
                logger.warning(f"Image {annotation['image_id']} has an invalid bounding box: {new_label}. Skipping...")
                continue

            if new_label in annotations[annotation['image_id']]:
                logger.warning(f"Image {annotation['image_id']} has duplicated bounding boxes: {new_label}.")
                continue

            annotations[annotation['image_id']].append(new_label)

        for image in data['images']:
            image_filename = image['file_name']
            labels = annotations[image['id']]

            dataset_data.append((image_filename, labels))

        label_names = self._get_labels(data['categories'])
        return ObjectDetectionDataset(dataset_data, input_images_dir, label_names=label_names)

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('input_json_filepath', type=pathlib.Path)
        parser.add_argument('input_images_dir', type=pathlib.Path)

    @staticmethod
    def _get_labels(categories):
        name_map = {c['id']: c['name'] for c in categories}
        max_id = max(name_map.keys())
        return [name_map.get(i, str(i)) for i in range(max_id + 1)]


class CocoWriter:
    def write(self, dataset, output_filepath, images_dir):
        assert dataset.type == 'object_detection'

        annotations = []
        images = []
        annotation_index = len(dataset)
        for i, (image_filename, labels) in enumerate(tqdm.tqdm(dataset, "Copying images")):
            for class_id, x, y, x2, y2 in labels:
                area = (x2 - x) * (y2 - y)
                annotations.append({'id': annotation_index,
                                    'image_id': i,
                                    'category_id': class_id + 1,  # COCO class_id is 1-indexed in the official dataset file.
                                    'area': area,
                                    'bbox': [x, y, x2 - x, y2 - y],
                                    'iscrowd': 0})
                annotation_index += 1

            image = dataset.load_image(image_filename)
            ext = image_filename.split('.')[-1]
            new_filename = f'{i}.{ext}'
            images.append({'id': i,
                           'width': image.width,
                           'height': image.height,
                           'file_name': new_filename
                           })

            image_binary = dataset.read_image_binary(image_filename)
            (images_dir / new_filename).write_bytes(image_binary)

        categories = []
        for i, label in enumerate(dataset.get_labels()):
            # 1-indexed category id.
            categories.append({'id': i + 1, 'name': label, 'supercategory': 'none'})

        coco_data = {'info': {}, 'images': images, 'annotations': annotations, 'categories': categories}

        output_filepath.write_text(json.dumps(coco_data))
