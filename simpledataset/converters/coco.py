import collections
import json
import pathlib
from simpledataset.common import ObjectDetectionDataset


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
            annotations[annotation['image_id']].append((annotation['category_id'], int(bbox[0]), int(bbox[1]), int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])))

        for image in data['images']:
            image_filename = image['file_name']
            labels = annotations[image['id']]

            dataset_data.append((image_filename, labels))

        return ObjectDetectionDataset(dataset_data, input_images_dir)

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('input_json_filepath', type=pathlib.Path)
        parser.add_argument('input_images_dir', type=pathlib.Path)


class CocoWriter:
    def write(self, dataset, output_filepath, images_dir):
        assert dataset.type == 'object_detection'

        annotations = []
        images = []
        annotation_index = len(dataset)
        for i, (image_filename, labels) in enumerate(dataset):
            for class_id, x, y, x2, y2 in labels:
                area = (x2 - x) * (y2 - y)
                annotations.append({'id': annotation_index,
                                    'image_id': i,
                                    'category_id': class_id,
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
            categories.append({'id': i, 'name': label})

        coco_data = {'info': {}, 'images': images, 'annotations': annotations, 'categories': categories}

        output_filepath.write_text(json.dumps(coco_data))
