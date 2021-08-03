import json
import pathlib
import tqdm
from simpledataset.common import VisualRelationshipDataset


class VisualGenomeVRReader:
    def read(self, input_json_filepath, input_images_dir, **args):
        def load_box(obj):
            x = obj['x']
            y = obj['y']
            x2 = x + obj['w']
            y2 = y + obj['h']
            name = obj['names'][0] if 'names' in obj else obj['name']
            return (name.lower(), x, y, x2, y2)

        print(f"Loading {input_json_filepath}")
        with open(input_json_filepath) as f:
            data = json.load(f)

        print(f"Loaded {len(data)} entries.")

        dataset_data = []
        for d in tqdm.tqdm(data, "Converting", disable=None):
            image_filename = str(d['image_id']) + '.jpg'
            labels = []
            for relationship in d['relationships']:
                labels.append((*load_box(relationship['subject']),
                               *load_box(relationship['object']),
                               'verb_' + relationship['predicate'].lower()))
            dataset_data.append((image_filename, labels))

        # Get label names.
        predicate_set = set()
        object_set = set()

        for d in dataset_data:
            for r in d[1]:
                predicate_set.add(r[10])
                object_set.add(r[0])
                object_set.add(r[5])

        label_names = sorted(list(object_set)) + sorted(list(predicate_set))
        label_name_map = {name: i for i, name in enumerate(label_names)}

        # Replace label names with label indexes.
        dataset_data = [(image_filename, [(label_name_map[x[0]], *x[1:5], label_name_map[x[5]], *x[6:10], label_name_map[x[10]]) for x in labels])
                        for image_filename, labels in dataset_data]

        return VisualRelationshipDataset(dataset_data, input_images_dir, label_names=label_names)

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('input_json_filepath', type=pathlib.Path)
        parser.add_argument('input_images_dir', type=pathlib.Path)


class VisualGenomeVRWriter:
    def write(self, dataset, output_filepath, images_dir):
        raise NotImplementedError("Writing VisualGenome dataset is not supported.")
