import logging
import pathlib
import scipy.io
import tqdm
from simpledataset.common import VisualRelationshipDataset


logger = logging.getLogger(__name__)


class HicoDetReader:
    def read(self, input_mat_filepath, input_images_dir, key_name, **args):
        def read_box(input_array):
            x, x2, y, y2 = (int(input_array[i][0][0]) for i in range(4))
            assert x < x2 and y < y2
            return (x, y, x2, y2)

        data = scipy.io.loadmat(input_mat_filepath)

        action_list = []
        for action in data['list_action']:
            object_name = action[0][0][0]
            predicate_name = action[0][1][0]
            action_list.append((object_name, predicate_name))

        object_name_set = set(x[0] for x in action_list)
        predicate_name_set = set('verb_' + x[1] for x in action_list)
        object_name_set.add('person')

        label_names = sorted(list(object_name_set)) + sorted(list(predicate_name_set))
        action_list = [(label_names.index(x[0]), label_names.index('verb_' + x[1])) for x in action_list]
        person_id = label_names.index('person')

        dataset_data = []
        for d in tqdm.tqdm(data[key_name][0], "Converting", disable=None):
            image_filename = d[0][0]
            labels = []
            for annotation in d[2][0]:
                is_visible = annotation[4][0][0] == 0
                if not is_visible:
                    continue
                action_id = annotation[0][0][0]
                object_id, predicate_id = action_list[action_id - 1]

                person_bbox = [read_box(ann) for ann in annotation[1][0]]
                object_bbox = [read_box(ann) for ann in annotation[2][0]]
                connections = [(x[0], x[1]) for x in annotation[3]]
                for person_index, object_index in connections:
                    labels.append((person_id, *person_bbox[person_index - 1], object_id, *object_bbox[object_index - 1], predicate_id))
            dataset_data.append((image_filename, labels))

        return VisualRelationshipDataset(dataset_data, input_images_dir, label_names=label_names)

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('input_mat_filepath', type=pathlib.Path)
        parser.add_argument('input_images_dir', type=pathlib.Path)
        parser.add_argument('key_name', choices=['bbox_train', 'bbox_test'])


class HicoDetWriter:
    def write(self, dataset, output_filepath, images_dir):
        raise NotImplementedError("Writing HICO-DET format is not supported.")
