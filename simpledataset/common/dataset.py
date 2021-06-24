import io
import logging
import pathlib
import zipfile
import PIL.Image


logger = logging.getLogger(__name__)


class ImageDataset:
    # This variable must be overwritten by a child class.
    LABEL_LOADER_CLASS = None

    def __init__(self, data, directory, label_names=None):
        assert isinstance(data, list)
        if data:
            assert len(data[0]) == 2
            assert isinstance(data[0][0], str)

        self._data = data
        self._directory = directory
        self._label_names = label_names
        self._reader = FileReader(directory)

    @classmethod
    def load(cls, main_txt, directory):
        data = []

        label_loader = cls.LABEL_LOADER_CLASS(directory)
        try:
            line = ''
            for line in main_txt.splitlines():
                image_path, labels = line.strip().split(maxsplit=1)
                labels = label_loader.load(labels)
                data.append((image_path, labels))
        except Exception:
            print(f"Failed to parse '{line}'")
            raise

        return cls(data, directory)

    def __iter__(self):
        for d in self._data:
            yield d

    def __len__(self):
        return len(self._data)

    def load_image(self, image_filename):
        with io.BytesIO(self.read_image_binary(image_filename)) as f:
            return PIL.Image.open(f)

    def read_image_binary(self, image_filename):
        return self._reader.read(image_filename, 'rb')

    def get_labels(self):
        labels_filepath = self._directory / 'labels.txt'
        if labels_filepath.exists():
            labels = labels_filepath.read_text().splitlines()
        else:
            logger.warning("labels.txt is not found. Generating class names...")
            labels = [str(i) for i in range(self.get_max_class_id() + 1)]

        assert len(labels) == len(set(labels))
        assert len(labels) >= self.get_max_class_id()
        return labels

    def get_num_classes(self):
        raise NotImplementedError

    def get_max_class_id(self):
        raise NotImplementedError

    @property
    def base_directory(self):
        return self._directory

    @property
    def labels(self):
        if self._label_names:
            return self._label_names
        self._label_names = self.get_labels()
        return self._label_names


class LabelLoader:
    def __init__(self, directory):
        self._directory = directory
        self._reader = FileReader(self._directory)


class ImageClassificationLabelLoader(LabelLoader):
    def load(self, labels):
        data = [int(s) for s in labels.split(',')]
        assert len(set(data)) == len(data), f"Duplicated labels found: {data}"
        return data


class ObjectDetectionLabelLoader(LabelLoader):
    def load(self, filepath):
        data = []
        for line in self._reader.read(filepath).splitlines():
            label_id, x_min, y_min, x_max, y_max = [int(s) for s in line.strip().split()]
            data.append((label_id, x_min, y_min, x_max, y_max))
        return data


class ImageClassificationDataset(ImageDataset):
    LABEL_LOADER_CLASS = ImageClassificationLabelLoader

    @property
    def type(self):
        return 'image_classification'

    def get_num_classes(self):
        classes_set = set()
        for image, labels in self:
            classes_set.update(labels)
        return len(classes_set)

    def get_max_class_id(self):
        max_id = 0
        for image, labels in self:
            if max(labels) > max_id:
                max_id = max(labels)
        return max_id


class ObjectDetectionDataset(ImageDataset):
    LABEL_LOADER_CLASS = ObjectDetectionLabelLoader

    @property
    def type(self):
        return 'object_detection'

    def get_num_classes(self):
        classes_set = set()
        for image, labels in self:
            classes_set.update(s[0] for s in labels)
        return len(classes_set)

    def get_max_class_id(self):
        max_id = 0
        for image, labels in self:
            if labels:
                m = max(s[0] for s in labels)
                max_id = max(m, max_id)
        return max_id


class SimpleDatasetFactory:
    SUPPORTED_DATASET = {'image_classification': ImageClassificationDataset,
                         'object_detection': ObjectDetectionDataset}

    def load(self, main_txt, directory=pathlib.Path('.')):
        dataset_type = self._detect_dataset_type(main_txt)
        if dataset_type not in self.SUPPORTED_DATASET:
            raise RuntimeError(f"Unsupported dataset type: {dataset_type}")

        dataset_class = self.SUPPORTED_DATASET[dataset_type]
        return dataset_class.load(main_txt, directory)

    @staticmethod
    def _detect_dataset_type(main_txt):
        for line in main_txt.splitlines():
            if '.' in line.strip().split()[1]:
                return 'object_detection'
            else:
                return 'image_classification'


class FileReader:
    """Read a file in <zip_filepath>@<entry_name> format, or a normal file path.."""
    def __init__(self, base_dir):
        assert isinstance(base_dir, pathlib.Path)
        self._zip_objects = {}
        self._base_dir = base_dir

    def read(self, filepath, mode='r'):
        assert mode in ('r', 'rb')

        if '@' in filepath:
            zip_filepath, entrypath = filepath.split('@')
            if zip_filepath not in self._zip_objects:
                self._zip_objects[zip_filepath] = zipfile.ZipFile(self._base_dir / zip_filepath)

            with self._zip_objects[zip_filepath].open(entrypath) as f:
                return f.read().decode('utf-8') if mode == 'r' else f.read()
        else:
            with open(self._base_dir / filepath, mode) as f:
                return f.read()
