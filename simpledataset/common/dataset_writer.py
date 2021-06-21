import zipfile


class DatasetWriter:
    def __init__(self, directory):
        self._directory = directory

    def write(self, dataset, output_filepath):
        assert '/' not in str(output_filepath)

        if dataset.type == 'image_classification':
            with open(self._directory / output_filepath, 'w') as f:
                for image, labels in dataset:
                    f.write(f"{image} {'.'.join([str(i) for i in labels])}\n")
        elif dataset.type == 'object_detection':
            label_zip_filename = self._get_unique_filename('labels.zip')
            with open(self._directory / output_filepath, 'w') as f:
                with zipfile.ZipFile(self._directory / label_zip_filename, mode='w', compression=zipfile.ZIP_STORED) as zip_f:
                    for i, (image, labels) in enumerate(dataset):
                        with zip_f.open(f'{i}.txt', 'w') as zf:
                            self._write_object_detection_labels(zf, labels)
                        f.write(f'{image} {label_zip_filename}@{i}.txt\n')

    @staticmethod
    def _write_object_detection_labels(file_obj, labels):
        for label_id, x, y, x2, y2 in labels:
            file_obj.write(f'{label_id} {x} {y} {x2} {y2}\n'.encode('utf-8'))

    def _get_unique_filename(self, filename):
        if not (self._directory / filename).exists():
            return filename

        stem, ext = filename.split('.', maxsplit=1)

        for i in range(100):
            tmp_filename = f'{stem}{i}.{ext}'
            if not (self._directory / tmp_filename).exists():
                return tmp_filename

        raise RuntimeError(f"Failed to find a unique filename for {filename}")
