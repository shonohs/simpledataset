import logging
import zipfile

logger = logging.getLogger(__name__)


def _get_unique_filename(directory, filename):
    if not (directory / filename).exists():
        return filename

    stem, ext = filename.split('.', maxsplit=1)

    for i in range(100):
        tmp_filename = f'{stem}{i}.{ext}'
        if not (directory / tmp_filename).exists():
            return tmp_filename

    raise RuntimeError(f"Failed to find a unique filename for {filename}")


class LabelWriter:
    def __init__(self, directory):
        self._directory = directory


class ImageClassificationLabelWriter(LabelWriter):
    def write(self, file_handler, dataset):
        for image, labels in dataset:
            assert all(isinstance(i, int) for i in labels)
            file_handler.write(f"{image} {'.'.join([str(i) for i in labels])}\n")


class ZipLabelWriter(LabelWriter):
    def write(self, file_handler, dataset):
        label_zip_filename = _get_unique_filename(self._directory, 'labels.zip')
        # Use zlib to compress the labels.zip. It's fastest and have good compression ratio.
        with zipfile.ZipFile(self._directory / label_zip_filename, mode='w', compression=zipfile.ZIP_DEFLATED) as zip_f:
            for i, (image, labels) in enumerate(dataset):
                with zip_f.open(f'{i}.txt', 'w') as zf:
                    self.write_label(zf, labels)
                file_handler.write(f'{image} {label_zip_filename}@{i}.txt\n')


class ObjectDetectionLabelWriter(ZipLabelWriter):
    def write_label(self, file_handler, labels):
        for label_id, x, y, x2, y2 in labels:
            file_handler.write(f'{label_id} {x} {y} {x2} {y2}\n'.encode('utf-8'))


class VisualRelationshipLabelWriter(ZipLabelWriter):
    def write_label(self, file_handler, labels):
        for subject_id, sx, sy, sx2, sy2, object_id, ox, oy, ox2, oy2, predicate_id in labels:
            file_handler.write(f'{subject_id} {sx} {sy} {sx2} {sy2} {object_id} {ox} {oy} {ox2} {oy2} {predicate_id}\n'.encode('utf-8'))


class DatasetWriter:
    LABEL_WRITERS = {'image_classification': ImageClassificationLabelWriter,
                     'object_detection': ObjectDetectionLabelWriter,
                     'visual_relationship': VisualRelationshipLabelWriter}

    def __init__(self, directory):
        self._directory = directory

    def write(self, dataset, output_filepath, skip_labels_txt=False):
        assert '/' not in str(output_filepath)

        # Generate labels.txt
        if not skip_labels_txt:
            labels_txt_filepath = self._directory / 'labels.txt'
            if labels_txt_filepath.exists():
                labels_txt_filename = _get_unique_filename(self._directory, 'labels.txt')
                labels_txt_filepath = self._directory / labels_txt_filename
                logger.warning(f"labels.txt already exists. Saving to {labels_txt_filepath}")
            labels_txt_filepath.write_text('\n'.join(dataset.labels))

        label_writer = self.LABEL_WRITERS[dataset.type](self._directory)
        with open(self._directory / output_filepath, 'w') as f:
            label_writer.write(f, dataset)
