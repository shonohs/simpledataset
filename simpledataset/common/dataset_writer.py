import logging
import zipfile
import tqdm

logger = logging.getLogger(__name__)


def _make_unique_filepath(filepath):
    if not filepath.exists():
        return filepath

    for i in range(100):
        tmp_filepath = filepath.parent / f'{filepath.stem}{i}{filepath.suffix}'
        if not tmp_filepath.exists():
            return tmp_filepath

    raise RuntimeError(f"Failed to find a unique filename for {filepath}")


class LabelWriter:
    def __init__(self, directory):
        self._directory = directory


class ImageClassificationLabelWriter(LabelWriter):
    def write(self, file_handler, dataset):
        for image, labels in dataset:
            assert all(isinstance(i, int) for i in labels)
            file_handler.write(f"{image} {','.join([str(i) for i in labels])}\n")


class ZipLabelWriter(LabelWriter):
    def write(self, file_handler, dataset):
        label_zip_filepath = _make_unique_filepath(self._directory / 'labels.zip')
        label_zip_filename = label_zip_filepath.name
        # Use zlib to compress the labels.zip. It's fastest and have good compression ratio.
        with zipfile.ZipFile(label_zip_filepath, mode='w', compression=zipfile.ZIP_DEFLATED) as zip_f:
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

    def write(self, dataset, output_filepath, skip_labels_txt=False, copy_images=False):
        output_filepath.parent.mkdir(parents=True, exist_ok=True)

        # Generate labels.txt
        if not skip_labels_txt:
            labels_txt_filepath = _make_unique_filepath(output_filepath.parent / 'labels.txt')
            if labels_txt_filepath.name != 'labels.txt':
                logger.warning(f"Saving labels to {labels_txt_filepath} since labels.txt already exists. Please make sure to rename it to labels.txt before using the dataset.")
            labels_txt_filepath.write_text('\n'.join(dataset.labels))

        data = [(image, labels) for image, labels in dataset]

        if copy_images:
            new_data = []
            images_zip_filepath = _make_unique_filepath(output_filepath.parent / 'images.zip')
            images_zip_filename = images_zip_filepath.name
            logger.info(f"Saving images to {images_zip_filepath}")

            has_duplicated_entry_name = self._has_duplicated_entry_name(dataset)

            with zipfile.ZipFile(images_zip_filepath, mode='w', compression=zipfile.ZIP_STORED) as f:
                for i, (image, labels) in enumerate(tqdm.tqdm(data, "Copying images.", disable=None)):
                    entry_name = image.split('@')[-1]
                    if has_duplicated_entry_name:
                        suffix = entry_name.split('.')[-1]
                        entry_name = f'{i}.{suffix}'
                    image_binary = dataset.read_image_binary(image)
                    with f.open(entry_name, 'w') as zf:
                        zf.write(image_binary)
                    new_data.append((f'{images_zip_filename}@{entry_name}', labels))
            data = new_data

        label_writer = self.LABEL_WRITERS[dataset.type](output_filepath.parent)
        with open(output_filepath, 'w') as f:
            label_writer.write(f, data)

    @staticmethod
    def _has_duplicated_entry_name(dataset):
        entry_name_set = set()
        for image, _ in dataset:
            entry_name = image.split('@')[-1]
            if entry_name in entry_name_set:
                return True
            entry_name_set.add(entry_name)
        return False
