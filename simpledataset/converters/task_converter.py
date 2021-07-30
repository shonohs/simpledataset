"""Change the task of the dataset. For example, OD => IC.
"""
import zipfile
import tqdm
from simpledataset.common import ImageClassificationDataset, ObjectDetectionDataset


def _make_unique_filepath(filepath):
    if not filepath.exists():
        return filepath

    for i in range(100):
        tmp_filepath = filepath.parent / f'{filepath.stem}{i}{filepath.suffix}'
        if not tmp_filepath.exists():
            return tmp_filepath

    raise RuntimeError(f"Failed to find a unique filename for {filepath}")


class TaskConverter:
    def convert(self, dataset, destination_dataset_type, output_directory):
        ROUTES = {('visual_relationship', 'object_detection'): [self._convert_vr_od],
                  ('visual_relationship', 'image_classification'): [self._convert_vr_od, self._convert_od_ic],
                  ('object_detection', 'image_classification'): [self._convert_od_ic],
                  ('image_classification', 'object_detection'): [self._convert_ic_od]}

        source_dataset_type = dataset.type
        converters = ROUTES.get((source_dataset_type, destination_dataset_type))
        if not converters:
            raise RuntimeError(f"Cannot ocnvert {source_dataset_type} into {destination_dataset_type}")

        for c in converters:
            dataset = c(dataset, output_directory)

        return dataset

    def _convert_od_ic(self, dataset, output_directory):
        # Crop Bounding Box and make it into classification dataset.
        data = []
        images_zip_filepath = _make_unique_filepath(output_directory / 'images.zip')
        images_zip_filename = images_zip_filepath.name
        with zipfile.ZipFile(images_zip_filepath, mode='w', compression=zipfile.ZIP_STORED) as f:
            for image, labels in tqdm.tqdm(dataset, "Cropping images", disable=None):
                for label in labels:
                    index = len(data)
                    pil_image = dataset.load_image(image)
                    cropped_image = pil_image.crop(tuple(label[1:]))
                    with f.open(f'{index}.jpg', 'w') as zf:
                        cropped_image.save(zf, format='JPEG')
                    data.append((f'{images_zip_filename}@{index}.jpg', [label[0]]))

        return ImageClassificationDataset(data, output_directory, label_names=dataset.labels)

    def _convert_vr_od(self, dataset, output_directory):
        def convert_label(label):
            x = min(label[1], label[6])
            y = min(label[2], label[7])
            x2 = max(label[3], label[8])
            y2 = max(label[4], label[9])
            return (label[10], x, y, x2, y2)

        data = [(image, [convert_label(x) for x in labels]) for image, labels in dataset]
        return ObjectDetectionDataset(data, output_directory, label_names=dataset.labels, images_directory=dataset.base_images_directory)

    def _convert_ic_od(self, dataset, output_directory):
        def convert_labels(image, labels):
            image = dataset.load_image(image)
            w, h = image.size
            return [(x, 0, 0, w, h) for x in labels]

        data = [(image, convert_labels(image, labels)) for image, labels in dataset]
        return ObjectDetectionDataset(data, output_directory, label_names=dataset.labels, images_directory=dataset.base_images_directory)
