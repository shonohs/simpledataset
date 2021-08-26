import re
from .file_reader import FileReader


class DatasetTypeDetector:
    def detect(self, main_txt, directory):
        for line in main_txt.splitlines():
            fields = line.strip().split()
            if len(fields) == 1:
                continue

            label_field = fields[1]
            if re.match(r'[0-9,]+', label_field):
                return 'image_classification'

            reader = FileReader(directory)
            label_file_contents = reader.read(label_field)

            label_file_first_line = label_file_contents.splitlines()[0].strip()
            fields = label_file_first_line.split()
            if len(fields) == 5:
                return 'object_detection'

            if len(fields) == 11:
                return 'visual_relationship'

            raise RuntimeError("Failed to detect the dataset type.")

        # Empty dataset.
        return 'image_classification'
