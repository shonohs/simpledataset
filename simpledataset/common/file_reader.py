import pathlib
import zipfile


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
