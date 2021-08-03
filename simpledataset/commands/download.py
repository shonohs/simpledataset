import argparse
import pathlib
import urllib.parse

import requests
import tenacity
import tqdm


def _find_referenced_files(main_txt_filepath):
    referenced_files = set()
    with open(main_txt_filepath) as f:
        for line in f:
            image_filepath, labels = line.strip().split()
            referenced_files.add(image_filepath.split('@')[0])
            if '.' in labels:
                referenced_files.add(labels.split('@')[0])

    return referenced_files


def _replace_filename_in_url(url, filename):
    parts = urllib.parse.urlparse(url)
    path = '/'.join(parts[2].split('/')[:-1] + [filename])
    return urllib.parse.urlunparse((*parts[0:2], path, *parts[3:]))


@tenacity.retry(stop=tenacity.stop_after_attempt(2), retry=tenacity.retry_if_exception_type(requests.RequestException))
def _download_file(url, output_filepath):
    if output_filepath.exists():
        raise RuntimeError(f"{output_filepath} already exists.")

    output_filepath.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, allow_redirects=True) as r:
        total_size = int(r.headers['Content-Length'])
        with open(output_filepath, 'wb') as f:
            with tqdm.tqdm(total=total_size, desc=str(output_filepath), unit_scale=True, unit='B') as pbar:
                for data in r.iter_content(chunk_size=4000000):
                    f.write(data)
                    pbar.update(len(data))
    return output_filepath


def download_dataset(main_txt_url, output_dir):
    main_filename = urllib.parse.urlparse(main_txt_url).path.split('/')[-1]
    main_txt_filepath = output_dir / main_filename
    _download_file(main_txt_url, main_txt_filepath)

    # Download labels.txt
    url = _replace_filename_in_url(main_txt_url, 'labels.txt')
    try:
        _download_file(url, output_dir / 'labels.txt')
    except IOError:
        print("labels.txt is not found.")

    # Download referenced files.
    referenced_filenames = _find_referenced_files(main_txt_filepath)
    for filename in referenced_filenames:
        url = _replace_filename_in_url(main_txt_url, filename)
        _download_file(url, output_dir / filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('main_txt_url', help="URL to the main txt file.")
    parser.add_argument('output_dir', type=pathlib.Path)

    args = parser.parse_args()

    download_dataset(args.main_txt_url, args.output_dir)


if __name__ == '__main__':
    main()
