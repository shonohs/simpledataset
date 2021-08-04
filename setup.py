import pathlib
import setuptools


readme_filepath = pathlib.Path(__file__).parent / 'README.md'

setuptools.setup(name='simpledataset',
                 version='0.2.2',
                 description="Utility tools for SIMPLE vision dataset format.",
                 long_description=readme_filepath.read_text(),
                 long_description_content_type='text/markdown',
                 packages=setuptools.find_packages(),
                 install_requires=['pillow', 'requests', 'scipy', 'tenacity', 'tqdm'],
                 license='MIT',
                 url='https://github.com/shonohs/simpledataset',
                 classifiers=[
                     'Intended Audience :: Developers',
                     'License :: OSI Approved :: MIT License'
                 ],
                 entry_points={
                     'console_scripts': [
                         'dataset_concat=simpledataset.commands.concat:main',
                         'dataset_convert_from=simpledataset.commands.convert_from:main',
                         'dataset_convert_to=simpledataset.commands.convert_to:main',
                         'dataset_defrag=simpledataset.commands.defrag:main',
                         'dataset_download=simpledataset.commands.download:main',
                         'dataset_draw=simpledataset.commands.draw:main',
                         'dataset_filter=simpledataset.commands.filter:main',
                         'dataset_map=simpledataset.commands.map:main',
                         'dataset_pack=simpledataset.commands.pack:main',
                         'dataset_summary=simpledataset.commands.summary:main'
                     ]})
