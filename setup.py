import setuptools


setuptools.setup(name='simpledataset',
                 version='0.1.1',
                 description="Utility tools for SIMPLE vision dataset format.",
                 packages=setuptools.find_packages(),
                 install_requires=['pillow', 'requests', 'tenacity', 'tqdm'],
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
                         'dataset_download=simpledataset.commands.download:main',
                         'dataset_filter=simpledataset.commands.filter:main',
                         'dataset_map=simpledataset.commands.map:main',
                         'dataset_summary=simpledataset.commands.summary:main'
                     ]})
