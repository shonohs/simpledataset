import setuptools


setuptools.setup(name='simpledataset',
                 version='0.1.0',
                 description="Utility tools for SIMPLE vision dataset format.",
                 packages=setuptools.find_packages(),
                 license='MIT',
                 url='https://github.com/shonohs/simpledataset',
                 classifiers=[
                     'Intended Audience :: Developers',
                     'License :: OSI Approved :: MIT License'
                 ],
                 entry_points={
                     'console_scripts': [
                         'dataset_convert_from=simpledataset.commands.convert_from:main',
                         'dataset_convert_to=simpledataset.commands.convert_to:main',
                         'dataset_filter=simpledataset.commands.filter:main',
                         'dataset_map=simpledataset.commands.map:main',
                         'dataset_summary=simpledataset.commands.summary:main'
                     ]})
