from setuptools import setup, find_packages
from codecs import open
from os import path

this_folder = path.abspath(path.dirname(__file__))
with open(path.join(this_folder,'README.md'),encoding='utf-8') as inf:
  long_description = inf.read()

setup(
  name='pythologist',
  version='2.1.0',
  test_suite='nose2.collector.collector',
  description='inForm PerkinElmer Reader - Python interface to read outputs of the PerkinElmer inForm software;\
    Pythologist-image-utilities: Functions to assist in working with image files;\
    Pythologist-schemas: Check the assumptions of inputs for pythologist ahead of reading',
  long_description=long_description,
  url='https://github.com/jason-weirather/pythologist',
  author='Jason L Weirather',
  author_email='jason.weirather@gmail.com',
  license='Apache License, Version 2.0',
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering :: Bio-Informatics',
    'License :: OSI Approved :: Apache Software License'
  ],
  keywords='bioinformatics',
  packages=['pythologist',
            'pythologist_reader',
            'pythologist_image_utilities',
           'pythologist_schemas',
            'schema_data',
            'schema_data.inputs',
            'schema_data.inputs.platforms.InForm'],
  install_requires=['pandas>=1.2.2',
                    'numpy',
                    'scipy',
                    'h5py',
                    'imageio',
                    'Pillow',
                    'xmltodict', 
                    'scikit-image>=0.16.2',
                    'imagecodecs',
                    'tifffile>=2019.7.26',
                    'sklearn',
                    'jsonschema',
                    'importlib_resources',
                    'XlsxWriter',
                    'openpyxl',
                    'plotnine',
                    'IPython',
                    'umap',
                    'tables'
                    ], 
  extras_require = {
        'test':  ["pythologist-test-images"]
  },
  include_package_data = True,
  entry_points = {
    'console_scripts':['pythologist-stage=pythologist_schemas.cli.stage_tool:cli',
                       'pythologist-templates=pythologist_schemas.cli.template_tool:cli',
                       'pythologist-run=pythologist_schemas.cli.run_tool:cli',
                       'pythologist-report=pythologist_schemas.cli.report_tool:cli'
                      ]
  },
)
