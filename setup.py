from setuptools import setup, find_packages
from codecs import open
from os import path

this_folder = path.abspath(path.dirname(__file__))
with open(path.join(this_folder,'README.md'),encoding='utf-8') as inf:
  long_description = inf.read()

setup(
  name='pythologist',
  version='2.1.1',
  test_suite='nose2.collector.collector',
  description='inForm PerkinElmer Reader - Python interface to read outputs of the PerkinElmer inForm software;\
    Pythologist-image-utilities: Functions to assist in working with image files;\
    Pythologist-schemas: Check the assumptions of inputs for pythologist ahead of reading',
  long_description=long_description,
  url='https://github.com/jason-weirather/pythologist',
  author='Jason L Weirather',
  author_email='JasonL_Weirather@dfci.harvard.edu',
  license='Apache License, Version 2.0',
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering :: Bio-Informatics',
    'License :: OSI Approved :: Apache Software License'
  ],
  keywords='bioinformatics',
  packages=find_packages(include=['pythologist','pythologist.*']),
  package_data={
    'pythologist':[
      'schema_data/*.json',
      'schema_data/inputs/*.json',
      'schema_data/inputs/platforms/InForm/*.json'
    ]
  },
  install_requires=['pandas==2.3.3',
                    'numpy==1.26.4',
                    'scipy',
                    'h5py',
                    'imageio',
                    'xmltodict', 
                    'scikit-image>=0.16.2',
                    'imagecodecs',
                    'tifffile>=2019.7.26',
                    'scikit-learn',
                    'jsonschema',
                    'openpyxl',
                    'tables',
                    'importlib-metadata >= 1.0 ; python_version < "3.8"',
                    'opencv-python-headless'
                    ], 
  extras_require = {
        'test':  ["pythologist-test-images"]
  },
  include_package_data = True,
  entry_points = {
    'console_scripts':['pythologist-stage=pythologist.schemas.cli.stage_tool:cli',
                       'pythologist-templates=pythologist.schemas.cli.template_tool:cli',
                       'pythologist-run=pythologist.schemas.cli.run_tool:cli',
                       'pythologist-report=pythologist.schemas.cli.report_tool:cli'
                      ]
  },
)
