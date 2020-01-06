from setuptools import setup
from setuptools import find_packages

setup(name='DeepFRIer',
      version='0.0.1',
      description='Implementation of Deep Functional Residue Identification',
      author='Vladimir Gligorijevic',
      author_email='vgligorijevic@flatironinstitute.org',
      url='https://github.com/flatironinstitute/DeepFRIer',
      download_url='https://github.com/flatironinstitute/DeepFRIer',
      license='FlatironInstitute',
      install_requires=['numpy',
                        'keras',
                        'tensorflow',
                        'networkx',
                        'scikit-learn',
                        'pickle',
                        'glob',
                        ],
      extras_require={
          'visualization': ['matplotlib', 'seaborn'],
      },
      package_data={'DeepFRIer': ['README.md']},
      packages=find_packages())
