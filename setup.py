from setuptools import setup
from setuptools import find_packages

setup(name='DeepFRI',
      version='0.0.1',
      description='Implementation of Deep Functional Residue Identification',
      author='Vladimir Gligorijevic',
      author_email='vgligorijevic@flatironinstitute.org',
      url='https://github.com/flatironinstitute/DeepFRI',
      download_url='https://github.com/flatironinstitute/DeepFRI',
      license='FlatironInstitute',
      install_requires=['numpy==1.18.5',
                        'tensorflow-gpu==2.3.1',
                        'networkx==2.4',
                        'scikit-learn==0.23.1',
                        'biopython==1.76',
                        ],
      extras_require={
          'visualization': ['matplotlib', 'seaborn'],
      },
      package_data={'DeepFRI': ['README.md']},
      packages=find_packages())
