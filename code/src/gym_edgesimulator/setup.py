from setuptools import setup

setup(name='gym_edgesimulator',
      package_dir={'': 'src'},
      version='0.0.1',
      install_requires=['gym',
                        'networkx']
)