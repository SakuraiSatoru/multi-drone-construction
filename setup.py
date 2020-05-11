from setuptools import setup, find_packages

setup(name='mdac',
      version='0.0.1',
      description='Multi-Drone Autonomous Construction',
      url='',
      author='Zhihao Fang',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      install_requires=['gym', 'numpy-stl','numpy','torch']
)
