from setuptools import setup, find_packages

setup(name='pccf',
      version='0.1',
      packages=find_packages('src'),
      package_dir={"": "src"},
      install_requires=[
            "loguru==0.5.3",
            "matplotlib==3.3.3",
            "numpy==1.20.1",
            "pytest==6.2.2",
            "scipy==1.6.0",
            "tqdm==4.56.2"
      ]
      )
