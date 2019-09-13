from setuptools import setup

with open("README.md", 'r') as f:
    long_description = f.read()

setup(
   name='zellige',
   version='1.0',
   description='Tools for penrose and other aperiodic tilings',
   license="MIT",
   long_description=long_description,
   author='Dennis Yi',
   author_email='dtyi088@gmail.com',
   url="https://github.com/dtyi/penrose",
   packages=['zellige'],  #same as name
   install_requires=['numpy', 'scipy'], #external packages as dependencies
)