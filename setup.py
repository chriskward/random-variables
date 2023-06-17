from setuptools import setup, find_packages

setup( name = "simulation",
	version = "1.05",
	author = 'Chris Ward',
	author_email = 'chrisward@email.com',
	packages=find_packages(include=['simulation','simulation.*']),
	install_requires = ['numpy>=1.26.0'])
