try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


setup(name='The_day_after_tomorrow',
      use_scm_version=True,
      setup_requires=['setuptools_scm'],
      version='1.0',
      description='Asteroid atmospheric entry solver',
      author='ACSE project',
      packages=['speed']
      )
