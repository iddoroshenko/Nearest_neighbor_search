from glob import glob
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
      Pybind11Extension(
            "engineWrapper",
            ['python_wrapper/enginewrapper.cpp', 'engine.cpp']
      ),
]

setup(name='engineWrapper',
      version='0.1',
      author="Nobody",
      description="""Install precompiled extension""",
      packages=[''],
      package_data={'': ['engineWrapper.so']},
      cmdclass={"build_ext": build_ext},
      ext_modules=ext_modules
      )
