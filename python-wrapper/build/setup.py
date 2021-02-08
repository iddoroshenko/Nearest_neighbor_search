from distutils.core import setup

setup(name='engineWrapper',
      version='0.1',
      author="Nobody",
      description="""Install precompiled extension""",
      packages=[''],
      package_data={'': ['engineWrapper.so']},
      )
