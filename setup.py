from setuptools import setup

setup(
    name='Peepo',
    version='0.1.1',
    author='Hayo Carrette',
    author_email='hayo.ce@gmail.com',
    packages=['peepo', 'peepo.bot', 'peepo.evolution', 'peepo.playground', 'peepo.pp', 'peepo.tests',
              'peepo.visualize'],
    scripts=[],
    url='https://peepo.ai',
    license='LICENSE',
    description='Personhood through Predictive Processing.',
    long_description=open('README.md').read(),
    install_requires=['scipy', 'numpy', 'matplotlib', 'pgmpy', 'pandas', 'wrapt', 'networkx', 'pygame', 'flask',
                      'cython', 'pomegranate', 'pathos'],
    setup_requires=[]
)

# "pip install git+https://github.com/jmschrei/pomegranate.git"
