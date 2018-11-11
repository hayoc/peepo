from setuptools import setup

setup(
    name='Peepo',
    version='0.1.0',
    author='Hayo Carrette',
    author_email='hayo.ce@gmail.com',
    packages=['peepo', 'peepo.bot', 'peepo.predictive_processing', 'peepo.predictive_processing.v3', 'peepo.playground'],
    scripts=[],
    url='https://peepo.ai',
    license='LICENSE',
    description='Personhood through Predictive Processing.',
    long_description=open('README.md').read(),
    install_requires=['scipy', 'numpy', 'matplotlib', 'pgmpy', 'pandas', 'wrapt', 'networkx==1.11', 'pygame'],
    setup_requires=['scipy', 'numpy', 'matplotlib', 'pgmpy', 'pandas', 'wrapt', 'networkx==1.11', 'pygame']
)
