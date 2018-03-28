from setuptools import setup

setup(
    name='Peepo',
    version='0.1.0',
    author='Hayo Carrette',
    author_email='hayo.ce@gmail.com',
    packages=['peepo', 'peepo.bot', 'peepo.predictive_processing.v3'],
    scripts=[],
    url='https://peepo.ai',
    license='LICENSE',
    description='Personhood through Predictive Processing.',
    long_description=open('README.md').read(),
    install_requires=['scipy', 'numpy', 'pgmpy', 'pandas', 'wrapt'],
    setup_requires=['scipy', 'numpy', 'pgmpy', 'pandas', 'wrapt']
)