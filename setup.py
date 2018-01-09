from distutils.core import setup

setup(
    name='Peepo',
    version='0.1.0',
    author='Hayo Carrette',
    author_email='hayo.ce@gmail.com',
    packages=['peepo', 'peepo.bot', 'peepo.predictive_processing.discrete'],
    scripts=[],
    url='http://conventiont.com',
    license='LICENSE',
    description='Personhood through Predictive Processing.',
    long_description=open('README.rst').read(),
    install_requires=[
        'scipy', 'numpy'],
)