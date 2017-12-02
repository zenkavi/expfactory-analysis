from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

setup(
    # Application name:
    name="expanalysis",

    # Version number (initial):
    version="0.0.0",

    # Application author details:
    author=["Vanessa Sochat","Ian Eisenberg"],
    author_email=["vsochat@stanford.edu","ieisenbe@stanford.edu "],

    # Packages
    packages=find_packages(),

    # Data files
    include_package_data=True,
    zip_safe=False,

    # Details
    url="http://www.github.com/expfactory",

    license="LICENSE",
    description="Python module for experiment factory experiment analysis.",
    keywords='analysis behavior neuroscience experiment factory',
    install_requires = [
                        'kabuki',
                        'hddm',
                        'numpy==1.11.3',
                        'numexpr',
                        'pymc',
                        'scipy',
                        'seaborn',
                        'statsmodels'],
    dependency_links = [
        "git+https://github.com/hddm-devs/kabuki@df8b748bd455e06f29fe88391e6d936eb90a562c#egg=kabuki-999.0.0"
    ]

)
