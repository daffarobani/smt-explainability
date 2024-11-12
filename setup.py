from setuptools import setup

CLASSIFIERS = """\
Development Status :: 5 - Production/Stable
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved :: BSD License
Programming Language :: C++
Programming Language :: Python
Programming Language :: Python :: 3
Programming Language :: Python :: 3.8
Programming Language :: Python :: Implementation :: CPython
Topic :: Software Development
Topic :: Scientific/Engineering
Operating System :: Microsoft :: Windows
Operating System :: Unix
Operating System :: MacOS
"""

metadata = dict(
    name="smt-explainability",
    version="0.1.0",
    description="",
    long_description="",
    author="Daffa Robani et al.",
    author_email="robani.daffa@gmail.com",
    license="BSD-3",
    classifiers=[_f for _f in CLASSIFIERS.split("\n") if _f],
    packages=["smt_explainability"],
    install_requires=["smt~=2.7.0"],
    extras_require={},
    python_requires=">=3.8",
    zip_safe=False,
    # url="https://github.com/SMTorg/smt",  # use the URL to the github repo
    # download_url="https://github.com/SMTorg/smt/releases",
)

setup(**metadata)
