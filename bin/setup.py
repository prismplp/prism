import setuptools
import shutil
import os
from pathlib import Path
path=os.path.dirname(os.path.abspath(__file__))
#shutil.copyfile(path+"/gcn.py", path+"/kgcn/gcn.py")
print(setuptools.find_packages())
setuptools.setup(
    name="T-PRISM",
    version="1.0",
    author="Ryosuke Kojima",
    author_email="kojima.ryosuke.8e@kyoto-u.ac.jp",
    description="T-PRISM",
    long_description="T-PRISM",
    long_description_content_type="text/markdown",
    url="https://github.com/prismplp/prism",
    packages=setuptools.find_packages(),
    entry_points = {
        'console_scripts' : [
            'tprism = tprism.torch_tprism:main',
            ],
    },
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
