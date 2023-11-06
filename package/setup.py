# setup.py

from setuptools import setup, find_packages

setup(
    name='glen',
    version='1.4',
    packages=find_packages(),
    package_data={'glen': ['resources/xpo_glen.json', 'resources/node_tokenized_ids_64_with_event_tag.pt']},
    install_requires=[
        "torch>=1.13.1",
        "transformers>=4.30.2",
        "pytorch-transformers>=1.2.0"
    ],
)
