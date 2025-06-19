#!/bin/bash
set -xeuf -o pipefail

pip install --upgrade pip
pip install -r requirements.txt

pip install git+https://github.com/facebookresearch/segment-anything.git
