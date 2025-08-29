#!/bin/bash

# SmartCut PyPI Upload Script

set -e

# Navigate to project root (two levels up from distribution/pypi/)
cd "$(dirname "$0")/../.."

echo "Building SmartCut package for PyPI..."

# Clean previous builds
rm -rf build/ dist/ *.egg-info/

# Build the package
python -m build

ls -la dist/

python -m twine upload dist/*