#!/bin/bash

# Navigate to project root (two levels up from distribution/pyinstaller/)
cd "$(dirname "$0")/../.."

rm -rf build dist

pyinstaller --distpath ./dist --workpath ./build --onefile -n smartcut smartcut/__main__.py

tar -czvf "dist/smartcut_linux.tar" -C dist smartcut
mv ./dist/smartcut dist/smartcut_x86_64
./dist/smartcut_x86_64 --version
