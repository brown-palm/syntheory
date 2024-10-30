#!/bin/bash
set -e

# cd to the location of this file on disk
cd "$(dirname "$0")"

# run ruff formatter on the below directories, better to be 
# specific than to search the whole repo.
ruff format ../tests
ruff format ../dataset
ruff format ../probe