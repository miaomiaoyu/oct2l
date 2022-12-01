#!/bin/bash

# This script extracts all files in the .tar downloaded from Flywheel using
# fw_project_tar_get.py
# 
# Nov 30, 2022

# Extracts all files with .npy extension
tar -xvzf ../data/project-files.tar '*.npy'; 

# Pulls all the files and move them to parent folder
find . -type f -exec mv {} ./ \;  

# Delete files with 'Radial' or 'fundus' in their names
find . -type f -name '**Radial**' -delete
find . -type f -name '**fundus**' -delete

# Remove all subfolders
rm -R -- *(-/)