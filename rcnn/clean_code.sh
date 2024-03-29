#!/bin/bash

# This script cleans all Python files in the specified directory by removing invalid characters and null bytes.
DIR="$1"

# Check if the directory argument is provided
if [ -z "$DIR" ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

# Loop through all .py files in the specified directory
for file in "$DIR"/*.py; do
    echo "Cleaning $file..."
    # Use iconv to convert the file, dropping invalid characters, and store the output in a temporary file
    iconv -f utf-8 -t utf-8 -c "$file" > "${file}.cleaned"
    # Use tr to remove null bytes, if any, and overwrite the original file
    tr -d '\000' < "${file}.cleaned" > "${file}"
    # Remove the temporary file
    rm "${file}.cleaned"
done

echo "All Python files in $DIR have been cleaned."
