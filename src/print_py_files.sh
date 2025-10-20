#!/bin/bash

# Output file
output_file="python_files_output.txt"

# Clear the output file if it already exists
> "$output_file"

# Loop over all .py files in the current directory
for file in *.py; do
    # Skip if no .py files found
    [ -e "$file" ] || continue

    {
        echo "$file"
        echo "-----------"
        cat "$file"
        echo ""
    } >> "$output_file"
done

echo "Done! Output written to $output_file"
