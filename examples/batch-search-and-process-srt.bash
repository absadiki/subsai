#!/bin/bash
#
# Script to find media files and export them into a list if they DO NOT have a corrosponding .srt, does NOT find subtitles internal to mkv or mp4, or even .srt files with a different filename.
# ex. if show.mkv and show.srt both exist, file will be excluded from the list
# Script does not search for ALL media files, must be edited if you have file extensions not listed.
# subsai command is hard coded as the basic example,(subsai media.txt --model openai/whisper --format srt) if you require something different editing is required.

# Initialize file_written variable
file_written=false
echo Enter root directory for search, subdirectories will also be searched.
echo ""
read -p 'Search root directory: ' ROOT_DIRECTORY
echo ""
# Check if the directory exists
if [ ! -d "$ROOT_DIRECTORY" ]; then
    echo "Error: Directory does not exist. Exiting."
    exit 1
elif [ ! -r "$ROOT_DIRECTORY" ]; then
    echo "Error: Permission denied, check user has access to directory. Exiting."
    exit 1
fi

# Search for files and display the results
files=$(find "$ROOT_DIRECTORY" -iregex '.*\.\(mkv\|m4v\|mp4\|avi\|mov\|mpg\)$' -exec sh -c '[ ! -f "${0%.*}.srt" ] && echo "$0"' {} \;)

# Count the number of files found
file_count=$(echo "$files" | wc -l)

# Display the list of files
echo "$files"
echo "=============================="
echo "Found $file_count files without corresponding .srt files."
echo "=============================="

# Prompt the user to decide if they want to save the list to a file
read -p "Do you want to write the output to a file? (Y/n, default Y): " response
echo ""

# If no response is provided (Enter is pressed), default to "yes"
response=${response:-Y}

# Convert the response to lowercase for case-insensitive comparison
response=$(echo "$response" | tr '[:upper:]' '[:lower:]')

# If the user says "yes", write the output to a file
if [[ "$response" == "y" ]] || [[ "$response" == "yes" ]]; then
    echo "subsai requires *.txt as input file, .txt extension will be added automatically"
    read -p 'Output file name: ' OUTPUT_FILENAME
    echo "$files" > $OUTPUT_FILENAME.txt
    echo ""
    # Check if the file was successfully written
    if [[ -s "$OUTPUT_FILENAME.txt" ]]; then
        echo "$OUTPUT_FILENAME.txt successfully created."
        echo ""
        file_written=true
    else
        echo "Error: Failed to write to $OUTPUT_FILENAME."
        echo ""
        file_written=false
    fi
else
    echo "Output was not written to a file."
fi

if [[ "$file_written" == true ]]; then
    read -p "Begin processing $file_count files with subsai as a background task? (Y/n, default Y): " response_subsai

    # If no response is provided (Enter is pressed), default to "yes"
    response_subsai=${response_subsai:-Y}

    # Convert the response to lowercase for case-insensitive comparison
    response_subsai=$(echo "$response_subsai" | tr '[:upper:]' '[:lower:]')

    # If the user says "yes", begin processing the file with subsai
    if [[ "$response" == "y" ]] || [[ "$response" == "yes" ]]; then
        subsai $OUTPUT_FILENAME.txt --model openai/whisper --format srt > "$OUTPUT_FILENAME.log" 2>&1 &
        # Capture the PID of the last background process
        subsai_pid=$!
        # Output the PIDs of the running background processes
        echo ""
        echo "Background process started with PID: $subsai_pid"
        echo "Output is being logged to $OUTPUT_FILENAME.log"
        echo ""
    else
        echo ""
        echo "Files not processed by subsai."
        echo ""
    fi
else
    echo "Processing will not begin because the output file was not successfully written."
fi
