#!/bin/bash

# Check if the input directory is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <input_directory>"
  exit 1
fi

INPUT_DIR="$1"
RAM_DIR="/mnt/ramdisk"  # Temporary directory in RAM
OUTPUT_DIR="/mnt/nvme_disk/output"  # The final output directory on the disk

# Ensure the RAM directory exists
mkdir -p "$RAM_DIR"

# Ensure the final output directory exists
mkdir -p "$OUTPUT_DIR"

# Mount the RAM disk (15GB in this example; adjust as needed)
mount -t tmpfs -o size=15G tmpfs "$RAM_DIR"
echo "Mounted RAM disk at $RAM_DIR"

# Start the record_parallel_io script and save images to the RAM disk
echo "Starting the image recording script..."

# Run the record_parallel_io script with the target directory set to the RAM disk
./record_parallel_io "$RAM_DIR" &

# Get the process ID of the recording script
RECORD_PID=$!

# Wait for the recording process to finish
wait $RECORD_PID

echo "Recording completed. Moving images from RAM to output directory..."

# Now move the recorded files from RAM disk to the final output directory
# Find all image files in the RAM directory and move them to the output directory
find "$RAM_DIR" -type f | while read -r file; do
    # Create the corresponding path in the final output directory
    target_path="$OUTPUT_DIR$(echo "$file" | sed "s|$RAM_DIR|$OUTPUT_DIR|")"
    
    # Create the necessary subdirectories in the output directory
    mkdir -p "$(dirname "$target_path")"
    
    # Move the file to the final destination
    mv "$file" "$target_path"
done

echo "Files successfully moved to $OUTPUT_DIR"

# Clean up RAM disk by unmounting and freeing up memory
echo "Cleaning up RAM disk..."
umount "$RAM_DIR"
rmdir "$RAM_DIR"
echo "RAM disk unmounted and cleaned."

echo "Process complete. All files are now in $OUTPUT_DIR."

