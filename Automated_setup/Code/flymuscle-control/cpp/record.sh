#!/bin/bash

# Ensure the correct number of arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <tracking|no_tracking> <output_directory>"
    exit 1
fi

# Read arguments
tracking_mode=$1
output_folder=$2

# Validate tracking_mode argument
if [[ "$tracking_mode" != "tracking" && "$tracking_mode" != "no_tracking" ]]; then
    echo "Invalid tracking mode. Use 'tracking' or 'no_tracking'."
    exit 1
fi

# Create and mount RAM disk
#sudo mkdir -p /mnt/nvme_disk/tmpfs
#sudo mount -t tmpfs -o size=8G tmpfs /mnt/tmpfs/
nvme_output_dir="/mnt/nvme_disk/$output_folder"

# Record datetime
datetime=$(date +"%Y%m%d-%H%M%S")
echo "Recording started at: $datetime"

# Run the appropriate recording program based on tracking_mode
if [ "$tracking_mode" == "tracking" ]; then
    sudo nice -n -20 ionice -c 1 -n 0 ./record_parallel_io $nvme_output_dir &
    PID=$!
elif [ "$tracking_mode" == "no_tracking" ]; then
    sudo nice -n -20 ionice -c 1 -n 0 ./record_parallel_no_track $nvme_output_dir &
    PID=$!
fi

# Wait for the recording program to finish
wait $PID

echo "Recording completed saved at" $nvme_output_dir
