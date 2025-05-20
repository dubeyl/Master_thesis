import os
import shutil
from pathlib import Path
import yaml
import sys

def move_mkv_file(processed_folder, output_folder):
    """Moves a single .mkv file from the processed folder to the server, keeping the full structure."""
    processed_folder = Path(processed_folder)
    output_folder = Path(output_folder)

    print(f"Looking for MKV file in: {processed_folder}")

    # Find .mkv files in the processed folder
    mkv_files = list(processed_folder.rglob("*.mkv"))

    if not mkv_files:
        print(f"No MKV file found in {processed_folder}. Nothing to move.")
        return

    if len(mkv_files) > 1:
        print(f"Warning: More than one MKV file found. Moving only the first one: {mkv_files[0]}")

    mkv_file = mkv_files[0]  # Select the first MKV file

    # Find the parent folder of the processed folder (should contain pupa_X)
    base_root = processed_folder.parents[1]  # Assuming structure is Automated_test/pupa_X/recording_X

    # Preserve the full path from Automated_test downwards
    relative_path = processed_folder.relative_to(base_root)
    target_file = output_folder / relative_path / mkv_file.name

    # Ensure the target directory exists
    os.makedirs(target_file.parent, exist_ok=True)

    # Move the file
    shutil.move(mkv_file, target_file)
    print(f"Moved: {mkv_file} -> {target_file}")

    # Remove the processed folder after moving the MKV file
    shutil.rmtree(processed_folder)
    print(f"Removed processed folder: {processed_folder}")

def main():
    """Main function that moves a processed MKV file to the server."""
    config = yaml.load(open("LD_config.yaml", 'r'), Loader=yaml.FullLoader)
    output_folder = config['output_folder']  # Server destination folder

    if len(sys.argv) > 1:  # Check if a folder path is passed
        processed_folder = sys.argv[1]
    else:
        print("No folder path provided. Exiting.")
        sys.exit(1)

    move_mkv_file(processed_folder, output_folder)
    print("Done moving MKV file and deleting the processed folder.")

if __name__ == "__main__":
    main()
