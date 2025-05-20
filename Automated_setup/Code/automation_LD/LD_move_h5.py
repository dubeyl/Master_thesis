import os
import shutil
from pathlib import Path
import yaml

def move_h5_files(base_folder, output_folder):
	# Walk through all subfolders in the base folde"
	print("Looking up ", base_folder)
	base_folder = Path(base_folder)
	h5_files = list(base_folder.rglob("*.h5"))
	#collect folders containing h5 files to delete them later
	h5_folders = list(set([h5_file.parent for h5_file in h5_files]))
	for h5_file in h5_files:
		print("Found", h5_file)
    # For each .h5 file, move it to the corresponding location in the output folder
	for h5_file in h5_files:
		# Preserve the subfolder structure by determining the relative pa
		relative_path = h5_file.relative_to(base_folder.parent)
		target_file = Path(output_folder) / relative_path
                
		# Ensure the target folder exists
		os.makedirs(target_file.parent, exist_ok=True)       
		# Move the .h5 file to the output folder
		shutil.copy(h5_file, target_file)
		print(f"Moved: {h5_file} -> {target_file}")
	# Remove the original .h5 file and directory
	for h5_folder in h5_folders:
		shutil.rmtree(h5_folder)
		print(f"Removed folder and its content: {h5_folder}")

def main():
    # Example usage:
    config = yaml.load(open("LD_config.yaml", 'r'), Loader=yaml.FullLoader)
    base_folder = config['base_folder']
    output_folder = config['output_folder']

    move_h5_files(base_folder, output_folder)
    print("Done moving .h5 files and deleting them from the base folder")

if __name__ == "__main__":
    main()
