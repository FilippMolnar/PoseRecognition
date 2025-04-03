import os

def rename_files(folder_path):
    if not os.path.isdir(folder_path):
        print("Invalid folder path.")
        return
    
    files = sorted(os.listdir(folder_path))  # Sort to maintain order
    count = 1
    
    for file in files:
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):  # Ignore directories
            extension = os.path.splitext(file)[1]  # Get file extension
            new_name = f"{count}{extension}"
            new_path = os.path.join(folder_path, new_name)
            os.rename(file_path, new_path)
            count += 1
    
    print("Renaming complete.")

# Usage example
folder = "data/squat_new"
rename_files(folder)