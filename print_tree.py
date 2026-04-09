import os

def print_tree(start_path='.', prefix=''):
    """
    Recursively prints a tree-like structure of directories and files,
    ignoring hidden folders/files (those starting with '.').
    
    Args:
        start_path (str): The folder path to start from (default = current directory).
        prefix (str): Internal use for indentation.
    """
    # List all entries and sort so dirs come first
    entries = sorted(e for e in os.listdir(start_path) if not e.startswith('.'))
    entries_dirs = [e for e in entries if os.path.isdir(os.path.join(start_path, e))]
    entries_files = [e for e in entries if os.path.isfile(os.path.join(start_path, e))]

    # Print directories first
    for idx, folder in enumerate(entries_dirs):
        connector = "└── " if idx == len(entries_dirs)-1 and not entries_files else "├── "
        print(prefix + connector + folder + "/")
        # Recurse into directory
        extension = "    " if idx == len(entries_dirs)-1 and not entries_files else "│   "
        print_tree(os.path.join(start_path, folder), prefix + extension)

    # Print files
    for idx, file in enumerate(entries_files):
        connector = "└── " if idx == len(entries_files)-1 else "├── "
        print(prefix + connector + file)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Print a tree-like directory structure")
    parser.add_argument("path", nargs='?', default=".", help="Directory path to print (default=current dir)")
    args = parser.parse_args()

    print(args.path + "/")
    print_tree(args.path)