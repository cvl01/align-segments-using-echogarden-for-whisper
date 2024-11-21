import os
import argparse

def convert_encoding_inplace(directory):
    # Process each file in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            
            print(f"Converting {filename}")
            try:
                # Read the file with Windows-1252 encoding
                with open(file_path, 'r', encoding='cp1252') as file:
                    content = file.read()
                
                # Write the file with UTF-8 encoding
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(content)
            except:
                print(f"Failed to convert {filename} to UTF-8.")

            print(f"Converted {filename} to UTF-8 in place.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert text files from Windows-1252 encoding to UTF-8 in a given directory')
    parser.add_argument('directory', type=str, help='Directory containing the text files')

    args = parser.parse_args()
    # Specify the directory
    directory = args.directory

    convert_encoding_inplace(directory)
