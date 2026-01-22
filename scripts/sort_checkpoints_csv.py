import csv
import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Sort checkpoints CSV by creation time.")
    parser.add_argument("file", help="Path to the CSV file to sort.")
    parser.add_argument("--output", help="Path to output sorted CSV. Defaults to overwriting input.", default=None)
    args = parser.parse_args()

    input_path = args.file
    output_path = args.output if args.output else input_path

    if not os.path.exists(input_path):
        print(f"Error: File {input_path} not found.")
        sys.exit(1)

    print(f"Reading {input_path}...")
    with open(input_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        data = list(reader)

    print(f"Sorting {len(data)} rows by 'creation_time'...")
    # creation_time format is "YYYY-MM-DD HH:MM:SS" which sorts lexicographically correctly.
    # Handle missing values by putting them last or first? Let's assume valid strings or empty.
    # Empty strings ("") come before dates in string sort, which implies "oldest/unknown" first.
    # If we want unknown last, we need a key.
    
    def sort_key(row):
        return row.get('creation_time', "")

    data.sort(key=sort_key)

    print(f"Writing sorted data to {output_path}...")
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    
    print("Done.")

if __name__ == "__main__":
    main()
