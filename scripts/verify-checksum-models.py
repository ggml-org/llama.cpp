#!/usr/bin/env python3

import logging
import os
import hashlib

logger = logging.getLogger("verify-checksum-models")


def sha256sum(file):
    block_size = 16 * 1024 * 1024  # 16 MB block size
    b = bytearray(block_size)
    file_hash = hashlib.sha256()
    mv = memoryview(b)
    with open(file, 'rb', buffering=0) as f:
        while True:
            n = f.readinto(mv)
            if not n:
                break
            file_hash.update(mv[:n])

    return file_hash.hexdigest()


# Define the path to the llama directory (parent folder of script directory)
llama_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# Define the file with the list of hashes and filenames
hash_list_file = os.path.join(llama_path, "SHA256SUMS")

# Check if the hash list file exists
if not os.path.exists(hash_list_file):
    logger.error(f"Hash list file not found: {hash_list_file}")
    exit(1)

# Read the hash file content and split it into an array of lines
with open(hash_list_file, "r") as f:
    hash_list = f.read().splitlines()

# Create an array to store the results
results = []

# Loop over each line in the hash list
for line in hash_list:
    # Split the line into hash and filename
    hash_value, filename = line.split("  ")

    # Get the full path of the file by joining the llama path and the filename
    file_path = os.path.join(llama_path, filename)

    # Informing user of the progress of the integrity check
    logger.info(f"Verifying the checksum of {file_path}")

    # Check if the file exists
    if os.path.exists(file_path):
        # Calculate the SHA256 checksum of the file using hashlib
        file_hash = sha256sum(file_path)

        # Compare the file hash with the expected hash
        if file_hash == hash_value:
            valid_checksum = "V"
            file_missing = ""
        else:
            valid_checksum = ""
            file_missing = ""
    else:
        valid_checksum = ""
        file_missing = "X"

    # Add the results to the array
    results.append({
        "filename": filename,
        "valid checksum": valid_checksum,
        "file missing": file_missing
    })


# Print column headers for results table
print("filename".ljust(40) + "valid checksum".center(20) + "file missing".center(20)) # noqa: NP100
print("-" * 80) # noqa: NP100

# Output the results as a table
for r in results:
    print(f"{r['filename']:40} {r['valid checksum']:^20} {r['file missing']:^20}") # noqa: NP100
