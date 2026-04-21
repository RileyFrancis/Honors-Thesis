"""
Debug script to investigate why CSV files aren't matching with TXT data dictionaries
Run this in a new cell to see what's happening
"""

import os
from pathlib import Path

# Your directories
data_dir = "/home/rif17002/honors_thesis/ABCD_files"
data_dict_dir = "/home/rif17002/honors_thesis/ABCD_data_dicts"

print("=" * 80)
print("DEBUG: Investigating File Matching Issues")
print("=" * 80)

# Check if directories exist
print(f"\n1. Checking if directories exist:")
print(f"   Data dir exists: {os.path.exists(data_dir)}")
print(f"   Data dict dir exists: {os.path.exists(data_dict_dir)}")

# List CSV/TSV files
print(f"\n2. CSV/TSV files in {data_dir}:")
print("-" * 80)
csv_files = []
for pattern in ['*.csv', '*.tsv', '*.txt']:
    csv_files.extend(list(Path(data_dir).glob(pattern)))

if csv_files:
    print(f"   Found {len(csv_files)} files:")
    for i, f in enumerate(csv_files[:10], 1):  # Show first 10
        print(f"   {i}. {f.name}")
    if len(csv_files) > 10:
        print(f"   ... and {len(csv_files) - 10} more")
else:
    print("   NO FILES FOUND!")

# List TXT files in data dict directory
print(f"\n3. TXT files in {data_dict_dir}:")
print("-" * 80)
try:
    dict_files = list(Path(data_dict_dir).glob('*.txt'))
    if dict_files:
        print(f"   Found {len(dict_files)} files:")
        for i, f in enumerate(dict_files[:10], 1):  # Show first 10
            print(f"   {i}. {f.name}")
        if len(dict_files) > 10:
            print(f"   ... and {len(dict_files) - 10} more")
    else:
        print("   NO TXT FILES FOUND!")
except Exception as e:
    print(f"   ERROR: {e}")

# Try to match a few examples
print(f"\n4. Matching examples (trying to match first 5 CSV files):")
print("-" * 80)
if csv_files and dict_files:
    for csv_file in csv_files[:5]:
        csv_name = csv_file.name
        base_name = csv_file.stem  # filename without extension
        
        print(f"\n   CSV: {csv_name}")
        print(f"   Base name: {base_name}")
        
        # Try exact match
        exact_match = Path(data_dict_dir) / f"{base_name}.txt"
        print(f"   Looking for: {exact_match.name}")
        print(f"   Exact match exists: {exact_match.exists()}")
        
        # Try case-insensitive search
        matches = []
        base_lower = base_name.lower()
        for dict_file in dict_files:
            dict_lower = dict_file.stem.lower()
            if base_lower == dict_lower:
                matches.append(dict_file.name)
            elif base_lower in dict_lower or dict_lower in base_lower:
                matches.append(f"{dict_file.name} (partial)")
        
        if matches:
            print(f"   Possible matches: {', '.join(matches)}")
        else:
            print(f"   No matches found")

# Show naming patterns
print(f"\n5. Naming pattern analysis:")
print("-" * 80)
if csv_files:
    print(f"\n   Sample CSV names:")
    for f in csv_files[:5]:
        print(f"   - {f.name}")

if dict_files:
    print(f"\n   Sample TXT dictionary names:")
    for f in dict_files[:5]:
        print(f"   - {f.name}")

print("\n" + "=" * 80)
print("DEBUG COMPLETE")
print("=" * 80)