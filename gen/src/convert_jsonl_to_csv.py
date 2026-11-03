#!/usr/bin/env python3
"""
convert_jsonl_to_csv.py

A utility script to convert JSONL (JSON Lines) files to CSV format.
Specially designed to handle multi-line strings, markdown, and varying keys.

Usage:
  python convert_jsonl_to_csv.py [-i INPUT_PATH] [-o OUTPUT_PATH]

Defaults:
  Input:  personas_output_qwen_test/personas.jsonl
  Output: personas_output_qwen_test/personas.csv
"""

import os
import sys
import json
import csv
import argparse


def convert_jsonl_to_csv(input_path: str, output_path: str):
    """
    Reads a JSONL file, collects all unique fields, and writes them to a CSV file.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    print(f"Reading '{input_path}'...")
    
    # First pass: collect all unique keys to build the header
    headers = []
    headers_set = set()
    data = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                data.append(record)
                # Keep keys in order of appearance
                for key in record.keys():
                    if key not in headers_set:
                        headers_set.add(key)
                        headers.append(key)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping line {line_num} due to JSON decode error: {e}", file=sys.stderr)

    if not data:
        print(f"Warning: No valid data found in '{input_path}'. Output CSV will be empty or not created.", file=sys.stderr)
        return

    print(f"Writing {len(data)} rows with fields {headers} to '{output_path}'...")
    
    # Create target directory if it doesn't exist
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        # Use DictWriter to map dictionary keys to headers. Missing keys will be written as empty.
        writer = csv.DictWriter(f, fieldnames=headers, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for row in data:
            writer.writerow(row)
            
    print(f"Success! Converted '{input_path}' -> '{output_path}'")


def main():
    parser = argparse.ArgumentParser(description="Convert a JSONL file to a CSV file.")
    parser.add_argument(
        '-i', '--input',
        help="Path to the input JSONL file. Defaults to 'personas_output_qwen_test/personas.jsonl' if it exists."
    )
    parser.add_argument(
        '-o', '--output',
        help="Path to the output CSV file. Defaults to the same directory and base name as input, but with .csv extension."
    )
    
    args = parser.parse_args()
    
    # Determine default paths
    default_input = "personas_output_qwen_test/personas.jsonl"
    
    if args.input:
        input_path = args.input
    else:
        if os.path.exists(default_input):
            input_path = default_input
        else:
            # Fall back to asking or showing help
            print(f"Default input file '{default_input}' not found.", file=sys.stderr)
            parser.print_help()
            sys.exit(1)
            
    if args.output:
        output_path = args.output
    else:
        # Replace .jsonl extension with .csv
        if input_path.lower().endswith('.jsonl'):
            output_path = input_path[:-6] + '.csv'
        elif input_path.lower().endswith('.json'):
            output_path = input_path[:-5] + '.csv'
        else:
            output_path = input_path + '.csv'
            
    convert_jsonl_to_csv(input_path, output_path)


if __name__ == '__main__':
    main()
