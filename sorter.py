#!/usr/bin/env python3
"""
Offline RT-Sort processing script for Maxwell MEA recordings using S3 paths.

Usage:
  python sorter.py [--detection_model <model_identifier_or_path>] <input_s3_path1> <input_s3_path2> ... <output_s3_directory>

The last argument is interpreted as the S3 output directory, and all preceding arguments are S3 input paths.
If --detection_model is set to "maxwell", the detection model will be loaded from "core/spikedetector/detection_models/mea/".
"""

import os
import sys
import argparse
import gc
import shutil
import numpy as np
from utils import s3wrangler as wr
from core.spikedetector.model import ModelSpikeSorter
from core.spikesorter.rt_sort import detect_sequences

def process_dataset(s3_input_path, detection_model, local_input_dir, local_inter_dir, local_output_dir, recording_window_ms=None):
    # Determine the dataset name from the S3 input path.
    dataset_name = os.path.splitext(os.path.basename(s3_input_path))[0]
    local_input_file = os.path.join(local_input_dir, f"{dataset_name}.h5")
    
    # Download the dataset from S3.
    print(f"Downloading {s3_input_path} to {local_input_file}")
    try:
        wr.download(s3_input_path, local_input_file)
    except Exception as e:
        print(f"Error downloading {s3_input_path}: {e}")
        return None

    # Create an intermediate directory for RT-Sort processing.
    inter_path = os.path.join(local_inter_dir, f"{dataset_name}_inter")
    os.makedirs(inter_path, exist_ok=True)
    
    # Run sequence detection using the pretrained detection model.
    print("Running sequence detection...")
    # With return_spikes=True, detect_sequences returns the sorted spike trains directly.
    spike_trains = detect_sequences(local_input_file, inter_path, detection_model, verbose=True, return_spikes=True, recording_window_ms=recording_window_ms)
    
    # Save the sorted spike trains as a compressed NPZ file.
    result_file = os.path.join(local_output_dir, f"{dataset_name}_rtsort.npz")
    np.savez_compressed(result_file, spike_trains=spike_trains)
    print(f"Saved sorted results to {result_file}")

    # Clean up local files to save space.
    del spike_trains
    gc.collect()

    return result_file

def main():
    parser = argparse.ArgumentParser(description="Offline RT-Sort processing script for Maxwell MEA recordings using S3 paths.")
    parser.add_argument("--detection_model", type=str, default="maxwell",
                        help='Identifier for detection model. If "maxwell" is passed, the model is loaded from "core/spikedetector/detection_models/mea/". Otherwise, pass a custom path.')
    parser.add_argument("paths", nargs="+", help="Input S3 paths for datasets and the last argument is the S3 output directory.")
    parser.add_argument("--recording_window_ms", nargs=2, type=int, default=None, help="Optional recording window in ms: start end")
    args = parser.parse_args()
    
    # Ensure at least one input dataset and one output directory are provided.
    if len(args.paths) < 2:
        print("Error: At least one input S3 path and one output S3 directory must be provided.")
        sys.exit(1)
        
    # The last argument is treated as the output S3 directory.
    output_s3_path = args.paths[-1]
    input_s3_paths = args.paths[:-1]
    
    # Setup local directories for inputs, intermediate files, and outputs.
    local_input_dir = "/tmp/local_inputs"
    local_inter_dir = "/tmp/intermediate"
    local_output_dir = "/tmp/local_outputs"
    os.makedirs(local_input_dir, exist_ok=True)
    os.makedirs(local_inter_dir, exist_ok=True)
    os.makedirs(local_output_dir, exist_ok=True)
    
    # Determine detection model path based on the argument.
    if args.detection_model.lower() == "maxwell":
        detection_model_path = os.path.join("core", "spikedetector", "detection_models", "mea")
        print("Loading Maxwell detection model from default location:", detection_model_path)
    else:
        detection_model_path = args.detection_model
        print("Loading detection model from custom path:", detection_model_path)
    
    # Load the pretrained detection model.
    try:
        detection_model = ModelSpikeSorter.load(detection_model_path)
    except Exception as e:
        print("Error loading detection model:", e)
        sys.exit(1)
    
    # Parse the optional recording window parameter.
    recording_window_ms = None
    if args.recording_window_ms:
        recording_window_ms = tuple(args.recording_window_ms)
    
    # Process each input dataset.
    for s3_input in input_s3_paths:
        print(f"\nProcessing dataset from {s3_input}")
        result_file = process_dataset(s3_input, detection_model, local_input_dir, local_inter_dir, local_output_dir, recording_window_ms)
        if result_file:
            s3_dest = os.path.join(output_s3_path, os.path.basename(result_file))
            print(f"Uploading {result_file} to {s3_dest}")
            try:
                wr.upload(result_file, s3_dest)
            except Exception as e:
                print(f"Error uploading {result_file}: {e}")
    
    print("All datasets processed successfully.")

if __name__ == '__main__':
    main()
