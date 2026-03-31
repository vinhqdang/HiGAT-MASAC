"""
FORGE Evaluation Script

Runs evaluation on the trained FORGE checkpoints and plots resilience.
"""

import os
import argparse
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, required=False)
    return parser.parse_args()

def evaluate():
    print("Evaluating FORGE architecture...")
    # TODO: Load models and eval
    pass

if __name__ == "__main__":
    evaluate()
