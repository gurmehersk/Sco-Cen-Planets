"""
Usage:

    python run_pipeline.py <input_str> <input_int>

    e.g.

    python run_pipeline.py wotan 87
"""
import sys
from ScoCenPlanets.mypipeline import pipeline

input_str = sys.argv[1]
input_int = int(sys.argv[2])

pipeline(input_str, input_int)