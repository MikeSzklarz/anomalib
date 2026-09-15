"""Compatibility shim so `python train.py ...` (used by run_slurm.sh /
submit_sweep.sh) keeps working after the training script was split into the
train/ package. See train/cli.py for the actual argument parsing and pipeline.
"""

from train.cli import main

if __name__ == "__main__":
    main()
