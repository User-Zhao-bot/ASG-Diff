"""Command-line entry point for ASG-Diff."""

from runpy import run_module


if __name__ == "__main__":
    run_module("asgdiff.main", run_name="__main__")
