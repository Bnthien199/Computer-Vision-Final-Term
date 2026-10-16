"""Realtime classification and plank hold timer (no probability smoothing).

Shared implementation: posefit_runtime.py.
"""
from posefit_runtime import (
    Classifier, PlankTimer, RepCounter, Tracker, calculate_angle,
    draw_panel, geometry, main as run_demo,
)


def main():
    run_demo(stage='realtime')


if __name__ == "__main__":
    main()
