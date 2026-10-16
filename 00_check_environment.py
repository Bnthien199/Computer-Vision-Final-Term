import sys
from posefit_console import configure_console
configure_console()

print("=" * 70)
print("POSEFIT - ENVIRONMENT CHECK")
print("=" * 70)

print("\nPython:")
print(sys.version)

packages = [
    "numpy",
    "pandas",
    "cv2",
    "mediapipe",
    "sklearn",
    "matplotlib",
    "tensorflow",
    "joblib",
    "tqdm",
    "seaborn"
]

print("\nPACKAGE CHECK")
print("-" * 70)

failures = []
for package in packages:
    try:
        module = __import__(package)

        version = getattr(module, "__version__", "unknown")

        print(f"OK  {package:15s} {version}")
        if package == 'mediapipe' and not hasattr(module, 'solutions'):
            print('FAIL: this project requires MediaPipe legacy mp.solutions.pose')
            failures.append('mediapipe: missing mp.solutions.pose')

    except Exception as e:
        failures.append(package)
        print(f"FAIL {package:15s}")
        print("    ", e)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
sys.exit(1 if failures else 0)
