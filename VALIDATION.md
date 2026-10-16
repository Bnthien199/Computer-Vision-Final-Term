# Release validation — 2026-10-17

- Windows x64, Python 3.11.9; created an isolated virtual environment without
  system site packages. Installed demo dependencies using pip's resolver;
  recorded the resolved versions in `requirements-lock.txt`. No changes to
  the existing project environment. No training performed.
- `pip check`: no broken requirements, both before and after installing
  `requirements-training.txt`.
- `check_demo.py`: PASS. All three SHA256 digests match; 99-feature scaler,
  four-label encoder and existing final MLP load and execute. Installed
  MediaPipe full model/graph processes two blank RGB frames with source timestamps.
- `14_final_posefit.py --help`: PASS.
- `test_posefit_runtime.py`, `test_posefit_plank.py`,
  `test_posefit_continuity.py`: PASS, using synthetic inputs, including real
  bundle loading, rep transitions, occlusion, stream stalls and reset behavior.
- `python -m unittest test_posefit_plank_classification test_posefit_plank_handoff
  test_posefit_video_runtime`: 17 tests, 16 PASS, 1 skipped. The skipped test
  requires a private historical webcam trace. Native MediaPipe graph timestamp
  and reset checks passed.
- `python -m unittest test_posefit_pipeline test_posefit_evaluation`: 9 tests,
  8 PASS, 1 skipped. The skipped test requires private training source data.
  Schema checks now use synthetic fixtures instead of feature CSVs.
- **Webcam not verified:** `check_demo.py --webcam --camera 0` could not open
  the device. Camera 1 also could not open/read a frame. No camera frames were
  saved. Live recognition, OpenCV window rendering, keyboard controls and
  rep/timer quality still require manual verification with an accessible webcam.

These checks establish software/bundle compatibility, not exercise accuracy.
No training/evaluation videos, dataset arrays, personal webcam traces, local
logs or environment directories are included in the release.

The source folder had no Git metadata. Publishing uses a separate clone of
the requested GitHub repository on its existing `main` branch, based on
`08b57c20461bd03b9dceb2164510f99e7719eebc`; its original `.gitignore` is retained
and extended. Original local files remain untouched. The project owner explicitly
confirmed permission to publish the final MLP, scaler and label encoder.
