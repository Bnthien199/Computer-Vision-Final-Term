"""Regression tests for data integrity, provenance and sampling; no source writes."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import cv2
import numpy as np
import pandas as pd
from posefit_pipeline import ROOT, FEATURE_COLUMNS, new_output, validate_features


def synthetic_features():
    rows = []
    for split, subject in (('train', 1), ('val', 4), ('test', 5)):
        for label in ('plank', 'pushup', 'squat', 'standing'):
            rows.append(dict.fromkeys(FEATURE_COLUMNS, .5) |
                        dict(label=label, split=split, subject=subject if label == 'standing' else np.nan))
    return pd.DataFrame(rows)


def load_script(filename):
    spec = importlib.util.spec_from_file_location(filename[:-3], ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT / 'runs').mkdir(exist_ok=True)

    def test_environment_failure_exit_code(self):
        import builtins
        import io
        import runpy
        original_import = builtins.__import__
        def checked_import(name, *args, **kwargs):
            if name == 'seaborn':
                raise ImportError('deliberately unavailable dependency in regression test')
            return original_import(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=checked_import), patch('sys.stdout', new=io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                runpy.run_path(str(ROOT / '00_check_environment.py'), run_name='__main__')
        self.assertEqual(caught.exception.code, 1)

    def test_schema(self):
        df = synthetic_features()
        original = validate_features(df)
        shuffled = df[list(reversed(df.columns))]
        np.testing.assert_array_equal(validate_features(shuffled), original)
        only_test = df.loc[df.split.eq('test')]
        np.testing.assert_array_equal(validate_features(only_test, require_all_splits=False), original[df.split.eq('test')])
        with self.assertRaises(ValueError):
            validate_features(only_test)
        for mutate in (
            lambda d: d.drop(columns=['f_98']),
            lambda d: d.assign(f_0=np.inf),
            lambda d: d.assign(split='train'),
            lambda d: d.assign(filepath='same_clip.mp4'),
            lambda d: d.assign(label='flank')):
            with self.assertRaises(ValueError):
                validate_features(mutate(df.copy()))

    def test_output_protection(self):
        with self.assertRaises(ValueError):
            new_output(ROOT / 'metadata.csv')
        with tempfile.TemporaryDirectory(dir=ROOT / 'runs') as directory:
            with self.assertRaises(FileExistsError):
                new_output(directory, directory=True)
            target = new_output(Path(directory) / 'fresh.csv')
            target.write_text('test', encoding='utf-8')
            with self.assertRaises(FileExistsError):
                new_output(target)

    @unittest.skipUnless((ROOT / 'archive/verified_data').is_dir() and (ROOT / 'standing_raw').is_dir(),
                         'Optional historical split check requires private datasets')
    def test_split_reproducibility(self):
        module = load_script('04_create_metadata.py')
        # Main-only side effects: reset RNG as main does, so repeated runs match.
        module.random.seed(42)
        a = module.create_gym_metadata() + module.create_standing_metadata()
        module.random.seed(42)
        b = module.create_gym_metadata() + module.create_standing_metadata()
        self.assertEqual(a, b)
        standing = pd.DataFrame(a).query('label == "standing"')
        self.assertTrue((standing.groupby('subject').split.nunique() == 1).all())

    def test_frame_indices_and_decode_failures(self):
        module = load_script('05_extract_landmarks.py')
        class Capture:
            def __init__(self):
                self.index = None
                self.released = False
            def isOpened(self): return True
            def get(self, key):
                if key == cv2.CAP_PROP_FRAME_COUNT: return 3
                if key == cv2.CAP_PROP_POS_MSEC: return self.index * 1000 / 30
                if key == cv2.CAP_PROP_POS_FRAMES: return self.index + 1
                return 30
            def set(self, key, value): self.index = int(value)
            def read(self): return (False, None) if self.index == 1 else (True, np.zeros((4, 4, 3), dtype=np.uint8))
            def release(self): self.released = True
        cap = Capture()
        with patch.object(module.cv2, 'VideoCapture', return_value=cap):
            rows = module.extract_video_frames('unused.mp4')
        self.assertEqual([r[0] for r in rows], [0, 1, 2])
        self.assertIsNone(rows[1][2])
        self.assertAlmostEqual(rows[2][1], 2 / 30)
        self.assertTrue(cap.released)
        self.assertIsNone(module.read_image_unicode(ROOT / 'does_not_exist.png'))

    def test_invalid_count_and_seek_release_capture(self):
        module = load_script('05_extract_landmarks.py')
        from unittest.mock import MagicMock
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.return_value = np.nan
        with patch.object(module.cv2, 'VideoCapture', return_value=cap):
            self.assertEqual(module.extract_video_frames('unused'), [])
        cap.release.assert_called_once()
        cap.reset_mock()
        cap.get.return_value = 3
        cap.set.return_value = False
        with patch.object(module.cv2, 'VideoCapture', return_value=cap):
            with self.assertRaisesRegex(RuntimeError, 'seek'):
                module.extract_video_frames('unused')
        cap.release.assert_called_once()

    def test_provenance_and_subject_leakage(self):
        df = synthetic_features().copy()
        df['filepath'] = [f'clip_{i}.mp4' for i in range(len(df))]
        train = df.index[df.split.eq('train')][0]
        test = df.index[df.split.eq('test')][0]
        df.loc[train, 'filepath'] = 'runs/same.mp4'
        df.loc[test, 'filepath'] = 'runs/../runs/same.mp4'
        with self.assertRaisesRegex(ValueError, 'shared across splits'):
            validate_features(df)
        df = synthetic_features()
        df.loc[df.label.eq('standing'), 'subject'] = 1
        with self.assertRaisesRegex(ValueError, 'subject shared'):
            validate_features(df)


if __name__ == '__main__':
    (ROOT / 'runs').mkdir(exist_ok=True)
    unittest.main()
