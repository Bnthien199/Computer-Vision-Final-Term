"""Plank pipeline/UI boundaries; synthetic fixtures are not accuracy labels."""
from types import SimpleNamespace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from posefit_runtime import Classifier, Tracker, geometry, draw_panel


def classifier_fixture(probabilities, smoothing=.2):
    c=Classifier.__new__(Classifier)
    c.classes=np.array(['plank','pushup','squat','standing'])
    c.threshold=.65;c.smoothing_seconds=smoothing
    c.scaler=SimpleNamespace(transform=lambda x:x)
    values=iter(probabilities)
    def infer(x):
        result=np.array([next(values)],dtype=np.float32)
        return SimpleNamespace(numpy=lambda:result)
    c.infer=infer
    c.reset()
    return c


def forearm_landmarks():
    lm=[SimpleNamespace(x=.5,y=.5,z=0.,visibility=0.) for _ in range(33)]
    for i,xy in {11:(.25,.35),13:(.25,.55),15:(.45,.55),23:(.5,.37),25:(.75,.39)}.items():
        lm[i]=SimpleNamespace(x=xy[0],y=xy[1],z=0.,visibility=.95)
    return lm


class PlankClassificationTests(unittest.TestCase):
    def test_plank_with_visible_wrists_never_starts_pushup(self):
        landmarks=forearm_landmarks()
        body=geometry(landmarks,640,360)
        self.assertTrue(body['plank']['posture'])
        self.assertTrue(body['pushup']['posture'])
        self.assertAlmostEqual(body['pushup']['angle'],90)
        classifier=classifier_fixture([[.72,.28,0.,0.]]*31)
        tracker=Tracker()
        for i in range(31):
            label,confidence=classifier.update(landmarks,i/10)
            self.assertEqual(label,'plank')
            tracker.update(body,label,i/10)
        self.assertEqual(tracker.pushup.count,0)
        self.assertNotEqual(tracker.pushup.state,'DOWN')
        self.assertAlmostEqual(tracker.plank.total,2.)

    def test_wrong_72_percent_prediction_is_not_geometry_relabelled(self):
        landmarks=forearm_landmarks();body=geometry(landmarks,640,360)
        c=classifier_fixture([[.28,.72,0.,0.]]*31)
        tracker=Tracker()
        for i in range(31):
            label,confidence=c.update(landmarks,i/10)
            self.assertEqual(label,'pushup')
            self.assertAlmostEqual(confidence,.72,places=6)
            tracker.update(body,label,i/10)
        self.assertEqual(tracker.plank.total,0.)
        self.assertEqual(tracker.pushup.count,0, 'An initially static bent arm is not a full push-up cycle')

    def test_ema_carryover_is_distinct_from_current_mlp_output(self):
        c=classifier_fixture([[0.,1.,0.,0.],[.72,.28,0.,0.],[.72,.28,0.,0.]])
        lm=forearm_landmarks();c.update(lm,0)
        label,confidence=c.update(lm,.1)
        self.assertEqual(label,'pushup')
        self.assertAlmostEqual(confidence,.716702,places=5)
        self.assertEqual(c.update(lm,.5)[0],'plank')

    def test_99_xyz_order_and_saved_transform_path(self):
        c=classifier_fixture([[.72,.28,0.,0.]])
        lm=[SimpleNamespace(x=i/100,y=i/100+.1,z=-i/100) for i in range(33)]
        received={}
        def transform(raw):received['raw']=raw.copy();return raw+3
        def infer(scaled):received['scaled']=scaled.copy();return SimpleNamespace(numpy=lambda:np.array([[.72,.28,0.,0.]]))
        c.scaler=SimpleNamespace(transform=transform);c.infer=infer
        self.assertEqual(c.update(lm,0)[0],'plank')
        expected=np.array([[v for p in lm for v in (p.x,p.y,p.z)]],dtype=np.float32)
        np.testing.assert_array_equal(received['raw'],expected)
        np.testing.assert_array_equal(received['scaled'],expected+3)

    def test_panel_confidence_keeps_classifier_label_during_down_state(self):
        tracker=Tracker()
        def body(angle):return dict(pushup=dict(angle=angle,posture=True,side=11),
            plank=dict(angle=175,posture=True,side=11,signature=np.zeros(8)))
        for t,angle in [(0,170),(.11,170),(.22,170),(.33,80),(.44,80),(.55,80)]:
            tracker.update(body(angle),'pushup',t)
        displayed=tracker.update(body(80),'plank',.66)
        self.assertEqual(displayed,'pushup')
        texts=[]
        with patch('posefit_runtime.cv2.putText',side_effect=lambda frame,text,*a,**k:texts.append(text)):
            draw_panel(np.zeros((360,640,3),np.uint8),displayed,.72,tracker,30,classifier_label='plank')
        self.assertIn('plank  |  Confidence: 72%',texts)
        self.assertNotIn('pushup  |  Confidence: 72%',texts)
        self.assertEqual(tracker.plank.total,0., 'UI correction must not override timer gates')

    def test_optional_trace_preserves_outputs_and_clears_cached_prediction_on_reset(self):
        from posefit_plank_trace import PlankTrace
        from posefit_pipeline import ROOT
        run=ROOT/'runs'
        run.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=run,prefix='trace_unit_') as temporary:
            root=Path(temporary).resolve()
            self.assertTrue(root.is_relative_to(run.resolve()))
            model=root/'synthetic_model_fixture.txt';model.write_text('unit fixture, not a trained model')
            for name in ('scaler.pkl','label_encoder.pkl'):(root/name).write_bytes(b'unit fixture')
            c=classifier_fixture([[.72,.28,0.,0.]])
            original=c.infer
            trace=PlankTrace(root/'new_trace',c,model,root)
            landmarks=forearm_landmarks();body=geometry(landmarks,640,360);tracker=Tracker()
            label,confidence=c.update(landmarks,0)
            activity=tracker.update(body,label,0)
            trace.write(0,landmarks,body,label,confidence,activity,tracker,image_size=(640,360))
            c.reset();trace.write(.1,None,None,'No Pose',0.,'No Pose',tracker)
            trace.close()
            self.assertIs(c.infer,original)
            rows=[json.loads(line) for line in (root/'new_trace/trace.jsonl').read_text().splitlines()]
            self.assertEqual(rows[0]['classifier_label'],'plank')
            self.assertEqual(len(rows[0]['raw_xyz_current']),99)
            self.assertEqual(rows[0]['processing_width'],640)
            self.assertEqual(rows[0]['processing_height'],360)
            np.testing.assert_allclose(rows[0]['raw_model_probabilities_last_inference'],[.72,.28,0.,0.])
            self.assertEqual(rows[0]['plank_gate_reason'],'settling_or_motion')
            self.assertIsNone(rows[1]['raw_model_probabilities_last_inference'])
            self.assertEqual(rows[1]['plank_gate_reason'],'no_pose')
            self.assertFalse(list(root.glob('**/*.png')))

    def test_trace_rejects_existing_output_without_replacing_files(self):
        from posefit_plank_trace import PlankTrace
        from posefit_pipeline import ROOT
        (ROOT/'runs').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'runs',prefix='trace_unit_') as temporary:
            root=Path(temporary).resolve()
            self.assertTrue(root.is_relative_to((ROOT/'runs').resolve()))
            marker=root/'marker.txt';marker.write_text('keep')
            c=classifier_fixture([[.72,.28,0.,0.]])
            with self.assertRaises(FileExistsError):PlankTrace(root,c,root/'unused',root)
            self.assertEqual(marker.read_text(),'keep')

    def test_trace_records_rejected_nonfinite_features_as_missing(self):
        from posefit_plank_trace import PlankTrace
        from posefit_pipeline import ROOT
        (ROOT/'runs').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'runs',prefix='trace_unit_') as temporary:
            root=Path(temporary).resolve()
            self.assertTrue(root.is_relative_to((ROOT/'runs').resolve()))
            model=root/'synthetic_fixture.txt';model.write_text('synthetic fixture')
            for name in ('scaler.pkl','label_encoder.pkl'):(root/name).write_text('synthetic fixture')
            c=classifier_fixture([[.72,.28,0.,0.]])
            trace=PlankTrace(root/'new_trace',c,model,root)
            lm=forearm_landmarks();lm[0].x=float('nan')
            label,confidence=c.update(lm,0)
            self.assertEqual(label,'Unknown')
            trace.write(0,lm,None,label,confidence,label,Tracker())
            trace.close()
            row=json.loads((root/'new_trace/trace.jsonl').read_text())
            self.assertFalse(row['raw_xyz_finite'])
            self.assertIsNone(row['raw_xyz_current'][0])


if __name__=='__main__':unittest.main()
