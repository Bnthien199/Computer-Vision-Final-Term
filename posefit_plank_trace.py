"""Optional read-only plank diagnosis: pose/features/probabilities, no images or GT."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from posefit_pipeline import ROOT, new_output


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


class PlankTrace:
    """Observe actual inference tensors and outputs without changing decisions."""
    def __init__(self, output, classifier, model_path, data_dir, *, input_config=None):
        output=new_output(output,directory=True)
        self.classifier=classifier
        self.original_infer=classifier.infer
        self.scaled=self.raw_model=None
        self.frame=0
        artifacts=[Path(model_path),Path(data_dir)/'scaler.pkl',Path(data_dir)/'label_encoder.pkl',ROOT/'posefit_runtime.py']
        config=dict(classes=classifier.classes.tolist(),confidence=classifier.threshold,
                    smoothing_seconds=classifier.smoothing_seconds,
                    artifacts_sha256={str(p.resolve()):digest(p) for p in artifacts},
                    input='33 ordered normalized MediaPipe x/y/z (99 float32) and saved scaler; no labels supplied',
                    images_saved=False, ground_truth_created=False,
                    input_config=input_config,
                    clock='Original live runtime perf_counter timestamps; logging overhead may affect live cadence',
                    limitations='Softmax/EMA scores are not calibrated correctness. Logs contain pose data; no timer GT. Raw model probabilities/scaled input between calls are cached and timestamped.')
        with (output/'config.json').open('x',encoding='utf8') as f:json.dump(config,f,indent=2)
        self.stream=(output/'trace.jsonl').open('x',encoding='utf8')
        def observed_infer(features):
            result=self.original_infer(features)
            self.scaled=np.asarray(features,dtype=np.float32).copy()
            self.raw_model=result.numpy()[0].copy()
            return result
        self.observed_infer=observed_infer
        classifier.infer=observed_infer

    def write(self,now,landmarks,body,classifier_label,confidence,activity,tracker, *, image_size=None):
        c=self.classifier
        if c.last_inference is None:self.scaled=self.raw_model=None
        raw=None if landmarks is None else np.array([v for lm in landmarks for v in (lm.x,lm.y,lm.z)],dtype=np.float32)
        joint=(body or {}).get('plank')
        reason=('no_pose' if landmarks is None else 'classifier_not_plank' if classifier_label!='plank'
                else 'missing_plank_geometry' if joint is None else 'plank_posture_gate' if not joint['posture']
                else 'active_hold' if tracker.plank.active else 'settling_or_motion')
        row=dict(frame=self.frame,timestamp_seconds=now,raw_xyz_current=raw,
                 processing_width=None if image_size is None else image_size[0],
                 processing_height=None if image_size is None else image_size[1],
                 visibility=None if landmarks is None else [lm.visibility for lm in landmarks],
                 scaled_xyz_last_inference=self.scaled,raw_model_probabilities_last_inference=self.raw_model,
                 last_model_inference_timestamp=c.last_inference,EMA_probabilities=c.probabilities,
                 raw_xyz_finite=None if raw is None else bool(np.isfinite(raw).all()),
                 classifier_label=classifier_label,confidence=confidence,displayed_activity=activity,
                 body=body,plank_gate_reason=reason,pushup_state=tracker.pushup.state,pushup_reps=tracker.pushup.count,
                 plank_timer_state=tracker.plank.state,plank_total_seconds=tracker.plank.total,
                 counter_thresholds=dict(squat_down=tracker.squat.down,squat_up=tracker.squat.up,
                                         pushup_down=tracker.pushup.down,pushup_up=tracker.pushup.up))
        def safe(value):
            if isinstance(value,np.ndarray):return safe(value.tolist())
            if isinstance(value,np.generic):return safe(value.item())
            if isinstance(value,float) and not math.isfinite(value):return None
            if isinstance(value,dict):return {k:safe(v) for k,v in value.items()}
            if isinstance(value,(list,tuple)):return [safe(v) for v in value]
            return value
        self.stream.write(json.dumps(safe(row),allow_nan=False)+'\n');self.stream.flush()
        self.frame+=1

    def close(self):
        self.classifier.infer=self.original_infer
        self.stream.close()
