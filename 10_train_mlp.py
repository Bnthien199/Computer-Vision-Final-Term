"""Train the existing 99->128->64->4 MLP; select by validation loss."""
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import argparse
from pathlib import Path
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from posefit_pipeline import ROOT, new_output
from posefit_training import load_bundle, save_evaluation, save_run_info


def build_model():
    model = Sequential([
        tf.keras.Input(shape=(99,)),
        Dense(128, activation='relu'), BatchNormalization(), Dropout(.3),
        Dense(64, activation='relu'), BatchNormalization(), Dropout(.3),
        Dense(4, activation='softmax')])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.001),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'training_data')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/mlp')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error('epochs must be positive')
    data, encoder = load_bundle(args.data_dir)
    output = new_output(args.output, directory=True)
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()
    model = build_model()
    best_path = output / 'mlp_posefit_best.keras'
    callbacks = [
        ModelCheckpoint(str(best_path), monitor='val_loss', mode='min', save_best_only=True),
        EarlyStopping(monitor='val_loss', mode='min', patience=20, restore_best_weights=True)]
    x, y = data['train']
    history = model.fit(x, y, validation_data=data['val'], epochs=args.epochs,
                        batch_size=32, callbacks=callbacks, verbose=1)
    # Explicitly use the selected checkpoint, including when max epochs is reached.
    model = tf.keras.models.load_model(best_path, compile=False)
    model.save(output / 'mlp_posefit_final.keras')
    for split in ('val', 'test'):
        x, y = data[split]
        pred = model.predict(x, verbose=0).argmax(axis=1)
        save_evaluation(y, pred, encoder.classes_, output, split)
    (output / 'history.json').write_text(json.dumps(history.history, indent=2), encoding='utf-8')
    save_run_info(output, args.data_dir, dict(model='MLP 99-128-64-4', seed=args.seed,
                  epochs_requested=args.epochs, epochs_run=len(history.epoch),
                  best_epoch=int(np.argmin(history.history['val_loss'])) + 1,
                  selection='minimum val_loss', loss='sparse_categorical_crossentropy',
                  optimizer='Adam', learning_rate=.001, batch_size=32))
    print(f'Saved: {output}')


if __name__ == '__main__':
    main()
