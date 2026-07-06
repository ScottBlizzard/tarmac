#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def _prob_columns(df: pd.DataFrame) -> list[str]:
    columns = [column for column in df.columns if column.startswith('prob_')]
    if not columns:
        raise ValueError('No probability columns found.')
    return columns


def _combine(whole: pd.DataFrame, crop: pd.DataFrame, weight: float, prob_columns: list[str]) -> pd.DataFrame:
    out = whole[['case_id', 'label_idx', 'fold_id']].copy()
    out['n_images_whole'] = whole.get('n_images', pd.Series([None] * len(whole)))
    out['n_images_crop'] = crop.get('n_images', pd.Series([None] * len(crop)))
    for column in prob_columns:
        out[column] = weight * whole[column].to_numpy(dtype=float) + (1.0 - weight) * crop[column].to_numpy(dtype=float)
    out['pred_idx'] = out[prob_columns].to_numpy().argmax(axis=1)
    return out


def _summarize(frame: pd.DataFrame, prob_columns: list[str]) -> dict[str, float]:
    y_true = frame['label_idx'].to_numpy(dtype=int)
    y_pred = frame['pred_idx'].to_numpy(dtype=int)
    prob = frame[prob_columns].to_numpy(dtype=float)
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'macro_precision': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'macro_recall': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'macro_auc': float(roc_auc_score(y_true, prob, multi_class='ovr', average='macro')),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate weighted fusion of stage1 fine whole/crop case predictions.')
    parser.add_argument('--whole-predictions', required=True, type=Path)
    parser.add_argument('--crop-predictions', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--whole-weight', type=float, default=0.55)
    parser.add_argument('--grid-step', type=float, default=0.05)
    parser.add_argument('--metric', default='macro_f1', choices=('macro_f1', 'macro_auc', 'accuracy', 'balanced_accuracy'))
    args = parser.parse_args()
    whole = pd.read_csv(args.whole_predictions).sort_values(['case_id', 'fold_id']).reset_index(drop=True)
    crop = pd.read_csv(args.crop_predictions).sort_values(['case_id', 'fold_id']).reset_index(drop=True)
    key_cols = ['case_id', 'label_idx', 'fold_id']
    if whole[key_cols].to_dict('records') != crop[key_cols].to_dict('records'):
        raise ValueError('Whole and crop prediction files do not align on case_id/label_idx/fold_id.')
    prob_columns = _prob_columns(whole)
    missing = [column for column in prob_columns if column not in crop.columns]
    if missing:
        raise ValueError('Missing probability columns in crop predictions: {}'.format(missing))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sweep_rows = []
    best_weight = None
    best_metrics = None
    best_frame = None
    weights = np.round(np.arange(0.0, 1.0001, args.grid_step), 6)
    for weight in weights:
        frame = _combine(whole, crop, float(weight), prob_columns)
        metrics = _summarize(frame, prob_columns)
        row = {'whole_weight': float(weight), 'crop_weight': float(1.0 - weight), **metrics}
        sweep_rows.append(row)
        if best_metrics is None or row[args.metric] > best_metrics[args.metric]:
            best_weight = float(weight)
            best_metrics = row
            best_frame = frame
    sweep = pd.DataFrame(sweep_rows).sort_values(['whole_weight'])
    sweep.to_csv(args.output_dir / 'weight_sweep.csv', index=False)
    fixed_frame = _combine(whole, crop, float(args.whole_weight), prob_columns)
    fixed_metrics = _summarize(fixed_frame, prob_columns)
    fixed_frame.to_csv(args.output_dir / 'fixed_weight_predictions.csv', index=False)
    pd.DataFrame([{'whole_weight': float(args.whole_weight), 'crop_weight': float(1.0 - args.whole_weight), **fixed_metrics}]).to_csv(args.output_dir / 'fixed_weight_metrics.csv', index=False)
    if best_frame is not None and best_metrics is not None:
        best_frame.to_csv(args.output_dir / 'best_weight_predictions.csv', index=False)
        pd.DataFrame([best_metrics]).to_csv(args.output_dir / 'best_weight_metrics.csv', index=False)
    lines = [
        '# Weighted Fine View Evaluation',
        '',
        '- Whole predictions: `{}`'.format(args.whole_predictions),
        '- Crop predictions: `{}`'.format(args.crop_predictions),
        '- Selection metric: `{}`'.format(args.metric),
        '- Grid step: `{}`'.format(args.grid_step),
        '',
        '## Fixed weight result',
        '',
        '- whole/crop = {:.2f}/{:.2f}'.format(args.whole_weight, 1.0 - args.whole_weight),
        '- accuracy = {:.4f}'.format(fixed_metrics['accuracy']),
        '- balanced_accuracy = {:.4f}'.format(fixed_metrics['balanced_accuracy']),
        '- macro_f1 = {:.4f}'.format(fixed_metrics['macro_f1']),
        '- macro_auc = {:.4f}'.format(fixed_metrics['macro_auc']),
        '',
        '## Best grid weight',
        '',
        '- whole/crop = {:.2f}/{:.2f}'.format(best_weight, 1.0 - best_weight),
        '- accuracy = {:.4f}'.format(best_metrics['accuracy']),
        '- balanced_accuracy = {:.4f}'.format(best_metrics['balanced_accuracy']),
        '- macro_f1 = {:.4f}'.format(best_metrics['macro_f1']),
        '- macro_auc = {:.4f}'.format(best_metrics['macro_auc']),
        '',
    ]
    (args.output_dir / 'weight_sweep_summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print('Saved weighted fine-view evaluation to {}'.format(args.output_dir))


if __name__ == '__main__':
    main()