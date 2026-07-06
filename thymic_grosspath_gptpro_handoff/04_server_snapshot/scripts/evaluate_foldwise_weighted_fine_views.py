#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


CLASSES = ['A', 'AB', 'B1', 'B2', 'B3']


def prob_columns(df: pd.DataFrame) -> list[str]:
    cols = [column for column in df.columns if column.startswith('prob_')]
    if not cols:
        raise ValueError('No probability columns found.')
    return cols


def combine(df_whole: pd.DataFrame, df_crop: pd.DataFrame, weight: float, cols: list[str]) -> pd.DataFrame:
    key = ['case_id', 'label_idx']
    if 'fold_id' in df_whole.columns and 'fold_id' in df_crop.columns:
        key.append('fold_id')
    left = df_whole[key].copy()
    for column in cols:
        left[column] = weight * df_whole[column].to_numpy(dtype=float) + (1.0 - weight) * df_crop[column].to_numpy(dtype=float)
    left['pred_idx'] = left[cols].to_numpy().argmax(axis=1)
    return left


def summarize(frame: pd.DataFrame, cols: list[str]) -> dict[str, float]:
    y_true = frame['label_idx'].to_numpy(dtype=int)
    y_pred = frame['pred_idx'].to_numpy(dtype=int)
    prob = frame[cols].to_numpy(dtype=float)
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'macro_precision': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'macro_recall': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'macro_auc': float(roc_auc_score(y_true, prob, multi_class='ovr', average='macro')),
    }


def select_weight(val_whole: pd.DataFrame, val_crop: pd.DataFrame, cols: list[str], step: float, metric: str) -> dict[str, float]:
    rows = []
    best = None
    for weight in np.round(np.arange(0.0, 1.0001, step), 6):
        frame = combine(val_whole, val_crop, float(weight), cols)
        metrics = summarize(frame, cols)
        row = {'whole_weight': float(weight), 'crop_weight': float(1.0 - weight), **metrics}
        rows.append(row)
        if best is None:
            best = row
            continue
        better = False
        if row[metric] > best[metric]:
            better = True
        elif row[metric] == best[metric] and row['macro_auc'] > best['macro_auc']:
            better = True
        elif row[metric] == best[metric] and row['macro_auc'] == best['macro_auc'] and row['accuracy'] > best['accuracy']:
            better = True
        if better:
            best = row
    return {'best': best, 'grid': pd.DataFrame(rows)}


def per_class_recall(frame: pd.DataFrame, method: str) -> pd.DataFrame:
    rows = []
    for idx, cls in enumerate(CLASSES):
        sub = frame[frame['label_idx'] == idx]
        recall = float((sub['pred_idx'] == sub['label_idx']).mean()) if len(sub) else float('nan')
        rows.append({'method': method, 'class_name': cls, 'n': int(len(sub)), 'recall': recall})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description='Fold-wise validation-selected whole/crop weighting for Task B stage1 fine predictions.')
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--grid-step', type=float, default=0.05)
    parser.add_argument('--metric', default='macro_f1', choices=('macro_f1', 'macro_auc', 'accuracy', 'balanced_accuracy'))
    parser.add_argument('--fixed-whole-weight', type=float, default=0.55)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_rows = []
    fixed_frames = []
    selected_frames = []
    cols = None
    fold_dirs = sorted([p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith('fold_')])
    if not fold_dirs:
        raise FileNotFoundError('No fold_* directories found under {}'.format(run_dir))

    for fold_dir in fold_dirs:
        fold_id = int(fold_dir.name.split('_')[-1])
        val_whole = pd.read_csv(fold_dir / 'val_stage1_fine_whole_case_predictions_mean.csv').sort_values(['case_id']).reset_index(drop=True)
        val_crop = pd.read_csv(fold_dir / 'val_stage1_fine_crop_case_predictions_mean.csv').sort_values(['case_id']).reset_index(drop=True)
        test_whole = pd.read_csv(fold_dir / 'test_stage1_fine_whole_case_predictions_mean.csv').sort_values(['case_id']).reset_index(drop=True)
        test_crop = pd.read_csv(fold_dir / 'test_stage1_fine_crop_case_predictions_mean.csv').sort_values(['case_id']).reset_index(drop=True)
        cols = prob_columns(val_whole)
        selected = select_weight(val_whole, val_crop, cols, args.grid_step, args.metric)
        best = selected['best']
        selected['grid'].to_csv(args.output_dir / 'fold_{}_val_weight_grid.csv'.format(fold_id), index=False)
        fold_rows.append({'fold_id': fold_id, **best})

        fixed_frame = combine(test_whole, test_crop, float(args.fixed_whole_weight), cols)
        fixed_frame['fold_id'] = fold_id
        fixed_frame['method'] = 'fixed_{:.2f}'.format(args.fixed_whole_weight)
        fixed_frame['whole_weight'] = float(args.fixed_whole_weight)
        fixed_frames.append(fixed_frame)

        selected_frame = combine(test_whole, test_crop, float(best['whole_weight']), cols)
        selected_frame['fold_id'] = fold_id
        selected_frame['method'] = 'foldwise_val_selected'
        selected_frame['whole_weight'] = float(best['whole_weight'])
        selected_frames.append(selected_frame)

    pd.DataFrame(fold_rows).to_csv(args.output_dir / 'fold_selected_weights.csv', index=False)

    metric_rows = []
    recall_frames = []
    for method, frames in [('fixed_{:.2f}'.format(args.fixed_whole_weight), fixed_frames), ('foldwise_val_selected', selected_frames)]:
        oof = pd.concat(frames, ignore_index=True)
        oof.to_csv(args.output_dir / 'oof_predictions_{}.csv'.format(method), index=False)
        metrics = summarize(oof, cols)
        metric_rows.append({'method': method, **metrics})
        recall_frames.append(per_class_recall(oof, method))
    pd.DataFrame(metric_rows).to_csv(args.output_dir / 'oof_metrics.csv', index=False)
    pd.concat(recall_frames, ignore_index=True).to_csv(args.output_dir / 'per_class_recall.csv', index=False)

    lines = ['# Fold-wise Weighted Fine View Evaluation', '']
    for row in metric_rows:
        lines.extend([
            '## {}'.format(row['method']),
            '',
            '- accuracy = {:.4f}'.format(row['accuracy']),
            '- balanced_accuracy = {:.4f}'.format(row['balanced_accuracy']),
            '- macro_f1 = {:.4f}'.format(row['macro_f1']),
            '- macro_auc = {:.4f}'.format(row['macro_auc']),
            '',
        ])
    (args.output_dir / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print('Saved fold-wise weighted fine-view evaluation to {}'.format(args.output_dir))


if __name__ == '__main__':
    main()