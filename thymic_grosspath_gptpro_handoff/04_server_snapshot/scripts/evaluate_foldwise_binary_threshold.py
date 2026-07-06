#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def infer_score_column(df: pd.DataFrame, requested: str | None, positive_class_name: str | None) -> str:
    if requested:
        if requested not in df.columns:
            raise ValueError('Requested score column not found: {}'.format(requested))
        return requested
    if positive_class_name:
        candidate = 'prob_{}'.format(positive_class_name)
        if candidate in df.columns:
            return candidate
    prob_columns = [column for column in df.columns if column.startswith('prob_')]
    if len(prob_columns) == 1:
        return prob_columns[0]
    if len(prob_columns) == 2:
        return prob_columns[-1]
    raise ValueError('Could not infer score column from columns: {}'.format(list(df.columns)))


def metrics_at_threshold(y_true: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (score >= threshold).astype(int)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    sensitivity = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    precision = safe_div(tp, tp + fp)
    f1 = safe_div(2 * precision * sensitivity, precision + sensitivity)
    accuracy = safe_div(tp + tn, len(y_true))
    balanced_accuracy = (sensitivity + specificity) / 2.0
    return {
        'threshold': float(threshold),
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1': f1,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp,
    }


def summarize_binary(y_true: np.ndarray, score: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    tn = int(((y_true == 0) & (pred == 0)).sum())
    fp = int(((y_true == 0) & (pred == 1)).sum())
    fn = int(((y_true == 1) & (pred == 0)).sum())
    tp = int(((y_true == 1) & (pred == 1)).sum())
    sensitivity = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    precision = safe_div(tp, tp + fp)
    f1 = safe_div(2 * precision * sensitivity, precision + sensitivity)
    out = {
        'accuracy': float(accuracy_score(y_true, pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, pred)),
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1': f1,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tp': tp,
    }
    try:
        out['auc'] = float(roc_auc_score(y_true, score))
    except Exception:
        out['auc'] = float('nan')
    return out


def select_thresholds(val_df: pd.DataFrame, score_column: str, specificity_floor: float, step: float) -> dict[str, dict[str, float]]:
    y_true = (val_df['label_idx'].to_numpy() == 1).astype(int)
    score = val_df[score_column].to_numpy(dtype=float)
    grid = np.round(np.arange(0.0, 1.0001, step), 6)
    rows = [metrics_at_threshold(y_true, score, threshold) for threshold in grid]
    metrics = pd.DataFrame(rows)
    best_bacc = metrics.sort_values(['balanced_accuracy', 'sensitivity', 'specificity', 'f1'], ascending=False).iloc[0].to_dict()
    constrained = metrics[metrics['specificity'] >= specificity_floor]
    if constrained.empty:
        best_constrained = dict(best_bacc)
        best_constrained['fallback_used'] = 1
    else:
        best_constrained = constrained.sort_values(['sensitivity', 'balanced_accuracy', 'specificity', 'f1'], ascending=False).iloc[0].to_dict()
        best_constrained['fallback_used'] = 0
    best_bacc['fallback_used'] = 0
    return {'best_bacc': best_bacc, 'spec_floor': best_constrained}


def apply_threshold(df: pd.DataFrame, score_column: str, threshold: float, method: str, fold_id: int) -> pd.DataFrame:
    out = df.copy()
    out['score'] = out[score_column].astype(float)
    out['pred_idx'] = (out['score'] >= threshold).astype(int)
    out['method'] = method
    out['selected_threshold'] = threshold
    out['fold_id'] = fold_id
    cols = ['case_id', 'label_idx', 'pred_idx', 'score', 'method', 'selected_threshold', 'fold_id']
    keep = [col for col in cols if col in out.columns]
    return out[keep]


def main() -> None:
    parser = argparse.ArgumentParser(description='Fold-wise threshold selection using validation predictions, then OOF evaluation on test predictions.')
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--score-column', default=None)
    parser.add_argument('--positive-class-name', default='high_risk')
    parser.add_argument('--specificity-floor', type=float, default=0.70)
    parser.add_argument('--step', type=float, default=0.01)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_rows = []
    collected = {'best_bacc': [], 'spec_floor': []}
    fold_dirs = sorted([p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith('fold_')])
    if not fold_dirs:
        raise FileNotFoundError('No fold_* directories found under {}'.format(run_dir))

    for fold_dir in fold_dirs:
        fold_id = int(fold_dir.name.split('_')[-1])
        val_df = pd.read_csv(fold_dir / 'val_case_predictions_mean.csv')
        test_df = pd.read_csv(fold_dir / 'test_case_predictions_mean.csv')
        score_column = infer_score_column(val_df, args.score_column, args.positive_class_name)
        selected = select_thresholds(val_df, score_column, args.specificity_floor, args.step)
        for method, row in selected.items():
            fold_rows.append({
                'fold_id': fold_id,
                'method': method,
                'selected_threshold': row['threshold'],
                'val_accuracy': row['accuracy'],
                'val_balanced_accuracy': row['balanced_accuracy'],
                'val_sensitivity': row['sensitivity'],
                'val_specificity': row['specificity'],
                'val_f1': row['f1'],
                'fallback_used': row.get('fallback_used', 0),
            })
            collected[method].append(apply_threshold(test_df, score_column, float(row['threshold']), method, fold_id))

    pd.DataFrame(fold_rows).to_csv(args.output_dir / 'fold_thresholds.csv', index=False)

    metric_rows = []
    for method, frames in collected.items():
        oof = pd.concat(frames, ignore_index=True)
        oof.to_csv(args.output_dir / 'oof_predictions_{}.csv'.format(method), index=False)
        y_true = oof['label_idx'].to_numpy(dtype=int)
        score = oof['score'].to_numpy(dtype=float)
        pred = oof['pred_idx'].to_numpy(dtype=int)
        summary = summarize_binary(y_true, score, pred)
        metric_rows.append({'method': method, **summary})
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.output_dir / 'oof_metrics.csv', index=False)

    lines = ['# Fold-wise Binary Threshold Evaluation', '']
    for _, row in metrics.iterrows():
        lines.extend([
            '## {}'.format(row['method']),
            '',
            '- auc = {:.4f}'.format(row['auc']),
            '- accuracy = {:.4f}'.format(row['accuracy']),
            '- balanced_accuracy = {:.4f}'.format(row['balanced_accuracy']),
            '- sensitivity = {:.4f}'.format(row['sensitivity']),
            '- specificity = {:.4f}'.format(row['specificity']),
            '- f1 = {:.4f}'.format(row['f1']),
            '',
        ])
    (args.output_dir / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print('Saved fold-wise binary threshold evaluation to {}'.format(args.output_dir))


if __name__ == '__main__':
    main()