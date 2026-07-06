#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _metrics_at_threshold(y_true: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (score >= threshold).astype(int)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    sensitivity = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    precision = _safe_div(tp, tp + fp)
    f1 = _safe_div(2 * precision * sensitivity, precision + sensitivity)
    accuracy = _safe_div(tp + tn, len(y_true))
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


def _infer_score_column(df: pd.DataFrame, requested: str | None, positive_class_name: str | None) -> str:
    if requested:
        if requested not in df.columns:
            raise ValueError('Requested score column not found: {}'.format(requested))
        return requested
    if positive_class_name:
        candidate = 'prob_{}'.format(positive_class_name)
        if candidate in df.columns:
            return candidate
    preferred = ['prob_positive', 'positive_prob', 'prob_1', 'score']
    for column in preferred:
        if column in df.columns:
            return column
    prob_columns = [column for column in df.columns if column.startswith('prob_')]
    if len(prob_columns) == 1:
        return prob_columns[0]
    if len(prob_columns) == 2:
        return prob_columns[-1]
    raise ValueError('Could not infer score column from columns: {}'.format(list(df.columns)))


def _write_summary(summary_path: Path, prediction_path: Path, score_column: str, positive_class_name: str | None, specificity_floor: float, default_row: pd.Series, best_balanced_row: pd.Series, best_specificity_row: pd.Series | None) -> None:
    def fmt(row: pd.Series) -> str:
        return 'threshold={:.2f}, accuracy={:.4f}, balanced_accuracy={:.4f}, sensitivity={:.4f}, specificity={:.4f}, precision={:.4f}, f1={:.4f}, tn/fp/fn/tp={}/{}/{}/{}'.format(
            row['threshold'], row['accuracy'], row['balanced_accuracy'], row['sensitivity'], row['specificity'], row['precision'], row['f1'], int(row['tn']), int(row['fp']), int(row['fn']), int(row['tp'])
        )

    positive_desc = positive_class_name if positive_class_name else 'label_idx == 1'
    lines = [
        '# Binary threshold analysis',
        '',
        '- Prediction file: `{}`'.format(prediction_path),
        '- Score column: `{}`'.format(score_column),
        '- Positive class: `{}`'.format(positive_desc),
        '- Threshold sweep: 0.00 to 1.00, step 0.01',
        '',
        '## Key thresholds',
        '',
        '- Default 0.50: {}'.format(fmt(default_row)),
        '- Best balanced accuracy: {}'.format(fmt(best_balanced_row)),
    ]
    if best_specificity_row is not None:
        lines.append('- Best sensitivity with specificity >= {:.2f}: {}'.format(specificity_floor, fmt(best_specificity_row)))
    else:
        lines.append('- Best sensitivity with specificity >= {:.2f}: no threshold met this constraint'.format(specificity_floor))
    lines.append('')
    summary_path.write_text('\n'.join(lines), encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description='Threshold sweep for binary case-level predictions.')
    parser.add_argument('--predictions', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--score-column', default=None)
    parser.add_argument('--positive-class-name', default=None)
    parser.add_argument('--specificity-floor', type=float, default=0.70)
    parser.add_argument('--prefix', default='binary_threshold')
    args = parser.parse_args()
    df = pd.read_csv(args.predictions)
    if 'label_idx' not in df.columns:
        raise ValueError('Expected a label_idx column in the prediction file')
    score_column = _infer_score_column(df, args.score_column, args.positive_class_name)
    y_true = (df['label_idx'].to_numpy() == 1).astype(int)
    score = df[score_column].to_numpy(dtype=float)
    rows = [_metrics_at_threshold(y_true, score, threshold) for threshold in np.round(np.arange(0.0, 1.0001, 0.01), 2)]
    metrics = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / '{}_metrics.csv'.format(args.prefix)
    summary_path = args.output_dir / '{}_summary.md'.format(args.prefix)
    metrics.to_csv(metrics_path, index=False)
    default_idx = (metrics['threshold'] - 0.50).abs().idxmin()
    default_row = metrics.loc[default_idx]
    best_balanced_row = metrics.sort_values(['balanced_accuracy', 'sensitivity', 'specificity', 'f1'], ascending=False).iloc[0]
    constrained = metrics[metrics['specificity'] >= args.specificity_floor]
    best_specificity_row = None
    if not constrained.empty:
        best_specificity_row = constrained.sort_values(['sensitivity', 'balanced_accuracy', 'specificity', 'f1'], ascending=False).iloc[0]
    _write_summary(summary_path, args.predictions, score_column, args.positive_class_name, args.specificity_floor, default_row, best_balanced_row, best_specificity_row)
    print('Wrote {}'.format(metrics_path))
    print('Wrote {}'.format(summary_path))


if __name__ == '__main__':
    main()