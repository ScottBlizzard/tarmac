from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
from typing import Any

import pandas as pd
import torch

from thymic_baseline.train import build_dataloader, load_available_folds, resolve_device, set_seed, write_json
from thymic_surimage.strict_train import (
    FINE_CLASS_NAMES,
    StrictOrdinalSuRImageClassifier,
    StrictSuRImageClassifier,
    build_paired_dataloader,
    evaluate_stage1,
    fine_output_dim,
    metric_from_results,
    prepare_task4_images,
    prepare_task4_paired_images,
    summarize_prediction_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Select best Task-B stage1 view per fold using validation metrics.')
    parser.add_argument('--stage1-dir', required=True)
    parser.add_argument('--registry-csv', required=True)
    parser.add_argument('--split-csv', required=True)
    parser.add_argument('--images-root', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--manifest-csv', default=None)
    parser.add_argument('--manifest-view-set', choices=('whole_only', 'merged_crop_only', 'whole_plus_merged_crop'), default='whole_only')
    parser.add_argument('--fine-crop-source', choices=('cam', 'merged_manifest'), default='cam')
    parser.add_argument('--coarse-ckpt-name', default='stage1_coarse_best_auc.pt')
    parser.add_argument('--fine-ckpt-name', default='stage1_fine_best_joint_score.pt')
    parser.add_argument('--backbone', default='seresnext50_32x4d')
    parser.add_argument('--fine-loss', default='ce', choices=('ce', 'condor'))
    parser.add_argument('--image-size', type=int, default=352)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--fold', default='all')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--seed', type=int, default=108)
    parser.add_argument('--max-eval-batches', type=int, default=None)
    parser.add_argument('--selection-metric', default='macro_f1', choices=('macro_f1', 'macro_auc'))
    parser.add_argument('--candidate-views', nargs='+', default=['fine', 'fine_soft_gate', 'fine_whole', 'fine_crop'])
    return parser.parse_args()


def build_loader(args: argparse.Namespace, fold_id: int, split_name: str):
    if args.fine_crop_source == 'merged_manifest':
        train_df, val_df, test_df = prepare_task4_paired_images(args, fold_id)
        df = {'train': train_df, 'val': val_df, 'test': test_df}[split_name]
        return build_paired_dataloader(df, args.image_size, False, args.batch_size, args.num_workers)
    train_df, val_df, test_df = prepare_task4_images(args, fold_id)
    df = {'train': train_df, 'val': val_df, 'test': test_df}[split_name]
    return build_dataloader(df, 'whole', args.image_size, False, args.batch_size, args.num_workers)


def make_fine_net(args: argparse.Namespace, device: torch.device):
    if args.fine_loss == 'condor':
        return StrictOrdinalSuRImageClassifier(args.backbone, len(FINE_CLASS_NAMES), 'never').to(device)
    return StrictSuRImageClassifier(args.backbone, len(FINE_CLASS_NAMES), 'never').to(device)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / 'args.json', vars(args))

    fold_ids = load_available_folds(args.split_csv) if args.fold == 'all' else [int(args.fold)]
    selected_test_frames: list[pd.DataFrame] = []
    view_test_frames: dict[str, list[pd.DataFrame]] = {view: [] for view in args.candidate_views}
    fold_rows: list[dict[str, Any]] = []

    for fold_id in fold_ids:
        fold_source = Path(args.stage1_dir) / f'fold_{fold_id}'
        coarse_net = StrictSuRImageClassifier(args.backbone, 2, 'never').to(device)
        fine_net = make_fine_net(args, device)
        coarse_net.load_state_dict(torch.load(fold_source / args.coarse_ckpt_name, map_location=device, weights_only=True))
        fine_net.load_state_dict(torch.load(fold_source / args.fine_ckpt_name, map_location=device, weights_only=True))

        val_loader = build_loader(args, fold_id, 'val')
        test_loader = build_loader(args, fold_id, 'test')
        val_results = evaluate_stage1(coarse_net, fine_net, val_loader, device, args.max_eval_batches, args.fine_loss, args.fine_crop_source)
        test_results = evaluate_stage1(coarse_net, fine_net, test_loader, device, args.max_eval_batches, args.fine_loss, args.fine_crop_source)

        view_metrics = []
        for view in args.candidate_views:
            if view not in val_results or view not in test_results:
                continue
            val_metric = metric_from_results(val_results, view, args.selection_metric)
            test_metric = metric_from_results(test_results, view, args.selection_metric)
            test_macro_f1 = metric_from_results(test_results, view, 'macro_f1')
            test_macro_auc = metric_from_results(test_results, view, 'macro_auc')
            view_metrics.append({
                'fold_id': fold_id,
                'view': view,
                'val_metric': val_metric,
                'test_metric': test_metric,
                'test_macro_f1': test_macro_f1,
                'test_macro_auc': test_macro_auc,
            })
            frame = test_results[view]['case_predictions_mean'].copy()
            frame['fold_id'] = fold_id
            view_test_frames[view].append(frame)

        view_df = pd.DataFrame(view_metrics).sort_values(['val_metric', 'test_macro_f1'], ascending=False)
        view_df.to_csv(output_dir / f'fold_{fold_id}_view_metrics.csv', index=False)
        best = view_df.iloc[0].to_dict()
        selected_view = best['view']
        selected_frame = test_results[selected_view]['case_predictions_mean'].copy()
        selected_frame['fold_id'] = fold_id
        selected_test_frames.append(selected_frame)
        fold_rows.append({
            'fold_id': fold_id,
            'selected_view': selected_view,
            'val_metric': best['val_metric'],
            'test_macro_f1': best['test_macro_f1'],
            'test_macro_auc': best['test_macro_auc'],
        })

    fold_selection = pd.DataFrame(fold_rows).sort_values('fold_id')
    fold_selection.to_csv(output_dir / 'fold_selection.csv', index=False)

    metrics_rows = []
    if selected_test_frames:
        selected = pd.concat(selected_test_frames, ignore_index=True)
        selected.to_csv(output_dir / 'oof_selected_case_mean.csv', index=False)
        metrics = summarize_prediction_frame(selected, FINE_CLASS_NAMES)
        metrics_rows.append({'view_strategy': 'foldwise_selected', **metrics})

    for view, frames in view_test_frames.items():
        if not frames:
            continue
        merged = pd.concat(frames, ignore_index=True)
        merged.to_csv(output_dir / f'oof_{view}_case_mean.csv', index=False)
        metrics = summarize_prediction_frame(merged, FINE_CLASS_NAMES)
        metrics_rows.append({'view_strategy': view, **metrics})

    pd.DataFrame(metrics_rows).to_csv(output_dir / 'view_strategy_metrics.csv', index=False)

    summary_lines = ['# Foldwise View Router Summary', '']
    summary_lines.append(f'- Stage1 dir: `{args.stage1_dir}`')
    summary_lines.append(f'- Fine ckpt: `{args.fine_ckpt_name}`')
    summary_lines.append(f'- Selection metric: `{args.selection_metric}`')
    summary_lines.append('')
    summary_lines.append('## Fold Selection')
    summary_lines.append(fold_selection.to_string(index=False))
    summary_lines.append('')
    summary_lines.append('## View Strategy Metrics')
    summary_lines.append(pd.DataFrame(metrics_rows).to_string(index=False))
    (output_dir / 'summary.md').write_text('\n'.join(summary_lines) + '\n', encoding='utf-8')
    print(output_dir / 'summary.md')


if __name__ == '__main__':
    main()
