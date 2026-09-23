"""Reuse frozen trained experts; fit a train-only amplitude/utility table."""
import argparse
import json
import os
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader

from run import build_parser, configure_args, load_finetuned_model
from exp.exp_timedart import Exp_TimeDART
from utils.experiment_audit import checkpoint_info, data_file_info, write_run_manifest
from utils.utility_calibration import split_utility_training
from utils.wiki_gain_calibration import calibration_blocks, fit_gain_policy


def resolve_source(source, fold, seed):
    log = Path(source) / f'runs/f{fold}_s{seed}/finetune.log'
    marker = '[AUDIT] FINETUNE_CHECKPOINT='
    matches = [line.split(marker, 1)[1].strip() for line in log.read_text(encoding='utf-8').splitlines() if marker in line]
    if not matches:
        raise RuntimeError(f'No completed fine-tune checkpoint marker in {log}')
    # Never reuse the epoch-zero fallback: it has untrained event experts.
    checkpoint = Path(matches[-1]).parent / 'checkpoint_last.pth'
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Required trained last checkpoint missing: {checkpoint}')
    manifest = json.loads((checkpoint.parent / 'run_manifest.json').read_text(encoding='utf-8'))
    args = manifest['args']
    if not args.get('utility_factorized') or args.get('sdwpf_fold') != fold or args.get('seed') != seed:
        raise ValueError('Source must be a factorized run with matching fold and seed')
    return checkpoint, manifest


def restore_experiment(source_args, output):
    args = build_parser().parse_args(['--task_name', 'finetune', '--model_id', 'SDWPF',
                                     '--model', 'PromptTimeDART', '--data', 'SDWPF'])
    vars(args).update(source_args)
    args.is_training, args.report_split = 0, 'val'
    args.evaluate_test_after_train = False
    args.freeze_non_utility, args.overlay_checkpoint = False, None
    args.utility_adapter_warmup_epochs = 0
    args.pretrain_init, args.load_checkpoints, args.allow_random_init = 'none', None, True
    args.use_multi_gpu, args.wiki_diagnostic = False, False
    args.run_id = f'gain_policy_{output.name}'
    args = configure_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return Exp_TimeDART(args)


def collect_calibration(exp, dataset):
    if dataset.flag != 'train':
        raise ValueError('Calibration loader must contain training windows only')
    loader = DataLoader(dataset, batch_size=exp.args.eval_batch_size, shuffle=False,
                        drop_last=False, num_workers=0)
    arrays = {key: [] for key in ('base', 'candidates', 'target', 'available', 'trend')}
    exp.model.eval()
    with torch.no_grad():
        for i, (x, y, _, _) in enumerate(loader):
            with exp._autocast():
                exp.model(x.float().to(exp.device))
            aux = exp.model._last_utility_aux
            values = {'base': aux['base_prediction'][..., 0],
                      'candidates': aux['factor_predictions'][:, :, 0],
                      'target': y[:, -exp.args.pred_len:, -1],
                      'available': aux['factor_availability'], 'trend': aux['trend_probs'].argmax(-1)}
            for key, value in values.items():
                arrays[key].append(value.detach().cpu().numpy())
            if i % 100 == 0 or i + 1 == len(loader):
                print(f'[GAIN] train calibration inference {i + 1}/{len(loader)}', flush=True)
    return {key: np.concatenate(value) for key, value in arrays.items()}


def calibrate(exp, source_manifest, checkpoint, output, blocks=3, min_windows=32, penalty=.25):
    load_finetuned_model(exp, str(checkpoint))
    exp.model.eval().requires_grad_(False)
    if bool(exp.model.utility_gate.policy_enabled):
        raise ValueError('Source must be the original trained neural gate, not an already calibrated policy')
    actual_file = data_file_info(exp.args)
    old_hash = source_manifest.get('data_file', {}).get('sha256')
    if old_hash and actual_file['sha256'] != old_hash:
        raise ValueError('Dataset hash changed since source training')
    if not old_hash:
        print('[WARNING] Source manifest has no data hash; verifying scaler and chronological split only')
    train_data, _ = exp._get_data('train')
    _, calibration_data, split_audit = split_utility_training(train_data, exp.args.utility_calibration_fraction)
    expected = source_manifest['args'].get('utility_calibration_audit')
    if not expected:
        raise ValueError('Source lacks adapter/gate temporal split audit; cannot prove expert-label separation')
    for key in ('cutoff', 'adapter_windows', 'gate_windows', 'adapter_target_end', 'gate_target_start'):
        if expected.get(key) != split_audit[key]:
            raise ValueError(f'Source training/calibration boundary mismatch: {key}')
    np.testing.assert_allclose(train_data.scaler.mean_, exp.model.scene_scaler_mean.cpu().numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(train_data.scaler.scale_, exp.model.scene_scaler_scale.cpu().numpy(), rtol=1e-5, atol=1e-5)
    ids, block_audit = calibration_blocks(calibration_data, blocks)
    arrays = collect_calibration(exp, calibration_data)
    mean, scale = float(train_data.scaler.mean_[-1]), float(train_data.scaler.scale_[-1])
    bounds = ((-mean / scale), (exp.args.rated_power - mean) / scale) if exp.args.rated_power > 0 else None
    alpha, gain, rows = fit_gain_policy(**arrays, block_ids=ids, source_split='train',
                                      min_windows=min_windows, penalty=penalty,
                                      min_gain=exp.args.utility_min_gain, clip_bounds=bounds)
    names = [*exp.model.scene_wiki_scene_ids, 'composition']
    for row in rows:
        row['candidate_name'] = names[row['candidate']]
        row['score_kw'] = row['score'] * scale
    audit = {'source_split': 'train', 'fit_uses_validation': False, 'source_checkpoint': checkpoint_info(checkpoint),
             'split': split_audit, 'blocks': block_audit, 'min_windows_per_block': min_windows,
             'block_dispersion_penalty': penalty, 'amplitude_grid': [0., .25, .5, 1.],
             'gain_units': 'train_target_std', 'candidate_names': names, 'cells': rows}
    (output / 'gain_calibration.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    exp.model.utility_gate.install_policy(alpha, gain)
    target = output / 'checkpoint.pth'
    torch.save(exp.model.state_dict(), target)
    # Freeze and save the policy BEFORE accessing validation. No validation
    # outcome adjusts the table or chooses which event to keep.
    write_run_manifest(output, exp.args, 'wiki_gain_calibration', model=exp.model,
                       datasets={'train': train_data}, checkpoints={'policy': checkpoint_info(target)},
                       extra={'policy_fit': audit, 'validation_not_yet_read': True})
    print(f'[GAIN] Policy frozen: {target}; active cells={int((alpha > 0).sum())}/{alpha.size}', flush=True)
    for k, name in enumerate(names):
        print(f'[GAIN] {name}: enabled_trend_horizons={int((alpha[..., k] > 0).sum())}/{alpha.shape[0] * alpha.shape[1]}')
    val_data, val_loader = exp._get_data('val')
    criterion = exp._select_criterion()
    calibrated = exp.valid(val_loader, criterion)
    exp.model.utility_gate.install_policy(np.zeros_like(alpha), np.zeros_like(gain))
    trend = exp.valid(val_loader, criterion)
    exp.model.utility_gate.policy_enabled.fill_(False)
    neural = exp.valid(val_loader, criterion)
    exp.model.eval()
    exp.model.utility_gate.install_policy(alpha, gain)
    results = {'calibrated': calibrated, 'trend': trend, 'previous_neural_last': neural}
    (output / 'validation_metrics.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    base_mae, new_mae = trend['original_mae'], calibrated['original_mae']
    lines = ['EVALUATION_SPLIT=val', 'POLICY_FIT_SPLIT=train', f'SOURCE_CHECKPOINT={checkpoint}',
             f'FOLD={exp.args.sdwpf_fold}', f'SEED={exp.args.seed}',
             f'TREND_MAE_KW={base_mae:.6f}', f'PREVIOUS_NEURAL_LAST_MAE_KW={neural["original_mae"]:.6f}',
             f'CALIBRATED_MAE_KW={new_mae:.6f}', f'CALIBRATED_RMSE_KW={calibrated["original_rmse"]:.6f}',
             f'GAIN_VS_TREND_KW={base_mae - new_mae:.6f}',
             f'GAIN_VS_TREND_PCT={100 * (1 - new_mae / max(base_mae, 1e-12)):.6f}',
             f'ACTIVE_POLICY_CELLS={int((alpha > 0).sum())}', f'CHECKPOINT={target}']
    (output / 'summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    write_run_manifest(output, exp.args, 'wiki_gain_calibration', model=exp.model,
                       datasets={'train': train_data, 'val': val_data}, checkpoints={'policy': checkpoint_info(target)},
                       extra={'policy_fit': audit, 'status': 'complete', 'validation': results,
                              'policy_frozen_before_validation': True})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--blocks', type=int, default=3)
    parser.add_argument('--min-windows', type=int, default=32)
    parser.add_argument('--penalty', type=float, default=.25)
    options = parser.parse_args()
    checkpoint, manifest = resolve_source(options.source_dir, options.fold, options.seed)
    output = Path(options.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'checkpoint.pth').exists():
        raise FileExistsError('Refusing to overwrite an existing policy checkpoint')
    exp = restore_experiment(manifest['args'], output)
    try:
        calibrate(exp, manifest, checkpoint, output, options.blocks, options.min_windows, options.penalty)
    finally:
        exp.writer.close()


if __name__ == '__main__':
    main()
