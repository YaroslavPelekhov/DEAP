"""
CLI script for training on DEAP.

Usage:
    python scripts/train.py --protocol sd  --model multimodal --save
    python scripts/train.py --protocol loso --model temporal  --save
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import config as cfg
from src.features.pipeline import FeaturePipeline
from src.training.trainer import train_sd, train_loso, DEVICE
from src.utils.io import save_results, timestamped_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', choices=['sd', 'loso', 'both'], default='sd')
    parser.add_argument('--model',    choices=['multimodal', 'temporal', 'mmcat'],
                        default='multimodal')
    parser.add_argument('--subjects', nargs='+', type=int, default=None,
                        help='Subject IDs (default: all 32)')
    parser.add_argument('--save',     action='store_true', help='Save results + weights')
    parser.add_argument('--no-cache', action='store_true', help='Force re-extract features')
    args = parser.parse_args()

    subject_ids = args.subjects or list(range(1, cfg.N_SUBJECTS + 1))
    models_dir  = (ROOT / 'data' / 'models') if args.save else None
    if models_dir:
        models_dir.mkdir(parents=True, exist_ok=True)

    pipeline = FeaturePipeline(
        data_dir=cfg.DATA_DIR,
        cache_dir=cfg.CACHE_DIR,
        cache_version='v13',
    )
    features = pipeline.run(subject_ids, force=args.no_cache)

    def _print_agg(name, agg):
        print(f'\n{"="*40}')
        print(f'{name} results ({len(subject_ids)} subjects):')
        print(f'  Valence:  {agg["valence_acc"]:.1f}% acc  |  {agg["valence_f1"]:.1f}% F1')
        print(f'  Arousal:  {agg["arousal_acc"]:.1f}% acc  |  {agg["arousal_f1"]:.1f}% F1')
        print(f'  Mean acc: {agg["mean_acc"]:.1f}%')

    all_results = {}

    if args.protocol in ('sd', 'both'):
        sd_res = train_sd(features, model_name=args.model,
                          device=DEVICE, models_dir=models_dir)
        _print_agg('SD', sd_res['aggregate'])
        all_results['sd'] = sd_res
        if args.save:
            save_results(sd_res, timestamped_path(ROOT / 'results',
                                                   f'results_sd_{args.model}'))

    if args.protocol in ('loso', 'both'):
        loso_res = train_loso(features, model_name=args.model,
                               device=DEVICE, models_dir=models_dir)
        _print_agg('LOSO', loso_res['aggregate'])
        all_results['loso'] = loso_res
        if args.save:
            save_results(loso_res, timestamped_path(ROOT / 'results',
                                                     f'results_loso_{args.model}'))


if __name__ == '__main__':
    main()
