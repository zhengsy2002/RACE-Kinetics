from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..models.alignment import checkpoint_components
from ..models.initialization import initialize_model
from ..losses import AlignmentLoss
from ..data.datasets import MolFormerPairDataset, collate
from ..data.caches import load_side_cache, validate_protein_metadata
from ..utils import (_mean_metrics, atomic_torch_save, capture_rng_state,
                     restore_rng_state, move_batch, _signature)
from ..storage import read_record, freeze_record
from .supervision import DirectionMask


def run_epoch(model, loader, device, *, optimizer, criterion, supervision=None):
    training = optimizer is not None
    model.train(training)
    rows = []
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch in loader:
            batch = move_batch(batch, device)
            if supervision is not None:
                supervision.attach(batch, training=training)
            if training:
                optimizer.zero_grad(set_to_none=True)
            loss, metrics = criterion(model(batch), batch)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite pretraining loss')
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
                optimizer.step()
            if 'direction_eligible' in batch:
                metrics['direction_eligible_fraction'] = batch['direction_eligible'].float().mean().item()
            rows.append(metrics)
    return _mean_metrics(rows, 'train' if training else 'valid')


def make_criterion(args):
    return AlignmentLoss(**{key: getattr(args, key + '_weight') for key in
                            ('enzyme_reaction', 'reactant_product', 'direction')})


def checkpoint_state(model, *, epoch, valid_loss, args, cache_metadata):
    return {
        'format': 'race_kinetics_alignment',
        'epoch': int(epoch), 'valid_loss': float(valid_loss),
        'molformer_dim': model.molformer_dim,
        'molformer_model': cache_metadata['config']['model'],
        'molformer_revision': cache_metadata['config'].get('revision'),
        'molformer_pooling': cache_metadata['config']['pooling'],
        'loss_weights': make_criterion(args).weights,
        'supervision': {'direction': args.direction_policy},
        'pretraining_binding': read_record(args.output_dir.parent / 'pretraining_binding.json'),
        **checkpoint_components(model),
    }


def _finish_weights(args, config, last):
    path = args.output_dir / 'best_reaction_alignment.pt'
    state = torch.load(path, map_location='cpu', weights_only=False)
    state['experiment_config'] = config
    state['pretraining_binding'] = read_record(args.output_dir.parent / 'pretraining_binding.json')
    state['training_completion'] = {
        'last_epoch': int(last['epoch']), 'configured_epochs': int(args.epochs),
        'bad_epochs': int(last['bad_epochs']), 'early_stopping_patience': int(args.patience),
        'best_loss': float(last['best_loss']),
    }
    atomic_torch_save(state, path)
    (args.output_dir / 'last.pt').unlink(missing_ok=True)


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')
    device = torch.device(args.device)
    pair_path = args.prepared_dir / 'pairs.tsv'
    reaction_path = args.prepared_dir / 'reaction_index.tsv'
    required = [pair_path, reaction_path, args.protein_cache_dir / 'unique_embeddings.npy',
                args.protein_cache_dir / 'protein_row_to_cache_row.npy',
                args.protein_cache_dir / 'metadata.json']
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'Pretraining inputs are missing: {missing}')
    pairs = pd.read_csv(pair_path, sep='\t')
    reactions = pd.read_csv(reaction_path, sep='\t').fillna('')
    protein_metadata = json.loads((args.protein_cache_dir / 'metadata.json').read_text(encoding='utf-8'))
    if not protein_metadata.get('complete', False) or protein_metadata.get('input_key') != 'normalized_sequence_sha1':
        raise RuntimeError('Pretraining requires a complete sequence-keyed ESM-C HF cache')
    validate_protein_metadata(protein_metadata)
    protein_cache = np.load(args.protein_cache_dir / 'unique_embeddings.npy', mmap_mode='r')
    protein_rows = np.load(args.protein_cache_dir / 'protein_row_to_cache_row.npy', mmap_mode='r')
    if protein_cache.ndim != 2 or protein_cache.shape[1] != 1152:
        raise ValueError('Expected 1152-dimensional ESM-C HF input')
    if (protein_rows.ndim != 1 or protein_rows.dtype.kind not in 'iu' or not len(protein_rows)
            or protein_rows.min() < 0 or protein_rows.max() >= len(protein_cache)):
        raise ValueError('Invalid protein cache row mapping')
    side_embeddings, side_lookup, side_metadata = load_side_cache(args.molformer_cache_dir)
    reactant_rows = np.asarray([side_lookup[str(v)] for v in reactions['substrate']], dtype=np.int64)
    product_rows = np.asarray([side_lookup[str(v)] for v in reactions['product']], dtype=np.int64)
    train_pairs = pairs[pairs['split'] == 'train']
    valid_pairs = pairs[pairs['split'] == 'valid']
    if train_pairs.empty or valid_pairs.empty:
        raise RuntimeError('Both train and valid ReactZyme pairs are required')
    supervision = None
    if args.direction_policy == 'conflict_masked_bce':
        supervision = DirectionMask(reactions, pairs)
        if args.direction_weight > 0:
            supervision.require_samples()
        print('DIR record-conflict mask:', json.dumps(supervision.report()), flush=True)
    elif args.direction_policy != 'all_bce':
        raise ValueError('Unsupported direction policy')

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        'prepared_dir': str(args.prepared_dir.resolve()),
        'protein_cache_dir': str(args.protein_cache_dir.resolve()),
        'molformer_cache_dir': str(args.molformer_cache_dir.resolve()),
        'molformer_cache_metadata': _signature(args.molformer_cache_dir / 'metadata.json'),
        'train_pairs': int(len(train_pairs)), 'valid_pairs': int(len(valid_pairs)),
        'arguments': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    freeze_record(args.output_dir / 'experiment_config.json', config)
    binding = read_record(args.output_dir.parent / 'pretraining_binding.json')
    if binding is None:
        raise ValueError('Training requires input/code binding from the experiment runner')
    dataset_options = dict(protein_cache=protein_cache, protein_row_to_cache_row=protein_rows,
                           reactant_rows=reactant_rows, product_rows=product_rows, side_embeddings=side_embeddings)
    generator = torch.Generator().manual_seed(args.seed)
    loader_options = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                          collate_fn=collate, pin_memory=device.type == 'cuda')
    train_loader = DataLoader(MolFormerPairDataset(train_pairs, **dataset_options),
                              shuffle=True, generator=generator, **loader_options)
    valid_loader = DataLoader(MolFormerPairDataset(valid_pairs, **dataset_options),
                              shuffle=False, **loader_options)
    model = initialize_model(seed=args.seed, molformer_dim=side_embeddings.shape[1], dropout=args.dropout,
                             protein_input_dim=protein_cache.shape[1], protein_projection=args.protein_projection,
                             protein_bottleneck=args.protein_bottleneck).to(device)
    criterion = make_criterion(args).to(device)
    adapter_parameters = list(model.molecule_adapter.parameters())
    head_parameters = (list(model.reactant_projection.parameters()) + list(model.product_projection.parameters())
                       + list(model.direction_head.parameters()) + [model.reactant_product_logit_scale])
    separate_ids = {id(p) for p in adapter_parameters + head_parameters}
    base_parameters = [p for p in model.parameters() if id(p) not in separate_ids]
    optimizer = torch.optim.AdamW([
        {'params': base_parameters, 'lr': args.lr},
        {'params': adapter_parameters, 'lr': args.adapter_lr},
        {'params': head_parameters, 'lr': args.new_head_lr},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=args.scheduler_patience, min_lr=1e-6)
    last_path = args.output_dir / 'last.pt'
    start_epoch, best_loss, bad_epochs = 1, float('inf'), 0
    if last_path.is_file():
        state = torch.load(last_path, map_location='cpu', weights_only=False)
        if state.get('format') != 'race_kinetics_resume':
            raise ValueError('Unsupported resume checkpoint format')
        if state.get('experiment_config') != config or state.get('pretraining_binding') != binding:
            raise ValueError('Resume inputs, code or configuration changed; use a new results directory')
        model.load_state_dict(state['full_model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        start_epoch = int(state['epoch']) + 1
        best_loss, bad_epochs = float(state['best_loss']), int(state['bad_epochs'])
        if start_epoch > args.epochs or bad_epochs >= args.patience:
            _finish_weights(args, config, state)
            return
        restore_rng_state(state['rng_state'], generator)
        print(f'Resuming pretraining at epoch {start_epoch}', flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer=optimizer,
                                   criterion=criterion, supervision=supervision)
        valid_metrics = run_epoch(model, valid_loader, device, optimizer=None,
                                   criterion=criterion, supervision=supervision)
        valid_loss = float(valid_metrics['valid_loss'])
        scheduler.step(valid_loss)
        if valid_loss < best_loss - 1e-4:
            best_loss, bad_epochs = valid_loss, 0
            atomic_torch_save(checkpoint_state(model, epoch=epoch, valid_loss=valid_loss,
                                               args=args, cache_metadata=side_metadata),
                              args.output_dir / 'best_reaction_alignment.pt')
        else:
            bad_epochs += 1
        last = {
            'format': 'race_kinetics_resume', 'full_model': model.state_dict(),
            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
            'epoch': epoch, 'best_loss': best_loss, 'bad_epochs': bad_epochs,
            'experiment_config': config, 'pretraining_binding': binding,
            'rng_state': capture_rng_state(generator),
        }
        atomic_torch_save(last, last_path)
        print(json.dumps({'epoch': epoch, **train_metrics, **valid_metrics}), flush=True)
        if bad_epochs >= args.patience:
            print(f'Early stopping at epoch {epoch}; best={best_loss:.6f}', flush=True)
            break
    _finish_weights(args, config, last)
