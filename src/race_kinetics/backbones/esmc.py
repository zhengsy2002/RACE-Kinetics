from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from ..utils import write_json as _write_json
from ..config import PROJECT_ROOT
def _normalize(value: object) -> str:
    return re.sub('[UZOB]', 'X', re.sub('\\s+', '', str(value).upper()))

def _sha1(value: str) -> str:
    return hashlib.sha1(value.encode('utf-8')).hexdigest()

def _kcat_index(data_path: Path = PROJECT_ROOT / 'data/kcat.csv') -> tuple[list[str], pd.DataFrame, np.ndarray, dict[str, object]]:
    frame = pd.read_csv(data_path, usecols=['sequence', 'uniprot'], dtype=str).fillna('')
    sequences = frame['sequence'].map(_normalize).tolist()
    if any((not value for value in sequences)):
        raise ValueError('data contains empty normalized sequences')
    hashes = [_sha1(value) for value in sequences]
    unique_hashes = list(dict.fromkeys(hashes))
    sequence_by_hash = dict(zip(hashes, sequences))
    row_by_hash = {value: row for row, value in enumerate(unique_hashes)}
    unique_sequences = [sequence_by_hash[value] for value in unique_hashes]
    representatives = pd.DataFrame({'sequence_sha1': hashes, 'uniprot': frame['uniprot']}).drop_duplicates('sequence_sha1', keep='first').set_index('sequence_sha1')['uniprot']
    manifest = pd.DataFrame({'cache_row': np.arange(len(unique_hashes), dtype=np.int64), 'sequence_sha1': unique_hashes, 'sequence_length': [len(value) for value in unique_sequences], 'representative_uniprot': [representatives.get(value, '') for value in unique_hashes]})
    sample_to_cache = np.asarray([row_by_hash[value] for value in hashes], dtype=np.int64)
    uniprot_sequence_counts = pd.DataFrame({'uniprot': frame['uniprot'], 'sequence_sha1': hashes}).groupby('uniprot')['sequence_sha1'].nunique()
    multi = set(uniprot_sequence_counts[uniprot_sequence_counts > 1].index)
    stats = {'dataset': 'kcat', 'rows': len(frame), 'unique_sequences': len(unique_hashes), 'unique_uniprots': int(frame['uniprot'].nunique()), 'uniprots_with_multiple_sequences': len(multi), 'rows_affected_by_multi_sequence_uniprot': int(frame['uniprot'].isin(multi).sum())}
    return (unique_sequences, manifest, sample_to_cache, stats)

def _reactzyme_index(prepared_dir: Path, protein_tsv: Path) -> tuple[list[str], pd.DataFrame, np.ndarray, dict[str, object]]:
    index_path = prepared_dir / 'protein_index.tsv'
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    index = pd.read_csv(index_path, sep='\t', dtype=str).fillna('')
    required = {'protein_row', 'entry', 'sequence_sha1'}
    if not required.issubset(index.columns):
        raise KeyError(f'ReactZyme protein index requires {sorted(required)}')
    source = pd.read_csv(protein_tsv, sep='\t', usecols=['Entry', 'Sequence'], dtype=str).fillna('')
    source['_sequence'] = source['Sequence'].map(_normalize)
    source = source.drop_duplicates('Entry', keep='first')
    sequence_by_entry = dict(zip(source['Entry'], source['_sequence']))
    protein_sequences = [sequence_by_entry.get(value, '') for value in index['entry']]
    if any((not value for value in protein_sequences)):
        missing = [index.iloc[row]['entry'] for row, sequence in enumerate(protein_sequences) if not sequence]
        raise KeyError(f'Missing ReactZyme sequences: {missing[:5]}')
    actual_hashes = [_sha1(value) for value in protein_sequences]
    expected_hashes = index['sequence_sha1'].astype(str).tolist()
    mismatch = [row for row, (actual, expected) in enumerate(zip(actual_hashes, expected_hashes)) if actual != expected]
    if mismatch:
        first = mismatch[0]
        raise RuntimeError(f"ReactZyme source sequence does not match prepared sequence_sha1: protein_row={first}, entry={index.iloc[first]['entry']}")
    unique_hashes = list(dict.fromkeys(actual_hashes))
    sequence_by_hash = dict(zip(actual_hashes, protein_sequences))
    row_by_hash = {value: row for row, value in enumerate(unique_hashes)}
    unique_sequences = [sequence_by_hash[value] for value in unique_hashes]
    first_entry = dict(zip(actual_hashes, index['entry'].astype(str)))
    manifest = pd.DataFrame({'cache_row': np.arange(len(unique_hashes), dtype=np.int64), 'sequence_sha1': unique_hashes, 'sequence_length': [len(value) for value in unique_sequences], 'representative_entry': [first_entry[value] for value in unique_hashes]})
    protein_to_cache = np.asarray([row_by_hash[value] for value in actual_hashes], dtype=np.int64)
    stats = {'dataset': 'reactzyme', 'protein_rows': len(index), 'unique_sequences': len(unique_hashes), 'duplicate_sequence_rows_reused': len(index) - len(unique_hashes), 'sequence_hash_mismatches': 0}
    return (unique_sequences, manifest, protein_to_cache, stats)

def _embed_mean_transformers(model: torch.nn.Module, tokenizer: object, sequence: str, max_residues: int, overlap: int, device: torch.device) -> np.ndarray:
    starts = list(range(0, len(sequence), max_residues - overlap))
    chunks = [(start, sequence[start:start + max_residues]) for start in starts]
    sums = None
    counts = np.zeros(len(sequence), dtype=np.float32)
    backbone = getattr(model, 'esmc', None)
    if backbone is None:
        backbone = getattr(model, 'model', None)
    if backbone is None:
        raise RuntimeError('Transformers ESM-C masked-LM model exposes neither .esmc nor .model')
    for start, chunk in chunks:
        encoded = tokenizer(chunk, add_special_tokens=True, return_attention_mask=True, return_special_tokens_mask=True, return_tensors='pt')
        special_mask = encoded.pop('special_tokens_mask', None)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            output = backbone(**encoded, return_dict=True)
        hidden = output.last_hidden_state[0]
        attention_mask = encoded.get('attention_mask')
        if attention_mask is None:
            keep = torch.ones(hidden.shape[0], dtype=torch.bool, device=hidden.device)
        else:
            keep = attention_mask[0].bool()
        if special_mask is not None:
            keep = keep & ~special_mask[0].to(device).bool()
        else:
            input_ids = encoded['input_ids'][0]
            special_ids = torch.as_tensor(list(getattr(tokenizer, 'all_special_ids', [])), device=input_ids.device, dtype=input_ids.dtype)
            if special_ids.numel():
                keep = keep & ~torch.isin(input_ids, special_ids)
        hidden = hidden[keep]
        if hidden.shape[0] != len(chunk):
            raise RuntimeError(f'Transformers ESM-C tokenizer did not produce exactly one non-special token per residue: tokens={hidden.shape[0]}, residues={len(chunk)}')
        values = hidden.float().cpu().numpy()
        if sums is None:
            sums = np.zeros((len(sequence), values.shape[-1]), dtype=np.float32)
        end = start + len(chunk)
        sums[start:end] += values
        counts[start:end] += 1
    if sums is None or np.any(counts == 0):
        raise RuntimeError('Failed to cover the complete protein sequence')
    return (sums / counts[:, None]).mean(axis=0)

def extract(args) -> None:
    if not args.model_dir.is_dir():
        raise FileNotFoundError(args.model_dir)
    dataset = getattr(args, 'dataset', 'kcat')
    if dataset == 'kcat':
        kcat_csv = Path(getattr(args, 'kcat_csv', None) or PROJECT_ROOT / 'data/kcat.csv')
        sequences, manifest, mapping, stats = _kcat_index(kcat_csv)
        mapping_name = 'sample_row_to_cache_row.npy'
        source = str(kcat_csv.resolve())
    elif dataset == 'reactzyme':
        if args.prepared_dir is None or args.protein_tsv is None:
            raise ValueError('--prepared-dir and --protein-tsv are required for reactzyme')
        sequences, manifest, mapping, stats = _reactzyme_index(args.prepared_dir, args.protein_tsv)
        mapping_name = 'protein_row_to_cache_row.npy'
        source = str(args.protein_tsv.resolve())
    else:
        raise ValueError(f'Unsupported dataset: {dataset}; expected kcat or reactzyme')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / 'unique_sequences.csv'
    mapping_path = args.output_dir / mapping_name
    if manifest_path.exists():
        import pandas as pd
        old = pd.read_csv(manifest_path, keep_default_na=False)
        if old['sequence_sha1'].tolist() != manifest['sequence_sha1'].tolist():
            raise RuntimeError(f'Existing cache has different sequences: {manifest_path}')
    else:
        manifest.to_csv(manifest_path, index=False)
    if mapping_path.exists():
        if not np.array_equal(np.load(mapping_path), mapping):
            raise RuntimeError(f'Existing cache has a different row mapping: {mapping_path}')
    else:
        np.save(mapping_path, mapping)
    device = torch.device(args.device)
    from .download import model_status
    missing = model_status(args.model_dir, 'esmc')
    if missing:
        raise FileNotFoundError(f'Incomplete ESM-C encoder: {missing}')
    config_path = args.model_dir / 'config.json'
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    with config_path.open('r', encoding='utf-8') as handle:
        model_config = json.load(handle)
    try:
        hidden_size = int(model_config['d_model'] if 'd_model' in model_config else model_config['hidden_size'])
        n_heads = int(model_config['n_heads'] if 'n_heads' in model_config else model_config['num_attention_heads'])
        n_layers = int(model_config['n_layers'] if 'n_layers' in model_config else model_config['num_hidden_layers'])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f'Invalid ESM-C config in {config_path}; expected integer d_model/n_heads/n_layers or their standard Transformers equivalents') from error
    if hidden_size <= 0 or n_heads <= 0 or n_layers <= 0:
        raise RuntimeError(f'Invalid ESM-C architecture in {config_path}')
    backbone_name = {(960, 15, 30): 'ESMC-300M', (1152, 18, 36): 'ESMC-600M', (2560, 40, 80): 'ESMC-6B'}.get((hidden_size, n_heads, n_layers), 'ESMC-custom')
    print(f'Loading {backbone_name}: d_model={hidden_size}, n_heads={n_heads}, n_layers={n_layers}')
    tokenizer = None
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForMaskedLM.from_pretrained(args.model_dir, local_files_only=True).eval()
    dtype = getattr(torch, args.model_dtype)
    model = model.to(device=device, dtype=dtype)
    loaded_hidden_size = int(getattr(model.config, 'hidden_size', getattr(model.config, 'd_model', 0)))
    if loaded_hidden_size != hidden_size:
        raise RuntimeError(f'Loaded ESM-C hidden size {loaded_hidden_size} != config {hidden_size}')
    embedding_path = args.output_dir / 'unique_embeddings.npy'
    done_path = args.output_dir / 'done.npy'
    if embedding_path.exists():
        embeddings = np.load(embedding_path, mmap_mode='r+')
        if embeddings.shape != (len(sequences), hidden_size) or embeddings.dtype != np.float16:
            raise RuntimeError(f'Incompatible cache matrix: {embeddings.shape}/{embeddings.dtype}')
    else:
        embeddings = np.lib.format.open_memmap(embedding_path, mode='w+', dtype=np.float16, shape=(len(sequences), hidden_size))
        embeddings[:] = np.nan
        embeddings.flush()
    if done_path.exists():
        done = np.load(done_path, mmap_mode='r+')
        if done.shape != (len(sequences),):
            raise RuntimeError(f'Incompatible completion map: {done.shape}')
    else:
        done = np.lib.format.open_memmap(done_path, mode='w+', dtype=np.uint8, shape=(len(sequences),))
        done[:] = 0
        done.flush()
    pending = np.flatnonzero(np.asarray(done) == 0)
    if args.max_sequences > 0:
        pending = pending[:args.max_sequences]
    for completed, row in enumerate(tqdm(pending, desc=f'{dataset} ESM-C', unit='sequence'), 1):
        value = _embed_mean_transformers(model, tokenizer, sequences[int(row)], args.max_residues, args.overlap, device)
        embeddings[int(row)] = value.astype(np.float16)
        done[int(row)] = 1
        if completed % args.flush_every == 0:
            embeddings.flush()
            done.flush()
    embeddings.flush()
    done.flush()
    metadata = {**stats, 'complete': int(np.asarray(done).sum()) == len(sequences), 'completed_sequences': int(np.asarray(done).sum()), 'source': source, 'input_key': 'normalized_sequence_sha1', 'cache_semantics': 'sequence-keyed; never UniProt-keyed', 'backbone': backbone_name, 'backend': args.backend, 'model_dtype': args.model_dtype, 'hidden_size': hidden_size, 'dtype': 'float16', 'architecture': {'d_model': hidden_size, 'n_heads': n_heads, 'n_layers': n_layers}, 'pooling': 'residue mean after overlap-averaged ESM-C chunks', 'model_dir': str(args.model_dir.resolve()), 'uses_kcat_labels': False, 'num_unique_sequences': len(sequences), 'mapping_file': mapping_name}
    _write_json(args.output_dir / 'metadata.json', metadata)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
