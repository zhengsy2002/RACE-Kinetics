import atexit
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from .config import PROJECT_ROOT
from .utils import sha256, file_identity

DATA_ROOT = PROJECT_ROOT / 'data'
_RECORDS = {}


def content_key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def clear_runtime():
    _RECORDS.clear()


atexit.register(clear_runtime)


@contextmanager
def runtime_session():
    from .runtime import lifecycle
    clear_runtime()
    try:
        with lifecycle():
            yield
    finally:
        clear_runtime()


def read_record(logical_path, default=None):
    key = str(Path(logical_path).resolve())
    if key in _RECORDS:
        return json.loads(json.dumps(_RECORDS[key]))
    return default


def write_record(logical_path, value):
    _RECORDS[str(Path(logical_path).resolve())] = json.loads(json.dumps(value))


def freeze_record(logical_path, value):
    value = json.loads(json.dumps(value))
    old = read_record(logical_path)
    if old is not None and old != value:
        raise RuntimeError(f'Frozen inputs/configuration changed: {logical_path}. Use a new suite directory.')
    if old is None:
        write_record(logical_path, value)


def bind_record(logical_path, files, config):
    identity = file_identity(files, config)
    freeze_record(logical_path, identity)
    return identity


def checkpoint_file(arm):
    return Path(arm) / 'weights/best_reaction_alignment.pt'


def handcrafted_dir(config):
    return DATA_ROOT / 'features/handcrafted'


def aligned_dir(config, checkpoint):
    data = config['data']
    files = [Path(data['downstream_csv']), Path(checkpoint),
        Path(data['protein_downstream_cache'])/'unique_embeddings.npy',
        Path(data['protein_downstream_cache'])/'sample_row_to_cache_row.npy',
        Path(data['molformer_cache'])/'side_embeddings.npy',
        Path(data['molformer_cache'])/'unique_sides.csv']
    from importlib.metadata import version
    package = Path(__file__).parent
    identity = {'layout': 'reactant_enzyme_reaction_768', 'hashes': [sha256(p) for p in files],
        'torch':version('torch'),'numpy':version('numpy'),
        'code':{name:sha256(package/name) for name in ('features/alignment.py','models/alignment.py',
            'models/projections.py','data/protocol.py','data/caches.py')}}
    return DATA_ROOT / 'features/aligned' / content_key(identity)


def atomic_csv(frame, path):
    from .runtime import atomic_output
    with atomic_output(path) as temporary:
        frame.to_csv(temporary, index=False, lineterminator='\n')
