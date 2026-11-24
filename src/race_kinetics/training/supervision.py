from collections import Counter
from functools import lru_cache

import numpy as np
import torch
from rdkit import Chem


@lru_cache(maxsize=200000)
def canonical_side(value):
    parts = []
    for text in str(value).strip().split('.'):
        if not text:
            return None
        molecule = Chem.MolFromSmiles(text)
        if molecule is None:
            return None
        for atom in molecule.GetAtoms():
            atom.SetAtomMapNum(0)
        parts.append(Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True))
    return '.'.join(sorted(parts)) if parts else None


class DirectionMask:
    def __init__(self, reactions, pairs):
        if reactions.reaction_row.tolist() != list(range(len(reactions))):
            raise ValueError('Reaction rows must be contiguous and ordered')
        self.pairs = pairs.copy()
        self.eligible, self.reasons = {}, {}
        left = [canonical_side(x) for x in reactions.substrate]
        right = [canonical_side(x) for x in reactions['product']]
        valid = [a is not None and b is not None for a, b in zip(left, right)]
        for split in ('train', 'valid'):
            ids = set(pairs.loc[pairs.split.eq(split), 'reaction_row'])
            edges = {(left[i], right[i]) for i in ids if valid[i]}
            mask = np.zeros(len(reactions), dtype=bool)
            reasons = ['outside_split'] * len(reactions)
            for i in ids:
                a, b = left[i], right[i]
                if not valid[i]:
                    reasons[i] = 'invalid_sides'
                elif a == b:
                    reasons[i] = 'identical_sides'
                elif (b, a) in edges:
                    reasons[i] = 'reverse_record_conflict'
                else:
                    reasons[i] = 'eligible_record_orientation'
                    mask[i] = True
            self.eligible[split], self.reasons[split] = mask, reasons

    def require_samples(self):
        for split, mask in self.eligible.items():
            if not mask.any():
                raise ValueError(f'No DIR samples remain after conflict masking in {split}')

    def attach(self, batch, *, training):
        split = 'train' if training else 'valid'
        ids = batch['reaction_row'].detach().cpu().tolist()
        if any(self.reasons[split][i] == 'outside_split' for i in ids):
            raise ValueError('DIR batch contains a reaction outside its declared split')
        batch['direction_eligible'] = torch.as_tensor(
            self.eligible[split][ids], device=batch['reaction_row'].device)

    def report(self):
        result = {
            'policy': 'Within each split only; exclude invalid, identical and reversed-record conflicts. '
                      'Remaining record orientations are NOT verified biochemical directions.',
            'splits': {},
        }
        for split in ('train', 'valid'):
            ids = self.pairs.loc[self.pairs.split.eq(split), 'reaction_row'].to_numpy(dtype=int)
            count = int(np.count_nonzero(self.eligible[split][ids]))
            result['splits'][split] = {
                'pairs': len(ids), 'eligible_pairs': count, 'masked_pairs': len(ids) - count,
                'eligible_pair_fraction': count / len(ids) if len(ids) else 0.0,
                'reaction_row_reasons': dict(Counter(self.reasons[split][i] for i in set(ids))),
                'pair_reasons': dict(Counter(self.reasons[split][i] for i in ids)),
            }
        return result
