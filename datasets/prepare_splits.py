"""Organize existing frozen data without moving or modifying source files."""
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT.parent / 'benchmarks/super_resolution/artifacts'
CAMPAIGN = ARTIFACTS / 'text-low-bandwidth-e2e-v1'


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def link(destination, source):
    source = source.resolve(strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() and destination.resolve() == source:
        return
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.symlink_to(source, target_is_directory=source.is_dir())


def manifest(split, rows, **metadata):
    write(ROOT / split / 'manifest.json', dict(
        schema='isharescreen_directory_split_v1', split=split,
        count=len(rows), rows=rows, **metadata))


def training_source(row):
    group = row['source_group']
    if group.startswith('doclaynet:'):
        return 'doclaynet'
    if group.startswith('code:connectedhomeip:'):
        return 'matter_sdk'
    if group.startswith('code:iShareScreen:'):
        return 'isharescreen'
    if group.startswith('generated-document:') and row['kind'] in ('table', 'terminal'):
        return 'synthetic_' + row['kind']
    raise ValueError(f'Unrecognized training source: {group}')


def views(split, rows, base):
    output = []
    for original in rows:
        row = dict(original)
        row['original_split'] = row.get('split')
        row['split'] = split
        prefix = Path(training_source(row)) if split == 'train' else Path()
        if split == 'train':
            row['data_source'] = str(prefix)
        for field in ('lr', 'hr', 'labels'):
            if field not in row:
                continue
            source = Path(row[field])
            source = source if source.is_absolute() else base / source
            relative = prefix / 'views' / row['id'] / (field + source.suffix)
            link(ROOT / split / relative, source)
            row[field] = str(relative)
        output.append(row)
    return output


def main():
    packed_path = CAMPAIGN / 'packed-pilot-v1/manifest.json'
    protocol_path = CAMPAIGN / 'EVAL_PROTOCOL.json'
    packed = read(packed_path)
    protocol = read(protocol_path)
    train = [r for r in packed['rows'] if r['split'] == 'train']
    test = [r for r in protocol['cases'] if r['split'] == 'test']
    public_validation = (
        [dict(r, validation_role='checkpoint_proxy') for r in packed['rows']
         if r['split'] == 'validation'] +
        [dict(r, validation_role='configuration_selection') for r in protocol['cases']
         if r['split'] == 'validation'])
    groups = [{r['source_group'] for r in rows}
              for rows in (train, public_validation, test)]
    for i in range(len(groups)):
        for j in range(i):
            assert not groups[i] & groups[j], 'Public source groups cross splits'
    # Keep the Mac capture session together, including every derived frame.
    mac_base = ARTIFACTS / 'frozen-curated-v1'
    mac = read(mac_base / 'manifest.json')['pairs']
    mac_hashes = {r[k] for r in mac for k in ('hr_sha256', 'hr_rgb_sha256') if k in r}
    train_hashes = {r[k] for r in train for k in ('hr_sha256', 'hr_rgb_sha256') if k in r}
    assert not mac_hashes & train_hashes, 'Exact Mac frame duplicate in training'

    train_rows = views('train', train, CAMPAIGN)
    for source in sorted({r['data_source'] for r in train_rows}):
        subset = []
        for row in train_rows:
            if row['data_source'] != source:
                continue
            local = dict(row)
            for field in ('lr', 'hr', 'labels'):
                if field in local:
                    local[field] = str(Path(local[field]).relative_to(source))
            subset.append(local)
        write(ROOT / 'train' / source / 'manifest.json', dict(
            schema='isharescreen_directory_split_v1', split='train',
            data_source=source, count=len(subset), rows=subset))
    manifest('train', train_rows,
             kinds=dict(Counter(r['kind'] for r in train)),
             source_manifest=str(packed_path),
             source_manifest_sha256=hashlib.sha256(packed_path.read_bytes()).hexdigest())
    manifest('test', views('test', test, CAMPAIGN),
             historical_test_already_evaluated=True,
             source_protocol=str(protocol_path),
             source_protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest())

    originals = ROOT / 'cs682_dataset'
    original_alias = ROOT / 'validation/mac/original_recordings'
    if originals.is_dir():
        link(original_alias, originals)
    elif original_alias.is_symlink() and not original_alias.exists():
        original_alias.unlink()
    link(ROOT / 'validation/mac/processed_recordings', ARTIFACTS / 'source-videos')
    link(ROOT / 'validation/mac/frame_pairs', mac_base)
    mac_rows = []
    for original in mac:
        row = dict(original, split='validation', source_group='mac-capture-2026-10-15')
        row['historical_fold'] = row.pop('fold')
        row['historical_validation_eligible'] = row.pop('validation_eligible', None)
        row['validation_eligible'] = True
        for field in ('lr', 'hr'):
            row[field] = str(Path('mac/frame_pairs') / row[field])
        mac_rows.append(row)
    manifest('validation', mac_rows,
             role='Primary validation: all local Mac captures grouped together',
             original_recording_count=len(list((ROOT / 'cs682_dataset').glob('*.mov'))),
             original_recording_status='available' if originals.is_dir() else 'deleted_by_user_2026-10-29',
             processed_recording_count=len(read(ARTIFACTS / 'source-videos/manifest.json')['clips']),
             previously_used_for_training=True,
             source_manifest=str(mac_base / 'manifest.json'),
             source_manifest_sha256=hashlib.sha256((mac_base / 'manifest.json').read_bytes()).hexdigest(),
             scope='Validation for future runs excluding Mac captures; not unseen by historical models.')
    # Retain both existing public validation sets as auxiliary material.
    auxiliary = views('validation', public_validation, CAMPAIGN)
    write(ROOT / 'validation/public-manifest.json', dict(
        schema='isharescreen_directory_split_v1', split='validation',
        role='Auxiliary public validation, preserving original role per row',
        count=len(auxiliary), rows=auxiliary))
    print(json.dumps(dict(train=len(train), validation_mac_pairs=len(mac_rows),
                          validation_public=len(auxiliary), test=len(test))))


if __name__ == '__main__':
    main()
