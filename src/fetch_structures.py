"""Fetch target AlphaFold structures via public UniProt and Ensembl mappings."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import subprocess
import threading
import time

_NEXT_REQUEST = 0.0
_REQUEST_LOCK = threading.Lock()
AMINO_ACIDS = dict(zip(
    ('ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
     'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'),
    'ARNDCQEGHILKMFPSTWYV'))


def get(url):
    global _NEXT_REQUEST
    for attempt in range(6):
        with _REQUEST_LOCK:
            delay = max(0, _NEXT_REQUEST - time.monotonic())
            time.sleep(delay)
            _NEXT_REQUEST = time.monotonic() + .25
        result = subprocess.run(['curl', '-sS', '-L', '--max-time', '120',
                                 '-w', '\n%{http_code}', url], capture_output=True)
        body, _, status = result.stdout.rpartition(b'\n')
        if result.returncode == 0 and status == b'200':
            return body
        if result.returncode == 0 and status in (b'400', b'404'):
            return None
        if attempt < 5:
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError('Public structure service request failed: ' + url)


def accession(identifier):
    ensp = identifier.split('.', 1)[1]
    response = get('https://rest.uniprot.org/uniprotkb/search?query=' + ensp +
                   '+AND+organism_id:9606&fields=accession,reviewed&format=json')
    hits = json.loads(response).get('results', []) if response else []
    if hits:
        reviewed = [hit for hit in hits if hit.get('entryType', '').startswith('UniProtKB reviewed')]
        return (reviewed or hits)[0]['primaryAccession']
    response = get('https://rest.ensembl.org/xrefs/id/' + ensp + '?content-type=application/json')
    hits = json.loads(response) if response else []
    for database in ('Uniprot/SWISSPROT', 'Uniprot/SPTREMBL'):
        for hit in hits:
            if hit.get('dbname') == database:
                return hit.get('primary_id')
    return None


def sequence_identity(observed, expected):
    return max(sum(a == b for a, b in zip(observed[offset:], expected))
               if offset >= 0 else sum(a == b for a, b in zip(observed, expected[-offset:]))
               for offset in range(-5, 6)) / max(len(observed), len(expected), 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dictionary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    with args.dictionary.open(newline='') as stream:
        proteins = [(row[0], row[1].replace('\n', '')) for row in csv.reader(stream)
                    if len(row) >= 2 and row[0].startswith('9606.')]
    if not proteins or args.workers < 1:
        raise ValueError('A target dictionary and a positive worker count are required')
    args.output.mkdir(parents=True, exist_ok=True)
    mapping_path = args.output / 'mapping.json'
    mapping = json.loads(mapping_path.read_text()) if mapping_path.is_file() else {}
    mapping_lock = threading.Lock()
    pdb_directory = args.output / 'pdb'
    pdb_directory.mkdir(exist_ok=True)

    def fetch(item):
        identifier, expected = item
        try:
            if identifier not in mapping:
                mapped = accession(identifier)
                with mapping_lock:
                    mapping[identifier] = mapped
                    mapping_path.write_text(json.dumps(mapping, indent=2) + '\n')
            mapped = mapping[identifier]
            if not mapped:
                return identifier, '', 0., 'unavailable_mapping'
            path = pdb_directory / (identifier + '.pdb')
            if not path.is_file():
                response = get('https://alphafold.ebi.ac.uk/api/prediction/' + mapped)
                if response is None:
                    return identifier, mapped, 0., 'unavailable_structure'
                models = json.loads(response)
                if not models:
                    return identifier, mapped, 0., 'unavailable_structure'
                pdb = get(models[0]['pdbUrl'])
                if not pdb or not pdb.startswith((b'HEADER', b'TITLE', b'REMARK', b'ATOM')):
                    raise ValueError('Invalid downloaded PDB')
                path.write_bytes(pdb)
            observed = ''.join(AMINO_ACIDS.get(line[17:20].strip(), 'X')
                               for line in path.read_text().splitlines()
                               if line.startswith('ATOM') and line[12:16].strip() == 'CA')
            identity = sequence_identity(observed, expected)
            return identifier, mapped, identity, 'downloaded' if identity >= .95 else 'sequence_mismatch_retained'
        except Exception as error:
            return identifier, '', 0., 'error:' + str(error)

    failures = 0
    with (args.output / 'fetch_report.tsv').open('w', newline='') as stream:
        writer = csv.writer(stream, delimiter='\t')
        writer.writerow(('protein', 'accession', 'sequence_identity', 'status'))
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for result in executor.map(fetch, proteins):
                writer.writerow(result)
                stream.flush()
                failures += result[-1].startswith('error:')
    if failures:
        raise SystemExit('{} requests failed; rerun to retry using cached downloads'.format(failures))


if __name__ == '__main__':
    main()
