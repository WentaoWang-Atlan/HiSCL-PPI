"""Check source syntax and prevent accidental submission of generated assets."""
import ast
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    required = ('README.md', 'LICENSE', 'MAPE-PPI/LICENSE', 'THIRD_PARTY_NOTICES.md',
                '.gitignore', '.gitattributes',
                'environment.yml', 'configs/backbone.json', 'configs/method.json')
    for name in required:
        if not (ROOT / name).is_file():
            raise SystemExit('Required release file is missing: ' + name)
    result = subprocess.run(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'],
                            cwd=ROOT, capture_output=True)
    if result.returncode == 0:
        files = [ROOT / name.decode() for name in result.stdout.split(b'\0') if name]
    else:
        files = [path for path in ROOT.rglob('*') if path.is_file()
                 and not any(part in ('.git', '__pycache__', '.ruff_cache', '.venv')
                             for part in path.relative_to(ROOT).parts)]
    prohibited = {'.pt', '.pth', '.ckpt', '.pkl', '.pickle', '.npy', '.npz', '.pdb',
                  '.cif', '.m8', '.log', '.zip', '.pem', '.key', '.pyc'}
    private_root = re.compile(r'(?:/home/|/Users/)[A-Za-z0-9_.-]+')
    for path in files:
        relative = path.relative_to(ROOT)
        if path.is_symlink():
            raise SystemExit('Do not submit symlinks: ' + str(relative))
        if path.suffix in prohibited or relative.parts[0] in ('data', 'runs', 'logs', 'structures', 'results'):
            raise SystemExit('Generated or private asset in release: ' + str(relative))
        if path.stat().st_size > 2_000_000:
            raise SystemExit('Unexpected large source file: ' + str(relative))
        text = path.read_text(encoding='utf-8')
        if private_root.search(text):
            raise SystemExit('Personal absolute path in ' + str(relative))
        if path.suffix in ('.py', '.sh', '.json') and re.search(r'[\u4e00-\u9fff]', text):
            raise SystemExit('Non-English source text in ' + str(relative))
        if path.suffix == '.py':
            ast.parse(text, filename=str(relative))
        elif path.suffix == '.json':
            json.loads(text)
        elif path.suffix == '.sh':
            subprocess.run(['bash', '-n', str(path)], check=True)
    for name in ('CITATION.cff', 'CITATION.bib'):
        if (ROOT / name).exists():
            raise SystemExit('Author citation metadata is excluded from the anonymous release')
    readme_text = (ROOT / 'README.md').read_text()
    if re.search(r'(?im)^\*\*Authors?:\*\*|^authors?:', readme_text):
        raise SystemExit('Remove author metadata from the anonymous README')
    method = json.loads((ROOT / 'configs/method.json').read_text())
    if (method['batch_size_proteins'], method['r_cap'], method['lambda_hcl'], method['lambda_tm']) != (128, 12000, 1.0, 0.3):
        raise SystemExit('The distributed default configuration differs from the release protocol')
    readme = (ROOT / 'README.md').read_text()
    if 'ACCL' not in readme or 'not\nimplemented in this version' not in readme:
        raise SystemExit('README must explain the deferred ACCL implementation')
    print('Release checks passed for {} source files.'.format(len(files)))


if __name__ == '__main__':
    main()
