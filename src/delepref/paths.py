"""Portable asset paths; explicit absolute research paths remain supported."""
from pathlib import Path
import os


def resolve_path(value=None, *, environment=None, default=None, base=None, exists=False):
    raw = value if value is not None else os.environ.get(environment, '') if environment else None
    raw = raw or default
    if raw is None:
        raise ValueError('An explicit path or {} is required'.format(environment or 'asset path'))
    path = Path(raw).expanduser()
    if not path.is_absolute() and base is not None:
        path = Path(base) / path
    path = path.resolve()
    if exists and not path.exists():
        raise FileNotFoundError(path)
    return path


def packaged_asset(*parts):
    return Path(__file__).resolve().parent.joinpath('assets', *parts)
