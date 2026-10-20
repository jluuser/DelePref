"""Uncapped accounting with the existing atomic identity/retry/cache semantics."""
import hashlib
import json
import math
from pathlib import Path

from budget_ledger import BudgetLedger


class PredictionJournal(BudgetLedger):
    """An explicit null limit records predictions without a numerical quota.

    Historical capped protocols and their journals remain unchanged. Dispatch
    ownership, output verification, failed-call accounting and one technical
    retry retain the tested parent implementation.
    """
    def __init__(self, protocol_path, ledger_path):
        self.path = Path(ledger_path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        raw = Path(protocol_path).read_bytes()
        self.protocol_sha = hashlib.sha256(raw).hexdigest()
        config = json.loads(raw)['compute']
        limit = config['structure_call_limit']
        caps = config['structure_categories']
        self.retries = config['structure_retries_max']
        if limit is not None and (type(limit) is not int or limit < 1):
            raise ValueError('invalid prediction count limit')
        if self.retries not in (0, 1) or not caps:
            raise ValueError('invalid journal retry/categories')
        if any(v is not None and (type(v) is not int or v < 0) for v in caps.values()):
            raise ValueError('invalid category count limit')
        self.limit = math.inf if limit is None else limit
        self.caps = {k: math.inf if v is None else v for k,v in caps.items()}
