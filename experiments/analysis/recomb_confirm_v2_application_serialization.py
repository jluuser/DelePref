"""CPU readout retry: preserve metrics and normalize NumPy JSON scalars only."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

FREEZE = Path('configs/recomb_confirm_v2_application_serialization_freeze.json')
RAW = Path('reports/recomb_confirm_v2_application_evaluation_serialization_v5')
CORRECTED = Path('reports/recomb_confirm_v2_application_evaluation_missingness_serialization_v5')
REPORT = Path('reports/recomb_confirm_v2_application_final_report_serialization_v5')


def json_native(value):
    if isinstance(value, np.generic):
        return json_native(value.item())
    if isinstance(value, dict):
        return {json_native(k): json_native(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_native(v) for v in value]
    return value


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verify_completion(directory, phase):
    done = read(directory/'completion.json')
    if done['phase'] != phase or done['logical'] != {'main': 980, 'all': 2160}[phase]:
        raise ValueError('Frozen logical denominator differs')
    for name, expected in done['outputs'].items():
        if sha(directory/name) != expected:
            raise ValueError('Completed readout artifact changed: '+name)
    return done


def run(root, phase, expected_freeze, report=False):
    if not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Serialization readout requires CPU Slurm')
    if sha(root/FREEZE) != expected_freeze:
        raise ValueError('Serialization adapter freeze changed')
    frozen = read(root/FREEZE)
    for name, expected in frozen['files'].items():
        if sha(root/name) != expected:
            raise ValueError('Frozen readout source changed: '+name)
    import recomb_confirm_v2_application_evaluate_hardware_v3 as parent
    import recomb_confirm_v2_application_missingness as missingness
    from recomb_confirm_v2_diagnostics import write as original_write

    def write(path, value):
        original_write(path, json_native(value))

    # These are process-local I/O changes; original metric/statistical functions remain intact.
    parent.OUT = RAW
    parent.write = write
    directory = root/RAW/phase
    if not (directory/'completion.json').exists():
        parent.evaluate(root, phase)
    done = verify_completion(directory, phase)
    done.update(serialization_source_freeze_sha256=expected_freeze,
                serialization_only_numpy_scalars_to_native=True,
                original_metric_and_statistical_functions_unchanged=True,
                no_new_model_calls=True, no_new_structure_predictions=True)
    write(directory/'completion.json', done)

    missingness.PARENT = RAW
    missingness.OUT = CORRECTED
    missingness.write = write
    missingness.run(root, phase, report=False)
    corrected = verify_completion(root/CORRECTED/phase, phase)
    corrected.update(serialization_source_freeze_sha256=expected_freeze)
    write(root/CORRECTED/phase/'completion.json', corrected)
    if report:
        if phase != 'all':
            raise ValueError('Final report requires all2160 logical rows')
        import recomb_confirm_v2_application_repeat_report_hardware_v3 as presentation
        presentation.EVALUATION = CORRECTED/'all'
        presentation.OUT = REPORT
        presentation.write = write
        presentation.run(root)
        path = root/REPORT/'completion.json'
        result = read(path)
        result.update(serialization_source_freeze_sha256=expected_freeze,
                      missingness_source_freeze_sha256=sha(root/missingness.FZ),
                      raw_parent_all_metrics_completion_sha256=sha(directory/'completion.json'),
                      corrected_parent_all_metrics_completion_sha256=sha(root/CORRECTED/phase/'completion.json'))
        write(path, result)
    print(json.dumps(corrected, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--phase', choices=['main', 'all'], required=True)
    parser.add_argument('--freeze-sha256', required=True)
    parser.add_argument('--report', action='store_true')
    args = parser.parse_args()
    run(args.root, args.phase, args.freeze_sha256, args.report)
