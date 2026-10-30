import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recomb_confirm_v2_E10_timing import observe_process, EVENT


class ActualRequestBoundaryTests(unittest.TestCase):
    def test_process_start_and_real_output_io_are_timed_but_between_request_idle_is_excluded(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            code = '''import sys,time,json
from pathlib import Path
time.sleep(.06)
Path('cold_output.txt').write_text('actual first output')
print(PREFIX+json.dumps(dict(scenario='cold',measurement=dict(scenario='cold'))),flush=True)
assert sys.stdin.readline().strip()=='RUN_CACHED_REQUEST'
time.sleep(.04)
Path('cached_output.txt').write_text('actual cached output')
print(PREFIX+json.dumps(dict(scenario='cached',measurement=dict(scenario='cached'))),flush=True)
'''.replace('PREFIX', repr(EVENT))
            started = time.perf_counter()
            rc, result = observe_process([sys.executable, '-u', '-c', code], directory, dict(os.environ),
                directory/'log.txt', before_cached_request=lambda: time.sleep(.15))
            elapsed = time.perf_counter()-started
            self.assertEqual(rc, 0)
            self.assertEqual([r['scenario'] for r in result], ['cold', 'cached'])
            self.assertEqual((directory/'cold_output.txt').read_text(), 'actual first output')
            self.assertEqual((directory/'cached_output.txt').read_text(), 'actual cached output')
            self.assertGreaterEqual(result[0]['parent_observed_request_seconds'], .06)
            self.assertGreaterEqual(result[1]['parent_observed_request_seconds'], .04)
            self.assertGreater(elapsed-sum(r['parent_observed_request_seconds'] for r in result), .12)

    def test_failed_child_is_reported_without_fabricating_second_measurement(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            rc, result = observe_process([sys.executable, '-c', 'raise SystemExit(3)'], path,
                dict(os.environ), path/'failure.log')
            self.assertEqual((rc, result), (3, []))


if __name__ == '__main__':
    unittest.main()
