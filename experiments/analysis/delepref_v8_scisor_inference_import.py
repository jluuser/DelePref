"""Import the inference-only SCISOR class without optional DeepSpeed training ops."""
import importlib
import sys


def load_native_class():
    import pkg_resources
    original=pkg_resources.require
    def inference_require(*requirements):
        parsed=list(pkg_resources.parse_requirements(requirements))
        blocked=[r for r in parsed if r.key=='deepspeed']
        if blocked:raise pkg_resources.DistributionNotFound(blocked[0])
        return original(*requirements)
    if 'deepspeed' in sys.modules and sys.modules['deepspeed'] is not None:
        raise RuntimeError('DeepSpeed already imported; inference compatibility must run before Lightning')
    sys.modules['deepspeed']=None
    pkg_resources.require=inference_require
    try:
        cls=importlib.import_module('SCISOR.shortening_scud').ShorteningSCUD
        plugin=importlib.import_module('pytorch_lightning.strategies.deepspeed')
        if bool(plugin._DEEPSPEED_AVAILABLE):raise RuntimeError('optional DeepSpeed training plugin still active')
        return cls
    finally:
        pkg_resources.require=original
