"""
Common only for examples
"""
from dataclasses import dataclass
import json
import tempfile
from loguru import logger
import os
logger.add("/home/ms314/1/phd/src/pccf/log.txt")


@dataclass
class SignalSettings:
    mu0: float
    mu1: float
    sigma: float
    half_len: int


def save_json_to_temp_file(results_dict,
                           prefix='results',
                           temp_dir='/home/ms314/1/phd/src/pccf/temp'):
    with tempfile.NamedTemporaryFile(dir=temp_dir,
                                     prefix=prefix,
                                     delete=False,
                                     mode='w') as fp:
        json.dump(results_dict, fp, indent=4)
        save_path = fp.name
        logger.info(f"Save simulation results into {fp.name}")
    return save_path
