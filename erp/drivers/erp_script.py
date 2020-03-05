"""

"""

__all__ = ['cli_parser']

import logging
import argparse
import json
import time
from functools import partial

from erp.functions.pipelines import run_erp
from erp.drivers.driver_support import get_logger

current_version = "0.0.1"

def cli_parser():
    '''This function parses and returns arguments passed in
    
    '''
    # Assign description to the help doc
    description = 'e-MERLIN RASCIL pipeline. Visit: https://github.com/timcornwell/eMERLIN_RASCIL_pipeline'
    usage = 'python erp_script.py [--inputs erp_params.json]'
    parser = argparse.ArgumentParser(description=description, usage=usage)
    parser.add_argument('-p', '--params', dest='erp_params',
                        help='JSON file containing parameters',
                        default='./erp_params.json')
    return parser.parse_args()

if __name__ == "__main__":
    
    args = cli_parser()
    erp_params_file = args.erp_params
    
    erp_params = json.loads(open(erp_params_file).read())
    
    # We need to ensure that all workers are using the same logger so we pass
    # a function to be run on each worker.
    log_file_info = "{log_name}/erp.log".format(log_name=erp_params['configure']['results_directory'])
    # p_get_logger = partial(get_logger, log_file_info=log_file_info)
    p_get_logger = get_logger
    logger = p_get_logger()

    start_epoch = time.asctime()
    logger.info("eMERLIN RASCIL imaging pipeline, started at  %s" % start_epoch)
    logger.info('Loading default parameters from {0}:'.format(erp_params_file))
    run_erp(erp_params, get_logger)
    
    stop_epoch = time.asctime()
    logger.info("eMERLIN RASCIL imaging pipeline, started at  %s" % start_epoch)
    logger.info("eMERLIN RASCIL imaging pipeline, finished at %s" % stop_epoch)
    
    exit(0)
