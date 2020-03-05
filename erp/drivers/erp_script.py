"""

"""

import logging
import argparse
import json
import time
from functools import partial

from erp.functions.pipelines import run_erp

current_version = "0.0.1"


def get_logger(log_format='%(asctime)s | %(levelname)s | %(message)s',
               date_format='%Y-%m-%d %H:%M:%S',
               log_name='logger',
               log_file_info='./erp.log'):
    """ Get a logger

    :param log_format:
    :param date_format:
    :param log_name:
    :param log_file_info:
    :return:
    """
    log = logging.getLogger(log_name)
    log_formatter = logging.Formatter(fmt=log_format, datefmt=date_format)
    logging.Formatter.converter = time.gmtime
    
    # comment this to suppress console output
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)
    log.addHandler(stream_handler)
    
    # eMRP.log with pipeline log
    file_handler_info = logging.FileHandler(log_file_info, mode='a')
    file_handler_info.setFormatter(log_formatter)
    file_handler_info.setLevel(logging.DEBUG)
    log.addHandler(file_handler_info)
    
    log.setLevel(logging.INFO)
    return log


def cli_parser():
    '''This function parses and returns arguments passed in
    
    '''
    # Assign description to the help doc
    description = 'e-MERLIN RASCIL pipeline. Visit: https://github.com/timcornwell/eMERLIN_RASCIL_pipeline'
    usage = 'python eMERLIN_RASCIL_pipeline/eMERLIN_RASCIL_pipeline.py [-i erp_params.json]'
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
    p_get_logger = partial(get_logger, log_file_info=log_file_info)
    logger = p_get_logger()

    start_epoch = time.asctime()
    logger.info("eMERLIN RASCIL imaging pipeline, started at  %s" % start_epoch)
    logger.info('Loading default parameters from {0}:'.format(erp_params_file))
    run_erp(erp_params, p_get_logger)
    
    stop_epoch = time.asctime()
    logger.info("eMERLIN RASCIL imaging pipeline, started at  %s" % start_epoch)
    logger.info("eMERLIN RASCIL imaging pipeline, finished at %s" % stop_epoch)
    
    exit(0)
