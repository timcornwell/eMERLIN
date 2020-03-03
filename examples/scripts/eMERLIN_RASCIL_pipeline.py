"""

"""

import argparse
import json
import os
import sys
import time

from erp.functions.support import start_eMRP_dict, list_steps, read_inputs, \
    find_run_steps, get_pipeline_version, get_logger, get_defaults, save_obj, exit_pipeline
from erp.functions.pipelines import run_erp_ical

current_version = "0.0.1"


def cli_parser():
    '''This function parses and returns arguments passed in'''
    # Assign description to the help doc
    description = 'e-MERLIN RASCIL pipeline. Visit: https://github.com/timcornwell/eMERLIN_RASCIL_pipeline'
    usage = 'python eMERLIN_RASCIL_pipeline/eMERLIN_RASCIL_pipeline.py -r [steps]'
    parser = argparse.ArgumentParser(description=description, usage=usage)
    parser.add_argument('-i', '--inputs', dest='inputs_file',
                        help='Inputs file to use. Default is inputs.ini',
                        default='./inputs.ini')
    parser.add_argument('-r', '--run-steps', dest='run_steps',
                        type=str, nargs='+',
                        help='Whitespace separated list of steps to run. ' \
                             'Apart from individual steps, it also accepts "all", ' \
                             '"pre_processing" and "calibration"',
                        default=[])
    parser.add_argument('-s', '--skip-steps', dest='skip_steps',
                        type=str, nargs='+',
                        help='Whitespace separated list of steps to skip',
                        default=[])
    parser.add_argument('-l', '--list-steps', dest='list_steps',
                        action='store_true',
                        help='Show list of available steps and exit')
    parser.add_argument('-c', help='Ignore, needed for casa')

    return parser.parse_args()


def run_pipeline(inputs_file='./inputs.ini', run_steps=[], skip_steps=[]):
    # Create directory structure
    info_dir = '../../'
    
    # Initialize eMRP dictionary, or continue with previous pipeline configuration if possible:
    eMRP = start_eMRP_dict(info_dir)
    
    # Get git info about pipeline version
    try:
        branch, short_commit = get_pipeline_version(pipeline_path)
    except:
        branch, short_commit = 'unknown', 'unknown'
    pipeline_version = current_version
    
    logger = get_logger()
    
    start_epoch = time.asctime()
    logger.info(
        "eMERLIN RASCIL imaging pipeline, started at  %s" % start_epoch)
    
    logger.info('Starting pipeline')
    logger.info('Running pipeline from:')
    logger.info('{}'.format(pipeline_path))
    logger.info('Pipeline version: {}'.format(pipeline_version))
    logger.info('Using github branch: {}'.format(branch))
    logger.info('github last commit: {}'.format(short_commit))
    logger.info('This log uses UTC times')
    eMRP['pipeline_path'] = pipeline_path
    eMRP['pipeline_version'] = pipeline_version
    save_obj(eMRP, info_dir + 'eMRP_info.pkl')
    
    # Load default parameters
    #    if os.path.isfile('../../default_params.json'):
    #        defaults_file = '../../default_params.json'
    #    if os.path.isfile('../../default_params.json'):
    #        defaults_file = '../../default_params.json'
    #    else:
    defaults_file = 'default_params.json'
    logger.info('Loading default parameters from {0}:'.format(defaults_file))
    eMRP['defaults'] = json.loads(open(defaults_file).read())
    
    # Steps to run:
    eMRP['input_steps'] = find_run_steps(eMRP, run_steps, skip_steps)
    
    # Pipeline processes, inputs are read from the inputs dictionary
    
    eMRP = get_defaults(eMRP)

    run_erp_ical(eMRP)

    stop_epoch = time.asctime()
    logger.info("eMERLIN RASCIL imaging pipeline, started at  %s" % start_epoch)
    logger.info("eMERLIN RASCIL imaging pipeline, finished at %s" % stop_epoch)
    
    return eMRP

if __name__ == "__main__":
    
    try:
        pipeline_filename = sys.argv[sys.argv.index('-c') + 1]
        pipeline_path = os.path.abspath(os.path.dirname(pipeline_filename))
    except:
        pipeline_path = '../..'
        pass
    
    if pipeline_path[-1] != '/':
        pipeline_path = pipeline_path + '/'
    sys.path.append(pipeline_path)
    
    eMRP = start_eMRP_dict()
    
    args = cli_parser()
    if args.list_steps:
        list_steps()
    else:
        run_pipeline(inputs_file=args.inputs_file,
                     run_steps=args.run_steps,
                     skip_steps=args.skip_steps)
