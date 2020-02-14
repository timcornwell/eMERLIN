"""

"""

import logging
import io
import os
import sys
import collections
import time
import argparse
import json

from erp.pipelines.support import start_eMRP_dict, list_steps, read_inputs, \
    find_run_steps, get_pipeline_version, \
    get_defaults, load_obj, save_obj, exit_pipeline
from erp.pipelines.imaging_steps import *

current_version = "0.0.1"

def run_pipeline(inputs_file='./inputs.ini',
                 run_steps=[], skip_steps=[]):
    # Create directory structure
    info_dir = './'
    
    # Initialize eMRP dictionary, or continue with previous pipeline configuration if possible:
    eMRP = start_eMRP_dict(info_dir)
    
    # Get git info about pipeline version
    try:
        branch, short_commit = \
            get_pipeline_version(pipeline_path)
    except:
        branch, short_commit = 'unknown', 'unknown'
    pipeline_version = current_version
    
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
    if os.path.isfile('./default_params.json'):
        defaults_file = './default_params.json'
    else:
        defaults_file = pipeline_path + 'default_params.json'
    logger.info('Loading default parameters from {0}:'.format(defaults_file))
    eMRP['defaults'] = json.loads(open(defaults_file).read())
    
    # Inputs
    if os.path.exists(inputs_file):
        inputs = read_inputs(inputs_file)
        eMRP['inputs'] = inputs
    else:
        logger.critical('No inputs file found: {}'.format(inputs_file))
        exit_pipeline()
    
    # Steps to run:
    eMRP['input_steps'] = find_run_steps(eMRP, run_steps, skip_steps)
    
    ##################################
    ###  LOAD AND PREPROCESS DATA  ###
    ##################################
    
    ## Pipeline processes, inputs are read from the inputs dictionary
    
    steps = list_steps()
    
    eMRP = get_defaults(eMRP, pipeline_path='\.')
    
    bvis_list = load_ms(eMRP)
    
    if steps['flag']:
        bvis_list = flag(bvis_list, eMRP)
    
    if steps['average_channels']:
        bvis_list = average_channels(bvis_list, eMRP)
    
    if steps['get_advice']:
        advice = get_advice(bvis_list, eMRP)
    
    model_list = list()
    if steps['create_images']:
        model_list = create_images(bvis_list, eMRP)
    
    if steps['weight']:
        bvis_list = weight(bvis_list, model_list, eMRP)
    
    if steps['cip']:
        results = cip(bvis_list, model_list, eMRP)
        write_results(eMRP, bvis_list, 'cip', results)
    
    if steps['ical']:
        results = ical(bvis_list, model_list, eMRP)
        write_results(eMRP, bvis_list, 'ical', results)
    
    # Keep important files
    save_obj(eMRP, info_dir + 'eMRP_info.pkl')
    os.system('cp eMRP.log {}eMRP.log.txt'.format(info_dir))
    os.system('cp casa_eMRP.log {}casa_eMRP.log.txt'.format(info_dir))
    
    try:
        os.system('mv casa-*.log *.last ./logs')
        logger.info('Moved casa-*.log *.last to ./logs')
    except:
        pass
    logger.info('Pipeline finished')
    logger.info('#################')
    
    return eMRP


def get_args():
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
                        help='Whispace separated list of steps to skip',
                        default=[])
    parser.add_argument('-l', '--list-steps', dest='list_steps',
                        action='store_true',
                        help='Show list of available steps and exit')
    parser.add_argument('-c', help='Ignore, needed for casa')
    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    
    cwd = os.getcwd()
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler('%s/eMERLIN_test.log' % cwd))
    
    logging.basicConfig(filename='%s/eMERLIN_test.log' % cwd,
                        filemode='w',
                        format='%(date)s %(asctime)s.%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)
    
    logger.info("Logging to %s/eMERLIN_test.log" % cwd)
    
    try:
        pipeline_filename = sys.argv[sys.argv.index('-c') + 1]
        pipeline_path = os.path.abspath(os.path.dirname(pipeline_filename))
    except:
        pipeline_path = '.'
        pass
    
    if pipeline_path[-1] != '/':
        pipeline_path = pipeline_path + '/'
    sys.path.append(pipeline_path)
    
    eMRP = start_eMRP_dict()
    
    args = get_args()
    if args.list_steps:
        list_steps()
    else:
        # Setup logger
        run_pipeline(inputs_file=args.inputs_file,
                     run_steps=args.run_steps,
                     skip_steps=args.skip_steps)


