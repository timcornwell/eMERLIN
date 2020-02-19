"""

"""

import argparse
import json
import os
import sys
import time

from erp.pipelines.imaging_steps import *
from erp.pipelines.support import start_eMRP_dict, list_steps, read_inputs, \
    find_run_steps, get_pipeline_version, get_logger, \
    get_defaults, save_obj, exit_pipeline

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
    
    ## Pipeline processes, inputs are read from the inputs dictionary
    
    eMRP = get_defaults(eMRP, pipeline_path='\.')
    
    initialize_pipeline(eMRP, get_logger=get_logger)
    
    bvis_list = None
    
    if eMRP['input_steps']['ms_list'] > 0:
        ms_list(eMRP)
    
    if eMRP['input_steps']['ms_load'] > 0:
        bvis_list = ms_load(eMRP)
    
    if eMRP['input_steps']['flag'] > 0:
        bvis_list = flag(bvis_list, eMRP)
    
    if eMRP['input_steps']['plot_vis'] > 0:
        plot_vis(bvis_list, eMRP)
    
    if eMRP['input_steps']['average_channels'] > 0:
        bvis_list = average_channels(bvis_list, eMRP)
    
    if eMRP['input_steps']['combine_spw'] > 0:
        bvis_list = combine_spw(bvis_list, eMRP)
    
    if eMRP['input_steps']['convert_stokesI'] > 0:
        bvis_list = convert_stokesI(bvis_list, eMRP)
    
    if eMRP['input_steps']['get_advice'] > 0:
        advice = get_advice(bvis_list, eMRP)
    
    model_list = list()
    if eMRP['input_steps']['create_images'] > 0:
        model_list = create_images(bvis_list, eMRP)
    
    if eMRP['input_steps']['weight'] > 0:
        bvis_list = weight(bvis_list, model_list, eMRP)
    
    if eMRP['input_steps']['cip'] > 0:
        results = cip(bvis_list, model_list, eMRP)
        if eMRP['input_steps']['write_images'] > 0:
            write_images(eMRP, bvis_list, 'cip', results)
    
    if eMRP['input_steps']['ical'] > 0:
        results = ical(bvis_list, model_list, eMRP)
        if eMRP['input_steps']['write_images'] > 0:
            write_images(eMRP, bvis_list, 'ical', results[0:3])
        if eMRP['input_steps']['write_gaintables'] > 0:
            write_gaintables(eMRP, bvis_list, 'ical', results[3])
        if eMRP['input_steps']['ms_save'] > 0:
            apply_calibration(results[3], bvis_list, eMRP)
        if eMRP['input_steps']['ms_save'] > 0:
            ms_save(bvis_list, eMRP)
    
    # Keep important files
    # save_obj(eMRP, info_dir + 'eMRP_info.pkl')
    # os.system('cp eMRP.log {}eMRP.log.txt'.format(info_dir))
    
    finalize_pipeline(eMRP)
    
    stop_epoch = time.asctime()
    logger.info(
        "eMERLIN RASCIL imaging pipeline, started at  %s" % start_epoch)
    logger.info(
        "eMERLIN RASCIL imaging pipeline, finished at %s" % stop_epoch)
    
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
        run_pipeline(inputs_file=args.inputs_file,
                     run_steps=args.run_steps,
                     skip_steps=args.skip_steps)
