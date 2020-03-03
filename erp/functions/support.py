""" Support functions needed for RASCIL pipeline

"""

__all__ = ['makedir', 'rmdir', 'rmfile', 'mvdir', 'exit_pipeline',
           'get_pipeline_version', 'read_inputs', 'find_run_steps', 'info_start_steps',
           'list_of_steps', 'list_steps', 'start_eMRP_dict', 'get_logger', 'get_defaults',
           'save_obj', 'load_obj']

import collections
import configparser
import json
import os
import pickle
import shutil
import sys
import time
from ast import literal_eval
import logging

logger = logging.getLogger('logger')

def makedir(pathdir):
    try:
        os.mkdir(pathdir)
        logger.info('Create directory: {}'.format(pathdir))
    except:
        logger.debug('Cannot create directory: {}'.format(pathdir))
        pass


def rmdir(pathdir, message='Deleted:'):
    if os.path.exists(pathdir):
        try:
            shutil.rmtree(pathdir)
            logger.info('{0} {1}'.format(message, pathdir))
        except:
            logger.debug('Could not delete: {0} {1}'.format(message, pathdir))
            pass


def rmfile(pathdir, message='Deleted:'):
    if os.path.exists(pathdir):
        try:
            os.remove(pathdir)
            logger.info('{0} {1}'.format(message, pathdir))
        except:
            logger.debug('Could not delete: {0} {1}'.format(message, pathdir))
            pass


def mvdir(pathdir, outpudir):
    if os.path.exists(pathdir):
        try:
            shutil.move(pathdir, outpudir)
            logger.info('Moved: {0} {1}'.format(pathdir, outpudir))
        except:
            logger.debug('Could not move: {0} {1}'.format(pathdir, outpudir))
            pass


def exit_pipeline(eMRP='', info_dir='./'):
    #os.system('cp eMRP.log {}eMRP.log.txt'.format(info_dir))
    logger.info('Now quitting')
    sys.exit()


def get_pipeline_version(pipeline_path):
    headfile = pipeline_path + '.git/HEAD'
    branch = open(headfile, 'rb').readlines()[0].strip().split('/')[-1]
    commit = \
    open(pipeline_path + '.git/refs/heads/' + branch, 'rb').readlines()[
        0].strip()
    short_commit = commit[:7]
    return branch, short_commit


def read_inputs(inputs_file):
    if not os.path.isfile(inputs_file):
        logger.critical('inputs file: {} not found'.format(inputs_file))
        sys.exit()
    config_raw = configparser.RawConfigParser()
    config_raw.read(inputs_file)
    config = config_raw._sections
    for key in config.keys():
        # config[key].pop('__name__')
        for key2 in config[key].keys():
            try:
                config[key][key2] = literal_eval(config[key][key2])
            except ValueError:
                pass
            except SyntaxError:
                pass
    inputs = config['inputs']
    for key, value in inputs.items():
        logger.info('{0:10s}: {1}'.format(key, value))
    return config['inputs']


def find_run_steps(eMRP, run_steps, skip_steps=[]):
    if run_steps == '': run_steps = []
    if skip_steps == '': skip_steps = []
    logger.info('Step selection')
    logger.info('run_steps : {}'.format(run_steps))
    logger.info('skip_steps: {}'.format(skip_steps))
    
    all_steps = list_of_steps()
    
    # Populate list of steps selected
    step_list = []
    if 'all' in run_steps:
        step_list += all_steps
        run_steps.remove('all')
    step_list += run_steps
    
    # Check if all are valid steps:
    wrong_steps = [s for s in step_list if s not in all_steps]
    if wrong_steps != []:
        ws = ', '.join(wrong_steps)
        logger.critical('Not available step(s) to run: {0}'.format(ws))
        exit_pipeline(eMRP='')
    
    wrong_steps = [s for s in skip_steps if s not in all_steps]
    if wrong_steps != []:
        ws = ', '.join(wrong_steps)
        logger.critical('Not available step(s) to skip: {0}'.format(ws))
        exit_pipeline(eMRP='')
    
    # Remove skipped steps:
    for skip_step in skip_steps:
        if skip_step != '':
            step_list.remove(skip_step)
    
    # Define final step dictionary:
    logger.info('Sorted list of steps to execute:')
    input_steps = collections.OrderedDict()
    for s in all_steps:
        if s in step_list:
            logger.info('{0:16s}: {1}'.format(s, eMRP['defaults']['global'][s]))
            input_steps[s] = eMRP['defaults']['global'][s]
        elif s not in step_list:
            logger.info('{0:16s}: {1}'.format(s, 0))
            input_steps[s] = 0
        else:
            pass
    
    return input_steps


def info_start_steps():
    default_value = [0, 0, '']
    all_steps = list_of_steps()
    
    steps = collections.OrderedDict()
    steps['start_pipeline'] = default_value
    for s in all_steps:
        steps[s] = default_value
    return steps


def list_of_steps():
    import erp.functions.imaging_steps as imsteps
    imaging_steps = imsteps.__all__
    return imaging_steps


def list_steps():
    all_steps = list_of_steps()
    print('\nimaging')
    for s in all_steps:
        print('    {}'.format(s))
    sys.exit()


def start_eMRP_dict(info_dir='./'):
    try:
        eMRP = load_obj(info_dir + 'eMRP_info.pkl')
    except:
        eMRP = collections.OrderedDict()
        eMRP['steps'] = info_start_steps()
    return eMRP


def get_logger(LOG_FORMAT='%(asctime)s | %(levelname)s | %(message)s',
               DATE_FORMAT='%Y-%m-%d %H:%M:%S',
               LOG_NAME='logger',
               LOG_FILE_INFO='eMRP.log'):
    log = logging.getLogger(LOG_NAME)
    log_formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)
    logging.Formatter.converter = time.gmtime
    
    # comment this to suppress console output
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)
    log.addHandler(stream_handler)
    
    # eMRP.log with pipeline log
    file_handler_info = logging.FileHandler(LOG_FILE_INFO, mode='a')
    file_handler_info.setFormatter(log_formatter)
    file_handler_info.setLevel(logging.DEBUG)
    log.addHandler(file_handler_info)
    
    log.setLevel(logging.INFO)
    return log


def get_defaults(eMRP):
    defaults_file = './default_params.json'
    
    logger.info('Loading default parameters from {0}:'.format(defaults_file))
    eMRP['defaults'] = json.loads(open(defaults_file).read())
    return eMRP


def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)
