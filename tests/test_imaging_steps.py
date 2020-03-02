""" Unit processing_components for RASCIL eMERLIN pipeline


"""
import unittest
import numpy
import logging

import argparse
import json
import os
import sys
import time

from erp.pipelines.imaging_steps import *
from erp.pipelines.support import start_eMRP_dict, list_steps, read_inputs, \
    find_run_steps, get_pipeline_version, get_logger, \
    get_defaults, save_obj, exit_pipeline

from rascil.workflows.rsexecute.execution_support import rsexecute

log = logging.getLogger('logger')

log.setLevel(logging.WARNING)

class TestArray_functions(unittest.TestCase):

    def setUp(self) -> None:

def run_pipeline(inputs_file='./inputs.ini', run_steps=[], skip_steps=[]):

        # Create directory structure
        info_dir = './'
        
        # Initialize eMRP dictionary, or continue with previous pipeline configuration if possible:
        eMRP = start_eMRP_dict(info_dir)
        
        logger = get_logger()
        
        start_epoch = time.asctime()
        logger.info(
            "eMERLIN RASCIL imaging pipeline, started at  %s" % start_epoch)
        
        save_obj(eMRP, info_dir + 'eMRP_info.pkl')
        
        # Load default parameters
        assert os.path.isfile('./default_params.json')
        
        defaults_file = './default_params.json'
        logger.info('Loading default parameters from {0}:'.format(defaults_file))
        eMRP['defaults'] = json.loads(open(defaults_file).read())
        
        # Inputs
        assert os.path.exists(inputs_file), 'No inputs file found: {}'.format(inputs_file)
        inputs = read_inputs(inputs_file)
        eMRP['inputs'] = inputs
        
        # Steps to run:
        eMRP['input_steps'] = find_run_steps(eMRP, run_steps, skip_steps)
        
        ## Pipeline processes, inputs are read from the inputs dictionary
        
        eMRP = get_defaults(eMRP, pipeline_path='\.')
        
        initialize_pipeline(eMRP, get_logger=get_logger)
        
        bvis_list = None
        
        if eMRP['input_steps']['list_ms'] > 0:
            list_ms(eMRP)
        
        if eMRP['input_steps']['load_ms'] > 0:
            bvis_list = load_ms(eMRP)
        
        if eMRP['input_steps']['flag'] > 0:
            bvis_list = flag(bvis_list, eMRP)
        
        if eMRP['input_steps']['average_channels'] > 0:
            bvis_list = average_channels(bvis_list, eMRP)
    
        if eMRP['input_steps']['plot_vis'] > 0:
            plot_vis(bvis_list, eMRP)
    
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
            results = rsexecute.compute(results, sync=True)
            if eMRP['input_steps']['write_images'] > 0:
                write_images(eMRP, 'cip', results)
        
        if eMRP['input_steps']['ical'] > 0:
            results = ical(bvis_list, model_list, eMRP)
            results = rsexecute.compute(results, sync=True)
            if eMRP['input_steps']['write_images'] > 0:
                write_images(eMRP, 'ical', results[0:3])
            if eMRP['input_steps']['write_gaintables'] > 0:
                write_gaintables(eMRP, 'ical', results[3])
            if eMRP['input_steps']['write_ms'] > 0:
                bvis_list = apply_calibration(results[3], bvis_list, eMRP)
                if eMRP['input_steps']['combine_spw'] > 0:
                    bvis_list = combine_spw(bvis_list, eMRP)
                write_ms(bvis_list, eMRP)
        
        # Keep important files
        # save_obj(eMRP, info_dir + 'eMRP_info.pkl')
        # os.system('cp eMRP.log {}eMRP.log.txt'.format(info_dir))
        
        finalize_pipeline(eMRP)
        
        return eMRP

if __name__ == '__main__':
    unittest.main()
