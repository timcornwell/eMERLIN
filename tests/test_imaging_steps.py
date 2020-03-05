""" Unit processing_components for RASCIL eMERLIN pipeline

"""
import json
import logging
import unittest

from erp.functions.pipelines import run_erp

log = logging.getLogger('logger')
log.setLevel(logging.WARNING)

class TestERP_functions(unittest.TestCase):
    
    def setUp(self) -> None:
        pass
    
    def test_pipeline(self):
        defaults_file = 'test_params.json'
        log.info('Loading default parameters from {0}:'.format(defaults_file))
        with open(defaults_file) as df:
            json_params = json.loads(df.read())
        run_erp(json_params)


if __name__ == '__main__':
    unittest.main()
