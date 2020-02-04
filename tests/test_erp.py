""" Unit tests for ERP

"""
import sys
import unittest
import logging

log = logging.getLogger(__name__)

log.setLevel(logging.DEBUG)
handler = logging.StreamHandler(stream=sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
log.addHandler(handler)

from erp.erp_functions import erp_list_ms, erp_load_ms

class TestERP(unittest.TestCase):
    
    def setUp(self):
        
        small = True
        if small:
            self.ms_name = "data/3C277.1C.ms"
        else:
            self.ms_name = "data/3C277.1_avg.ms"

    def tearDown(self):
        pass
    
    def test_list_ms(self):

        erp_list_ms(self.ms_name)
        
    def test_load_ms(self):
        
        bvis_list = erp_load_ms(self.ms_name)
        for bvis in bvis_list:
            log.info(bvis)

if __name__ == '__main__':
    unittest.main()

