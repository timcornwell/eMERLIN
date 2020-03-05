#!/bin/bash
pwd
export PYTHONPATH=/home/cornwell/Code/eMERLIN_RASCIL_pipeline:/home/cornwell/Code/rascil
python3 erp/drivers/erp_script.py --params eMERLIN_3C277_1.json
