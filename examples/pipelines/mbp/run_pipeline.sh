#!/bin/bash
pwd
export PYTHONPATH=/Users/timcornwell/Code/eMERLIN_RASCIL_pipeline:/Users/timcornwell/Code/rascil
python3 ../../scripts/eMERLIN_RASCIL_pipeline.py -i inputs.ini -r load_ms flag average_channels \
convert_stokesI create_images weight ical write_images write_gaintables apply_calibration write_ms

