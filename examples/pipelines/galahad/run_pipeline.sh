#!/bin/bash
pwd
python3 ../../scripts/eMERLIN_RASCIL_pipeline.py -i inputs.ini -r load_ms average_channels \
convert_stokesI create_images weight ical write_images write_gaintables write_ms

