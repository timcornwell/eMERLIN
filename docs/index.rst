.. _documentation_master:

.. toctree::

eMERLIN RASCIL Pipeline (ERP)
#############################

This package contains a pipeline for processing eMERLIN data. It currently (March 2020) supports imaging of previously
calibrated data, along with selfcalibration. It uses the RASCIL package for the processing: https://timcornwell.gitlab.io/rascil/

The structure of the ERP package is:

- A core set of simple functions are located in erp/functions/imaging_steps.py. These are
  convenient wrappers around RASCIL functions. The input parameters are passed by a dictionary
  called eMRP that is read from a JSON format file located in the local directory.
- A pipeline function runs these in a predetermined sequence, roughly load ms, average,
  imaging with selfcalibration, save results.

RASCIL uses the Dask package (https://dask.org) for distributing the processing over many cores or nodes. This allows
scaling from a laptop to a cluster. For a pipeline running locally, the Dask diagnostics can be seen at
http://127.0.0.1:8787/status

.. toctree::
   :maxdepth: 1

   ERP_install
   ERP_api

* :ref:`genindex`
* :ref:`modindex`

.. _feedback: mailto:realtimcornwell@gmail.com
