.. _ERP_galahad_install:

Installation of ERP on galahad
==============================

The eMERLIN_RASCIL_package and RASCIL are already installed on galahad so alll you need to do is set up your
PYTHONPATH::

    export PYTHONPATH=/home/cornwell/Code/eMERLIN_RASCIL_pipeline:/home/cornwell/Code/rascil/:$PYTHONPATH

To install your own version, first install RASCIL on galahad from the instructions at::

    https://timcornwell.gitlab.io/rascil/

and next install eMERLIN_RASCIL_pipeline using git::

    git clone https://github.com/timcornwell/eMERLIN_RASCIL_pipeline.git


.. _feedback: mailto:realtimcornwell@gmail.com
