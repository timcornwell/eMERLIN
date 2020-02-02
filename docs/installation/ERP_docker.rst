
Running ERP under docker
************************

docker-compose provides a simple way to create a local cluster of a Dask scheduler and a number of workers.
To scale to e.g. 4 dask workers::

    cd docker
    docker-compose --scale worker=4 up

The scheduler and 4 worker should now be running. To connect to the cluster, run the following into another window::

    docker run -it --network host timcornwell/erp

