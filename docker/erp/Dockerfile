FROM timcornwell/rascil-no-data

RUN apt-get update -y && apt-get install -y \
    libfftw3-dev libboost-dev libboost-date-time-dev \
    libboost-filesystem-dev libboost-system-dev libboost-signals-dev \
    libboost-thread-dev libxml2-dev liblapack-dev libcfitsio-dev libgtkmm-3.0-dev \
    cmake g++ build-essential gfortran libpng-dev casacore-dev && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN git clone https://git.code.sf.net/p/aoflagger/code aoflagger

WORKDIR /rascil/aoflagger

RUN mkdir build && \
    cd build && \
    cmake ../ && \
    make && \
    make install

RUN cp /usr/local/lib/aoflagger.so /opt/conda/lib/python3.7/site-packages

WORKDIR /rascil


