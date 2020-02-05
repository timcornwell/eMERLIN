FROM timcornwell/rascil-no-data

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && apt-get install -y \
    libfftw3-dev libboost-dev libboost-date-time-dev \
    libboost-filesystem-dev libboost-system-dev libboost-signals-dev \
    libboost-python-dev libboost-numpy-dev \
    libboost-thread-dev libxml2-dev liblapack-dev libcfitsio-dev libgtkmm-3.0-dev \
    cmake g++ build-essential gfortran libpng-dev casacore-dev \
    python3 python3-dev && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN apt-get update -y && apt-get install -y \
    git && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

WORKDIR /
RUN git clone https://git.code.sf.net/p/aoflagger/code aoflagger
WORKDIR /aoflagger

RUN mkdir build && \
    cd build && \
    cmake ../ && \
    make && \
    make install

RUN cp /usr/local/lib/aoflagger.so /usr/lib/python3.7

WORKDIR /rascil

# Set runtime environment variables.
ENV PYTHONPATH=/rascil:$PYTHONPATH
ENV RASCIL=/rascil

