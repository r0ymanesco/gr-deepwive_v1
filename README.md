# DeepWiVe OOT module
This repo is the out-of-tree (OOT) GNURadio module for DeepWiVe.

## Prerequisits
1. gnuradio (3.8.x)
2. gnuradio-build-deps
3. python (3.6.x)
4. pytorch (>1.7)
5. numpy
6. Miniconda (or full Anaconda)
7. TensorRT

## Install
Create a conda environment with the above prerequisits, then follow the steps below to install the OOT module

``` shell
conda activate <env>
mkdir build
cd build
cmake -G Ninja -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DCMAKE_PREFIX_PATH=$CONDA_PREFIX -DLIB_SUFFIX="" ..
cmake --build .
cmake --build . --target install
```

## Run
The flow graph ```examples/deepwive_tx_ofdm_custom.grc``` contains a loopback version of both Tx and Rx.
In order to run the flow graph, the TensorRT engines have to be downloaded before hand.
Please email me @ ```tt2114@ic.ac.uk``` to obtain the link to the engines.
