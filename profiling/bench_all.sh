#!/bin/bash

# images
mkdir -p $PROF_DIR/$OUTPUT_DIR
# logs
mkdir -p $PROF_DIR/$LOG_DIR


export SD_PT_MEM="sdp"
source profiling/blas_tuning.sh -s 8

export SD_PT_MEM="xformers"
source profiling/blas_tuning.sh -s 8

