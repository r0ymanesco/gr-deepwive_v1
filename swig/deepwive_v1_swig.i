/* -*- c++ -*- */

#define DEEPWIVE_V1_API
#define DIGITAL_API

%include "gnuradio.i"           // the common stuff

//load generated python docstrings
%include "deepwive_v1_swig_doc.i"

%{
#include "deepwive_v1/ofdm_sync_short.h"
#include "deepwive_v1/ofdm_sync_long.h"
#include "deepwive_v1/ofdm_frame_equalizer.h"
%}

%include "deepwive_v1/ofdm_sync_short.h"
%include "deepwive_v1/ofdm_sync_long.h"
%include "deepwive_v1/ofdm_frame_equalizer.h"

GR_SWIG_BLOCK_MAGIC2(deepwive_v1, ofdm_sync_short);
GR_SWIG_BLOCK_MAGIC2(deepwive_v1, ofdm_sync_long);
GR_SWIG_BLOCK_MAGIC2(deepwive_v1, ofdm_frame_equalizer);
