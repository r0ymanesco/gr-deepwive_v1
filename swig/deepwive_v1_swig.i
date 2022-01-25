/* -*- c++ -*- */

#define DEEPWIVE_V1_API
#define DIGITAL_API

%include "gnuradio.i"           // the common stuff

//load generated python docstrings
%include "deepwive_v1_swig_doc.i"

%{
#include "deepwive_v1/soft_equalizer_single_tap.h"
#include "deepwive_v1/deepwive_packet_header.h"
#include "deepwive_v1/header_format.h"
#include "deepwive_v1/ofdm_sync_short.h"
#include "deepwive_v1/ofdm_sync_long.h"
%}

%include "gnuradio/digital/ofdm_equalizer_base.h"
%template(ofdm_equalizer_base_sptr) boost::shared_ptr<gr::digital::ofdm_equalizer_base>;
%template(ofdm_equalizer_1d_pilots_sptr) boost::shared_ptr<gr::digital::ofdm_equalizer_1d_pilots>;
%pythoncode %{
ofdm_equalizer_1d_pilots_sptr.__repr__ = lambda self: "<OFDM equalizer 1D base class>"
%}

using namespace gr::digital;
%include "gnuradio/digital/constellation.h"
%include "deepwive_v1/soft_equalizer_single_tap.h"
%template(soft_equalizer_single_tap_sptr) boost::shared_ptr<gr::deepwive_v1::soft_equalizer_single_tap>;
%pythoncode %{
soft_equalizer_single_tap_sptr.__repr__ = lambda self: "<soft_equalizer_single_tap>"
soft_equalizer_single_tap = soft_equalizer_single_tap .make;
%}
//%rename(soft_equalizer_single_tap) make_soft_equalizer_single_tap;
//%ignore soft_equalizer_single_tap;

%include "gnuradio/digital/packet_header_default.h"
%template(packet_header_default_sptr) boost::shared_ptr<gr::digital::packet_header_default>;

%include "deepwive_v1/deepwive_packet_header.h"
%template(deepwive_packet_header_sptr) boost::shared_ptr<gr::deepwive_v1::deepwive_packet_header>;
%pythoncode %{
deepwive_packet_header_sptr.__repr__ = lambda self: "<deepwive_packet_header>"
deepwive_packet_header = deepwive_packet_header .make;
%}


%include "gnuradio/digital/header_format_base.h"
%template(header_format_base_sptr) boost::shared_ptr<gr::digital::header_format_base>;
%include "gnuradio/digital/header_format_crc.h"
%template(header_format_crc_sptr) boost::shared_ptr<gr::digital::header_format_crc>;
%include "deepwive_v1/header_format.h"
%template(header_format_sptr) boost::shared_ptr<gr::deepwive_v1::header_format>;
%pythoncode %{
header_format_sptr.__repr__ = lambda self: "<header_format>"
header_format = header_format .make;
%}

%include "deepwive_v1/ofdm_sync_short.h"
%include "deepwive_v1/ofdm_sync_long.h"

GR_SWIG_BLOCK_MAGIC2(deepwive_v1, ofdm_sync_short);
GR_SWIG_BLOCK_MAGIC2(deepwive_v1, ofdm_sync_long);
