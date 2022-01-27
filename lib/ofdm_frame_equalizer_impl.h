/* -*- c++ -*- */
/*
 * Copyright 2022 gr-deepwive_v1 author.
 *
 * This is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
 * any later version.
 *
 * This software is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this software; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 51 Franklin Street,
 * Boston, MA 02110-1301, USA.
 */

#ifndef INCLUDED_DEEPWIVE_V1_OFDM_FRAME_EQUALIZER_IMPL_H
#define INCLUDED_DEEPWIVE_V1_OFDM_FRAME_EQUALIZER_IMPL_H

#include <deepwive_v1/ofdm_frame_equalizer.h>
#include <gnuradio/digital/constellation.h>
#include <gnuradio/gr_complex.h>

#define dout d_debug&& std::cout
#define mylog(msg)                      \
    do {                                \
        if (d_log) {                    \
            GR_LOG_INFO(d_logger, msg); \
        }                               \
    } while (0);

namespace gr {
  namespace deepwive_v1 {

    class ofdm_frame_equalizer_impl : public ofdm_frame_equalizer
    {
     private:
        gr::thread::mutex d_mutex;
        std::vector<gr::tag_t> tags;
        bool d_debug;
        bool d_log;
        int d_current_symbol;
        int d_packet_len;

        double d_freq;                      // Hz
        double d_freq_offset_from_synclong; // Hz, estimation from "sync_long" block
        double d_bw;                        // Hz
        double d_er;
        double d_epsilon0;
        gr_complex d_prev_pilots[4];
        gr_complex symbols[48];
        gr_complex d_H[64];
        double d_snr;
        unsigned d_first_flag;
        unsigned d_alloc_idx;
        const double alpha = 0.5;

        const gr_complex POLARITY[127] = {
        1,  1,  1,  1,  -1, -1, -1, 1,  -1, -1, -1, -1, 1,  1,  -1, 1,  -1, -1, 1, 1,  -1, 1,
        1,  -1, 1,  1,  1,  1,  1,  1,  -1, 1,  1,  1,  -1, 1,  1,  -1, -1, 1,  1, 1,  -1, 1,
        -1, -1, -1, 1,  -1, 1,  -1, -1, 1,  -1, -1, 1,  1,  1,  1,  1,  -1, -1, 1, 1,  -1, -1,
        1,  -1, 1,  -1, 1,  1,  -1, -1, -1, 1,  1,  -1, -1, -1, -1, 1,  -1, -1, 1, -1, 1,  1,
        1,  1,  -1, 1,  -1, 1,  -1, 1,  -1, -1, -1, -1, -1, 1,  -1, 1,  1,  -1, 1, -1, 1,  1,
        1,  -1, -1, 1,  -1, -1, -1, 1,  1,  1,  -1, -1, -1, -1, -1, -1, -1
      };

        const gr_complex LONG[64] = {
        0,  0,  0,  0,  0,  0,  1,  1,  -1, -1, 1,  1,  -1,
        1,  -1, 1,  1,  1,  1,  1,  1,  -1, -1, 1,  1,  -1,
        1,  -1, 1,  1,  1,  1,  0,  1,  -1, -1, 1,  1,  -1,
        1,  -1, 1,  -1, -1, -1, -1, -1, 1,  1,  -1, -1, 1,
        -1, 1,  -1, 1,  1,  1,  1,  0,  0,  0,  0,  0
      };

     public:
        ofdm_frame_equalizer_impl(double freq, double bw, int packet_len, bool log, bool debug);
        ~ofdm_frame_equalizer_impl();

        // Where all the action really happens
        void forecast (int noutput_items, gr_vector_int &ninput_items_required);

        void set_bandwidth(double bw);

        void set_frequency(double freq);

        int general_work(int noutput_items,
                         gr_vector_int &ninput_items,
                         gr_vector_const_void_star &input_items,
                         gr_vector_void_star &output_items);

        virtual void equalize(gr_complex* in,
                              int n,
                              gr_complex* symbols,
                              uint8_t* bits);

        void extract_from_header(uint8_t* bits);

        double get_snr();

        std::vector<gr_complex> get_csi();
    };

  } // namespace deepwive_v1
} // namespace gr

#endif /* INCLUDED_DEEPWIVE_V1_OFDM_FRAME_EQUALIZER_IMPL_H */

