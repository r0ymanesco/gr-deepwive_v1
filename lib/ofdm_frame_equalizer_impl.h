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

#define BUTTERFLY(i, sym)                                         \
    {                                                             \
    int m0, m1, m2, m3;                                           \
    /* ACS for 0 branch */                                        \
    m0 = state[i].metric + mets[sym];          /* 2*i */          \
    m1 = state[i + 32].metric + mets[3 ^ sym]; /* 2*i + 64 */     \
    if (m0 > m1) {                                                \
        next[2 * i].metric = m0;                                  \
        next[2 * i].path = state[i].path << 1;                    \
    } else {                                                      \
        next[2 * i].metric = m1;                                  \
        next[2 * i].path = (state[i + 32].path << 1) | 1;         \
    }                                                             \
    /* ACS for 1 branch */                                        \
    m2 = state[i].metric + mets[3 ^ sym];  /* 2*i + 1 */          \
    m3 = state[i + 32].metric + mets[sym]; /* 2*i + 65 */         \
    if (m2 > m3) {                                                \
        next[2 * i + 1].metric = m2;                              \
        next[2 * i + 1].path = state[i].path << 1;                \
    } else {                                                      \
        next[2 * i + 1].metric = m3;                              \
        next[2 * i + 1].path = (state[i + 32].path << 1) | 1;     \
    }                                                             \
}

// max number of traceback bytes
#define TRACEBACK_MAX 24
#define MAX_PAYLOAD_SIZE 1500
#define MAX_PSDU_SIZE (MAX_PAYLOAD_SIZE + 28) // MAC, CRC
#define MAX_ENCODED_BITS ((16 + 8 * MAX_PSDU_SIZE + 6) * 2 + 288)

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
        int d_payload_symbols = 0;
        std::vector<gr_complex> payload_symbols;

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

        // viterbi params
        union branchtab27 {
            unsigned char c[32];
        } d_branchtab27_generic[2];

        unsigned char d_metric0_generic[64] __attribute__((aligned(16)));
        unsigned char d_metric1_generic[64] __attribute__((aligned(16)));
        unsigned char d_path0_generic[64] __attribute__((aligned(16)));
        unsigned char d_path1_generic[64] __attribute__((aligned(16)));
        // Position in circular buffer where the current decoded byte is stored
        int d_store_pos = 0;
        // Metrics for each state
        unsigned char d_mmresult[64] __attribute__((aligned(16)));
        // Paths for each state
        unsigned char d_ppresult[TRACEBACK_MAX][64] __attribute__((aligned(16)));

        // parity lookup table
        const unsigned char PARTAB[256] = {
        0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1,
        0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0,
        0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0,
        1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1,
        0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0,
        1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1,
        1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0,
        1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0,
        0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0,
    };

        int d_ntraceback = 1;
        uint8_t d_decoded[MAX_ENCODED_BITS * 3/4];

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

        // viterbi functions
        void extract_from_header_cc(uint8_t* bits);
        void cc_reset();
        virtual uint8_t* cc_decode(uint8_t* in);

        void viterbi_butterfly2_generic(unsigned char* symbols,
                                        unsigned char m0[],
                                        unsigned char m1[],
                                        unsigned char p0[],
                                        unsigned char p1[]);

        // find current best path
        unsigned char viterbi_get_output_generic(unsigned char* mm0,
                                                 unsigned char* pp0,
                                                 int ntraceback,
                                                 unsigned char* outbuf);

    };

  } // namespace deepwive_v1
} // namespace gr

#endif /* INCLUDED_DEEPWIVE_V1_OFDM_FRAME_EQUALIZER_IMPL_H */

