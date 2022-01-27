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

#include <cstring>
#include <gnuradio/thread/thread.h>
#include <pmt/pmt.h>
#include <stdexcept>
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <gnuradio/io_signature.h>
#include "ofdm_frame_equalizer_impl.h"

namespace gr {
  namespace deepwive_v1 {

    ofdm_frame_equalizer::sptr
    ofdm_frame_equalizer::make(double freq, double bw, int packet_len, bool log, bool debug)
    {
      return gnuradio::get_initial_sptr
        (new ofdm_frame_equalizer_impl(freq, bw, packet_len, log, debug));
    }


    ofdm_frame_equalizer_impl::ofdm_frame_equalizer_impl(double freq, double bw, int packet_len, bool log, bool debug)
      : gr::block("ofdm_frame_equalizer",
                  gr::io_signature::make(1, 1, 64*sizeof(gr_complex)),
                  gr::io_signature::make(1, 1, 48*sizeof(gr_complex))),
        d_current_symbol(0),
        d_log(log),
        d_debug(debug),
        d_freq(freq),
        d_bw(bw),
        d_freq_offset_from_synclong(0.0),
        d_packet_len(packet_len)
    {
      // message_port_register_out(pmt::mp("payload_IQ"));
      set_tag_propagation_policy(block::TPP_DONT);
    }

    ofdm_frame_equalizer_impl::~ofdm_frame_equalizer_impl() {}

    void ofdm_frame_equalizer_impl::set_bandwidth(double bw)
    {
      gr::thread::scoped_lock lock(d_mutex);
      d_bw = bw;
    }

    void ofdm_frame_equalizer_impl::set_frequency(double freq)
    {
      gr::thread::scoped_lock lock(d_mutex);
      d_freq = freq;
    }

    void ofdm_frame_equalizer_impl::forecast (int noutput_items, gr_vector_int &ninput_items_required)
    {
      ninput_items_required[0] = noutput_items;
    }

    int ofdm_frame_equalizer_impl::general_work (int noutput_items,
                                                 gr_vector_int &ninput_items,
                                                 gr_vector_const_void_star &input_items,
                                                 gr_vector_void_star &output_items)
    {
      gr::thread::scoped_lock lock(d_mutex);

      const gr_complex* in = (const gr_complex*) input_items[0];
      gr_complex* out = (gr_complex*) output_items[0];

      int i = 0;
      int o = 0;
      gr_complex symbols[48];
      gr_complex current_symbol[64];

      dout << "FRAME EQUALIZER: input " << ninput_items[0] << "  output " << noutput_items
         << std::endl;

      while ((i < ninput_items[0]) && (o < noutput_items)){
        get_tags_in_window(tags, 0, i, i+1, pmt::string_to_symbol("data_start"));

        //new frame
        if (tags.size()) {
          dout << "n frames produced " << d_current_symbol << std::endl;
          dout << "n payload symbols produced " << d_payload_symbols << std::endl;
          dout << "packet len " << d_packet_len << std::endl;
          // dout << "found start" << std::endl;
          d_current_symbol = 0;
          d_payload_symbols = 0;

          d_freq_offset_from_synclong = pmt::to_double(tags.front().value) * d_bw / (2 * M_PI);
          d_epsilon0 = pmt::to_double(tags.front().value) * d_bw / (2 * M_PI * d_freq);
          d_er = 0;

          // dout << "epsilon: " << d_epsilon0 << std::endl;
        }
        else if (d_payload_symbols >= d_packet_len) {
          // dout << "not interesting; skip" << std::endl;
          i++;
          continue;
        }

        std::memcpy(current_symbol, in + i * 64, 64 * sizeof(gr_complex));

        // compensate sampling offset
        for (int i = 0; i < 64; i++) {
            current_symbol[i] *= exp(gr_complex(0,
                                                2 * M_PI * d_current_symbol * 80 *
                                                    (d_epsilon0 + d_er) * (i - 32) / 64));
        }

        gr_complex p = POLARITY[(d_current_symbol - 2) % 127];

        double beta;
        if (d_current_symbol < 2) {
            beta = arg(current_symbol[11] - current_symbol[25] + current_symbol[39] +
                       current_symbol[53]);

        } else {
            beta = arg((current_symbol[11] * p) + (current_symbol[39] * p) +
                       (current_symbol[25] * p) + (current_symbol[53] * -p));
        }

        double er = arg((conj(d_prev_pilots[0]) * current_symbol[11] * p) +
                        (conj(d_prev_pilots[1]) * current_symbol[25] * p) +
                        (conj(d_prev_pilots[2]) * current_symbol[39] * p) +
                        (conj(d_prev_pilots[3]) * current_symbol[53] * -p));

        er *= d_bw / (2 * M_PI * d_freq * 80);

        if (d_current_symbol < 2) {
            d_prev_pilots[0] = current_symbol[11];
            d_prev_pilots[1] = -current_symbol[25];
            d_prev_pilots[2] = current_symbol[39];
            d_prev_pilots[3] = current_symbol[53];
        } else {
            d_prev_pilots[0] = current_symbol[11] * p;
            d_prev_pilots[1] = current_symbol[25] * p;
            d_prev_pilots[2] = current_symbol[39] * p;
            d_prev_pilots[3] = current_symbol[53] * -p;
        }

        // compensate residual frequency offset
        for (int i = 0; i < 64; i++) {
            current_symbol[i] *= exp(gr_complex(0, -beta));
        }

        // update estimate of residual frequency offset
        if (d_current_symbol >= 2) {
            double alpha = 0.1;
            d_er = (1 - alpha) * d_er + alpha * er;
        }

        uint8_t bits[48];
        equalize(current_symbol, d_current_symbol, symbols, bits);

        // process header
        if (d_current_symbol == 2) {
          extract_from_header(bits);

          add_item_tag(0,
                       nitems_written(0) + o,
                       pmt::intern("packet_len"),
                       pmt::from_long(d_packet_len));
          add_item_tag(0,
                       nitems_written(0) + o,
                       pmt::intern("first_flag"),
                       pmt::from_uint64(d_first_flag));
          add_item_tag(0,
                       nitems_written(0) + o,
                       pmt::intern("alloc_idx"),
                       pmt::from_uint64(d_alloc_idx));
          add_item_tag(0,
                       nitems_written(0) + o,
                       pmt::intern("snr"),
                       pmt::from_double(get_snr()));

          std::vector<gr_complex> csi = get_csi();
          add_item_tag(0,
                       nitems_written(0) + o,
                       pmt::intern("csi"),
                       pmt::init_c32vector(csi.size(), csi));
        }

        if (d_current_symbol > 2){
          std::memcpy(out + o * 48, symbols, 48 * sizeof(gr_complex));
          o++;
          d_payload_symbols += 48;
        }

        i++;
        d_current_symbol++;

      }

      // dout << "produced " << o << " consumed " << i << std::endl;

      consume(0, i);
      return o;
    }

    void ofdm_frame_equalizer_impl::extract_from_header(uint8_t* bits)
    {
      dout << "header bits" << std::endl;
      for (int i = 0; i < 48; i++){
        dout << (int)bits[i];
      }
      // TODO switch to convolutional coding
      dout << std::endl;
      std::vector<unsigned> header_first_flag(4);
      std::vector<unsigned> header_alloc_idx(4);

      int k = 0;
      for (int i = 0; i < 4; i++) {
        unsigned first_flag = 0;
        unsigned alloc_idx = 0;
        first_flag |= (((unsigned)bits[k]) & 1);
        header_first_flag[i] = first_flag;
        k++;
        while (k % 12 != 0) {
          alloc_idx |= (((unsigned)bits[k]) & 1) << ((k % 12) - 1);
          k++;
        }
        header_alloc_idx[i] = alloc_idx;
      }

      if (k > 48){
        std::runtime_error("header extraction failed");
      }

      // d_first_flag = header_first_flag[0];
      // d_alloc_idx = header_alloc_idx[0];

      d_first_flag = ((header_first_flag[0] & header_first_flag[1])
                      | (header_first_flag[0] & header_first_flag[2])
                      | (header_first_flag[0] & header_first_flag[3])
                      | (header_first_flag[1] & header_first_flag[2])
                      | (header_first_flag[1] & header_first_flag[3])
                      | (header_first_flag[2] & header_first_flag[3])
      );

      d_alloc_idx = ((header_alloc_idx[0] & header_alloc_idx[1])
                     | (header_alloc_idx[0] & header_alloc_idx[2])
                     | (header_alloc_idx[0] & header_alloc_idx[3])
                     | (header_alloc_idx[1] & header_alloc_idx[2])
                     | (header_alloc_idx[1] & header_alloc_idx[3])
                     | (header_alloc_idx[2] & header_alloc_idx[3])
      );

      dout << "first flag " << d_first_flag << " alloc idx " << d_alloc_idx << std::endl;
    }

    void ofdm_frame_equalizer_impl::equalize(gr_complex* in,
                                             int n,
                                             gr_complex* symbols,
                                             uint8_t* bits)
    {
      // NOTE this code assumes BPSK header

      if (n == 0) {
        std::memcpy(d_H, in, 64 * sizeof(gr_complex));
      }
      else if (n == 1) {
        double signal = 0;
        double noise = 0;
        for (int i = 0; i < 64; i++){
          if ((i == 32) || (i < 6) || (i > 58)) {
            continue;
          }
          noise += std::pow(std::abs(d_H[i] - in[i]), 2);
          signal += std::pow(std::abs(d_H[i] + in[i]), 2);
          d_H[i] += in[i];
          d_H[i] /= LONG[i] * gr_complex(2, 0);
        }

        d_snr = 10 * std::log10(signal / noise / 2);
      }
      else if (n == 2) {
        int c = 0;
        for (int i = 0; i < 64; i++) {
          if ((i == 11) || (i == 25) || (i == 32) || (i == 39) || (i == 53) ||
              (i < 6) || (i > 58)) {
            continue;
          } else {
            symbols[c] = in[i] / d_H[i];
            bits[c] = (real(symbols[c]) > 0);
            gr_complex point = gr_complex(2 * ((int)bits[c]) - 1, 0);
            d_H[i] = gr_complex(1 - alpha, 0) * d_H[i] +
                     gr_complex(alpha, 0) * (in[i] / point);
            c++;
          }
        }
      }
      else {
        int c = 0;
        for (int i = 0; i < 64; i++) {
          if ((i == 11) || (i == 25) || (i == 32) || (i == 39) || (i == 53) ||
              (i < 6) || (i > 58)) {
            continue;
          } else {
            symbols[c] = in[i] / d_H[i];
            c++;
          }
        }
      }
     
    }

    double ofdm_frame_equalizer_impl::get_snr() { return d_snr; }

    std::vector<gr_complex> ofdm_frame_equalizer_impl::get_csi()
    {
      std::vector<gr_complex> csi;
      csi.reserve(52);
      for (int i = 0; i < 64; i++) {
        if ((i == 32) || (i < 6) || (i > 58)) {
          continue;
        }
        csi.push_back(d_H[i]);
      }
      return csi;
    }

  } /* namespace deepwive_v1 */
} /* namespace gr */

