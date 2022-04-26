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


    ofdm_frame_equalizer_impl::ofdm_frame_equalizer_impl(
      double freq, double bw, int packet_len, bool log, bool debug)
      : gr::block("ofdm_frame_equalizer",
                  gr::io_signature::make(1, 1, 64*sizeof(gr_complex)),
                  gr::io_signature::make(0, 0, 0)),
        d_current_symbol(0),
        d_log(log),
        d_debug(debug),
        d_freq(freq),
        d_bw(bw),
        d_freq_offset_from_synclong(0.0),
        d_packet_len(packet_len)
    {
      message_port_register_out(pmt::mp("payload_IQ"));
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
      // gr_complex* out = (gr_complex*) output_items[0];

      int i = 0;
      int o = 0;
      gr_complex symbols[48];
      gr_complex current_symbol[64];

      // dout << "FRAME EQUALIZER: input " << ninput_items[0] << "  output " << noutput_items
      //    << std::endl;

      while ((i < ninput_items[0]) && (o < noutput_items)){
        get_tags_in_window(tags, 0, i, i+1, pmt::string_to_symbol("data_start"));

        //new frame
        if (tags.size()) {

          dout << "new frame" << std::endl;
          dout << "n frames produced " << d_current_symbol << std::endl;
          dout << "n payload symbols produced " << d_payload_symbols << std::endl;

          d_current_symbol = 0;
          d_payload_symbols = 0;
          payload_symbols.clear();

          d_freq_offset_from_synclong = pmt::to_double(tags.front().value) * d_bw / (2 * M_PI);
          d_epsilon0 = pmt::to_double(tags.front().value) * d_bw / (2 * M_PI * d_freq);
          d_er = 0;

          // dout << "epsilon: " << d_epsilon0 << std::endl;
        }
        else if (d_payload_symbols > d_packet_len) {
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
          // extract_from_header_cc(bits);

          dout << "snr " << get_snr() - 20 << " dB" << std::endl;

        }

        if (d_current_symbol > 2){
          // std::memcpy(out + o * 48, symbols, 48 * sizeof(gr_complex));
          for (int i = 0; i < 48; i++) {
            payload_symbols.push_back(symbols[i]);
          }
          // o++;
          d_payload_symbols += 48;
        }

        if (d_payload_symbols == d_packet_len){
          pmt::pmt_t dict = pmt::make_dict();
          dict = pmt::dict_add(
            dict, pmt::mp("packet_len"), pmt::from_long(d_packet_len));
          dict = pmt::dict_add(
            dict, pmt::mp("first_flag"), pmt::from_long(d_first_flag));
          dict = pmt::dict_add(
            dict, pmt::mp("alloc_idx"), pmt::from_long(d_alloc_idx));
          dict = pmt::dict_add(
            dict, pmt::mp("snr"), pmt::from_double(get_snr() - 20));

          message_port_pub(
            pmt::mp("payload_IQ"),
            pmt::cons(dict, pmt::init_c32vector(payload_symbols.size(), payload_symbols)));
        }

        i++;
        d_current_symbol++;

      }

      // dout << "consumed " << i << std::endl;

      consume(0, i);
      return o;
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
          }
          else {
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
          }
          else {
            symbols[c] = in[i] / d_H[i];
            c++;
          }
        }
      }
    }

    void ofdm_frame_equalizer_impl::deinterleave(uint8_t* rx_bits)
    {
      for (int i = 0; i < 48; i++) {
        d_deinterleaved[i] = rx_bits[interleaver_pattern[i]];
      }
    }

    void ofdm_frame_equalizer_impl::extract_from_header_cc(uint8_t* bits)
    {
      dout << "received header bits" << std::endl;
      for (int i = 0; i < 48; i++){
        dout << (int)bits[i];
      }
      dout << std::endl;

      deinterleave(bits);

      uint8_t* decoded_bits = cc_decode(d_deinterleaved);
      dout << "cc decoded header" << std::endl;

      d_first_flag = 0;
      d_alloc_idx = 0;
      bool parity = false;

      for (int i = 0; i < 17; i++) {
        dout << (unsigned)decoded_bits[i];

        parity ^= decoded_bits[i];

        if (i == 0) {
          d_first_flag |= (((unsigned)decoded_bits[i]) & 1);
        }
        else if (i < 12) {
          d_alloc_idx |= (((unsigned)decoded_bits[i]) & 1) << (i - 1);
        }

      }
      dout << std::endl;

      if (parity != decoded_bits[17]) {
        throw std::runtime_error("header parity check fail");
      }
      else {
        dout << "header parity check PASS" << std::endl;
        dout << "first flag " << d_first_flag << " alloc idx " << d_alloc_idx << std::endl;
      }

    }

    uint8_t* ofdm_frame_equalizer_impl::cc_decode(uint8_t *in)
    {
      cc_reset();

      int in_count = 0;
      int out_count = 0;
      int n_decoded = 0;

      while (n_decoded < 24) {

        if ((in_count % 4) == 0) { // 0 or 3
          viterbi_butterfly2_generic(&in[in_count & 0xfffffffc],
                                     d_metric0_generic,
                                     d_metric1_generic,
                                     d_path0_generic,
                                     d_path1_generic);

          if ((in_count > 0) && (in_count % 16) == 8) { // 8 or 11
            unsigned char c;

            viterbi_get_output_generic(
              d_metric0_generic, d_path0_generic, d_ntraceback, &c);

            if (out_count >= d_ntraceback) {
              for (int i = 0; i < 8; i++) {
                d_decoded[(out_count - d_ntraceback) * 8 + i] =
                  (c >> (7 - i)) & 0x1;
                n_decoded++;
              }
            }
            out_count++;
          }
        }
        in_count++;
      }

      return d_decoded;
    }

    void ofdm_frame_equalizer_impl::viterbi_butterfly2_generic(unsigned char *symbols,
                                                               unsigned char *mm0,
                                                               unsigned char *mm1,
                                                               unsigned char *pp0,
                                                               unsigned char *pp1)
    {
      int i, j, k;

      unsigned char *metric0, *metric1;
      unsigned char *path0, *path1;

      metric0 = mm0;
      path0 = pp0;
      metric1 = mm1;
      path1 = pp1;

      // Operate on 4 symbols (2 bits) at a time

      unsigned char m0[16], m1[16], m2[16], m3[16], decision0[16], decision1[16],
        survivor0[16], survivor1[16];
      unsigned char metsv[16], metsvm[16];
      unsigned char shift0[16], shift1[16];
      unsigned char tmp0[16], tmp1[16];
      unsigned char sym0v[16], sym1v[16];
      unsigned short simd_epi16;

      for (j = 0; j < 16; j++) {
        sym0v[j] = symbols[0];
        sym1v[j] = symbols[1];
      }

      for (i = 0; i < 2; i++) {
        if (symbols[0] == 2) {
          for (j = 0; j < 16; j++) {
            metsvm[j] = d_branchtab27_generic[1].c[(i * 16) + j] ^ sym1v[j];
            metsv[j] = 1 - metsvm[j];
          }
        } else if (symbols[1] == 2) {
          for (j = 0; j < 16; j++) {
            metsvm[j] = d_branchtab27_generic[0].c[(i * 16) + j] ^ sym0v[j];
            metsv[j] = 1 - metsvm[j];
          }
        } else {
          for (j = 0; j < 16; j++) {
            metsvm[j] = (d_branchtab27_generic[0].c[(i * 16) + j] ^ sym0v[j]) +
              (d_branchtab27_generic[1].c[(i * 16) + j] ^ sym1v[j]);
            metsv[j] = 2 - metsvm[j];
          }
        }

        for (j = 0; j < 16; j++) {
          m0[j] = metric0[(i * 16) + j] + metsv[j];
          m1[j] = metric0[((i + 2) * 16) + j] + metsvm[j];
          m2[j] = metric0[(i * 16) + j] + metsvm[j];
          m3[j] = metric0[((i + 2) * 16) + j] + metsv[j];
        }

        for (j = 0; j < 16; j++) {
          decision0[j] = ((m0[j] - m1[j]) > 0) ? 0xff : 0x0;
          decision1[j] = ((m2[j] - m3[j]) > 0) ? 0xff : 0x0;
          survivor0[j] = (decision0[j] & m0[j]) | ((~decision0[j]) & m1[j]);
          survivor1[j] = (decision1[j] & m2[j]) | ((~decision1[j]) & m3[j]);
        }

        for (j = 0; j < 16; j += 2) {
          simd_epi16 = path0[(i * 16) + j];
          simd_epi16 |= path0[(i * 16) + (j + 1)] << 8;
          simd_epi16 <<= 1;
          shift0[j] = simd_epi16;
          shift0[j + 1] = simd_epi16 >> 8;

          simd_epi16 = path0[((i + 2) * 16) + j];
          simd_epi16 |= path0[((i + 2) * 16) + (j + 1)] << 8;
          simd_epi16 <<= 1;
          shift1[j] = simd_epi16;
          shift1[j + 1] = simd_epi16 >> 8;
        }
        for (j = 0; j < 16; j++) {
          shift1[j] = shift1[j] + 1;
        }

        for (j = 0, k = 0; j < 16; j += 2, k++) {
          metric1[(2 * i * 16) + j] = survivor0[k];
          metric1[(2 * i * 16) + (j + 1)] = survivor1[k];
        }
        for (j = 0; j < 16; j++) {
          tmp0[j] = (decision0[j] & shift0[j]) | ((~decision0[j]) & shift1[j]);
        }

        for (j = 0, k = 8; j < 16; j += 2, k++) {
          metric1[((2 * i + 1) * 16) + j] = survivor0[k];
          metric1[((2 * i + 1) * 16) + (j + 1)] = survivor1[k];
        }
        for (j = 0; j < 16; j++) {
          tmp1[j] = (decision1[j] & shift0[j]) | ((~decision1[j]) & shift1[j]);
        }

        for (j = 0, k = 0; j < 16; j += 2, k++) {
          path1[(2 * i * 16) + j] = tmp0[k];
          path1[(2 * i * 16) + (j + 1)] = tmp1[k];
        }
        for (j = 0, k = 8; j < 16; j += 2, k++) {
          path1[((2 * i + 1) * 16) + j] = tmp0[k];
          path1[((2 * i + 1) * 16) + (j + 1)] = tmp1[k];
        }
      }

      metric0 = mm1;
      path0 = pp1;
      metric1 = mm0;
      path1 = pp0;

      for (j = 0; j < 16; j++) {
        sym0v[j] = symbols[2];
        sym1v[j] = symbols[3];
      }

      for (i = 0; i < 2; i++) {
        if (symbols[2] == 2) {
          for (j = 0; j < 16; j++) {
            metsvm[j] = d_branchtab27_generic[1].c[(i * 16) + j] ^ sym1v[j];
            metsv[j] = 1 - metsvm[j];
          }
        } else if (symbols[3] == 2) {
          for (j = 0; j < 16; j++) {
            metsvm[j] = d_branchtab27_generic[0].c[(i * 16) + j] ^ sym0v[j];
            metsv[j] = 1 - metsvm[j];
          }
        } else {
          for (j = 0; j < 16; j++) {
            metsvm[j] = (d_branchtab27_generic[0].c[(i * 16) + j] ^ sym0v[j]) +
              (d_branchtab27_generic[1].c[(i * 16) + j] ^ sym1v[j]);
            metsv[j] = 2 - metsvm[j];
          }
        }

        for (j = 0; j < 16; j++) {
          m0[j] = metric0[(i * 16) + j] + metsv[j];
          m1[j] = metric0[((i + 2) * 16) + j] + metsvm[j];
          m2[j] = metric0[(i * 16) + j] + metsvm[j];
          m3[j] = metric0[((i + 2) * 16) + j] + metsv[j];
        }

        for (j = 0; j < 16; j++) {
          decision0[j] = ((m0[j] - m1[j]) > 0) ? 0xff : 0x0;
          decision1[j] = ((m2[j] - m3[j]) > 0) ? 0xff : 0x0;
          survivor0[j] = (decision0[j] & m0[j]) | ((~decision0[j]) & m1[j]);
          survivor1[j] = (decision1[j] & m2[j]) | ((~decision1[j]) & m3[j]);
        }

        for (j = 0; j < 16; j += 2) {
          simd_epi16 = path0[(i * 16) + j];
          simd_epi16 |= path0[(i * 16) + (j + 1)] << 8;
          simd_epi16 <<= 1;
          shift0[j] = simd_epi16;
          shift0[j + 1] = simd_epi16 >> 8;

          simd_epi16 = path0[((i + 2) * 16) + j];
          simd_epi16 |= path0[((i + 2) * 16) + (j + 1)] << 8;
          simd_epi16 <<= 1;
          shift1[j] = simd_epi16;
          shift1[j + 1] = simd_epi16 >> 8;
        }
        for (j = 0; j < 16; j++) {
          shift1[j] = shift1[j] + 1;
        }

        for (j = 0, k = 0; j < 16; j += 2, k++) {
          metric1[(2 * i * 16) + j] = survivor0[k];
          metric1[(2 * i * 16) + (j + 1)] = survivor1[k];
        }
        for (j = 0; j < 16; j++) {
          tmp0[j] = (decision0[j] & shift0[j]) | ((~decision0[j]) & shift1[j]);
        }

        for (j = 0, k = 8; j < 16; j += 2, k++) {
          metric1[((2 * i + 1) * 16) + j] = survivor0[k];
          metric1[((2 * i + 1) * 16) + (j + 1)] = survivor1[k];
        }
        for (j = 0; j < 16; j++) {
          tmp1[j] = (decision1[j] & shift0[j]) | ((~decision1[j]) & shift1[j]);
        }

        for (j = 0, k = 0; j < 16; j += 2, k++) {
          path1[(2 * i * 16) + j] = tmp0[k];
          path1[(2 * i * 16) + (j + 1)] = tmp1[k];
        }
        for (j = 0, k = 8; j < 16; j += 2, k++) {
          path1[((2 * i + 1) * 16) + j] = tmp0[k];
          path1[((2 * i + 1) * 16) + (j + 1)] = tmp1[k];
        }
      }

    }

    unsigned char ofdm_frame_equalizer_impl::viterbi_get_output_generic(unsigned char* mm0,
                                                                        unsigned char* pp0,
                                                                        int ntraceback,
                                                                        unsigned char* outbuf)
    {
      int i;
      int bestmetric, minmetric;
      int beststate = 0;
      int pos = 0;
      int j;

      // circular buffer with the last ntraceback paths
      d_store_pos = (d_store_pos + 1) % ntraceback;

      for (i = 0; i < 4; i++) {
        for (j = 0; j < 16; j++) {
          d_mmresult[(i * 16) + j] = mm0[(i * 16) + j];
          d_ppresult[d_store_pos][(i * 16) + j] = pp0[(i * 16) + j];
        }
      }

      // Find out the best final state
      bestmetric = d_mmresult[beststate];
      minmetric = d_mmresult[beststate];

      for (i = 1; i < 64; i++) {
        if (d_mmresult[i] > bestmetric) {
          bestmetric = d_mmresult[i];
          beststate = i;
        }
        if (d_mmresult[i] < minmetric) {
          minmetric = d_mmresult[i];
        }
      }

      // Trace back
      for (i = 0, pos = d_store_pos; i < (ntraceback - 1); i++) {
        // Obtain the state from the output bits
        // by clocking in the output bits in reverse order.
        // The state has only 6 bits
        beststate = d_ppresult[pos][beststate] >> 2;
        pos = (pos - 1 + ntraceback) % ntraceback;
      }

      // Store output byte
      *outbuf = d_ppresult[pos][beststate];

      for (i = 0; i < 4; i++) {
        for (j = 0; j < 16; j++) {
          pp0[(i * 16) + j] = 0;
          mm0[(i * 16) + j] = mm0[(i * 16) + j] - minmetric;
        }
      }

      return bestmetric;
    }

    void ofdm_frame_equalizer_impl::cc_reset()
    {
      int i, j;

      for (i = 0; i < 4; i++) {
        d_metric0_generic[i] = 0;
        d_path0_generic[i] = 0;
      }

      int polys[2] = { 0x6d, 0x4f };
      for (i = 0; i < 32; i++) {
        d_branchtab27_generic[0].c[i] =
          (polys[0] < 0) ^ PARTAB[(2 * i) & abs(polys[0])] ? 1 : 0;
        d_branchtab27_generic[1].c[i] =
          (polys[1] < 0) ^ PARTAB[(2 * i) & abs(polys[1])] ? 1 : 0;
      }

      for (i = 0; i < 64; i++) {
        d_mmresult[i] = 0;
        for (j = 0; j < TRACEBACK_MAX; j++) {
          d_ppresult[j][i] = 0;
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

    void ofdm_frame_equalizer_impl::extract_from_header(uint8_t* bits)
    {
      dout << "header bits" << std::endl;
      for (int i = 0; i < 48; i++){
        dout << (int)bits[i];
      }
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
                      | (header_first_flag[1] & header_first_flag[2])
                      | (header_first_flag[0] & header_first_flag[2])
      );

      d_alloc_idx = ((header_alloc_idx[0] & header_alloc_idx[1])
                     | (header_alloc_idx[1] & header_alloc_idx[2])
                     | (header_alloc_idx[0] & header_alloc_idx[2])
      );

      // d_first_flag = ((header_first_flag[0] & header_first_flag[1])
      //                 | (header_first_flag[0] & header_first_flag[2])
      //                 | (header_first_flag[0] & header_first_flag[3])
      //                 | (header_first_flag[1] & header_first_flag[2])
      //                 | (header_first_flag[1] & header_first_flag[3])
      //                 | (header_first_flag[2] & header_first_flag[3])
      // );

      // d_alloc_idx = ((header_alloc_idx[0] & header_alloc_idx[1])
      //                | (header_alloc_idx[0] & header_alloc_idx[2])
      //                | (header_alloc_idx[0] & header_alloc_idx[3])
      //                | (header_alloc_idx[1] & header_alloc_idx[2])
      //                | (header_alloc_idx[1] & header_alloc_idx[3])
      //                | (header_alloc_idx[2] & header_alloc_idx[3])
      // );

      dout << "first flag " << d_first_flag << " alloc idx " << d_alloc_idx << std::endl;
    }

  } /* namespace deepwive_v1 */
} /* namespace gr */

