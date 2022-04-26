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

#include <cstdint>
#include <gnuradio/sptr_magic.h>
#include <gnuradio/types.h>
#include <pmt/pmt.h>
#include <stdexcept>
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <gnuradio/io_signature.h>
#include <deepwive_v1/header_insert.h>


using namespace gr::deepwive_v1;
using namespace std;


class header_insert_impl : public header_insert
{
  public:
    header_insert_impl(int packet_len, std::string length_tag_key, bool debug)
      : block("header_insert",
              gr::io_signature::make(1, 1, sizeof(gr_complex)),
              gr::io_signature::make(1, 1, sizeof(gr_complex))),
        d_packet_len(packet_len),
        d_length_tag_key(length_tag_key),
        d_debug(debug)
    {
      set_tag_propagation_policy(block::TPP_DONT);
    }

    ~header_insert_impl() {}

    int general_work(int noutput,
                     gr_vector_int& ninput_items,
                     gr_vector_const_void_star& input_items,
                     gr_vector_void_star& output_items)
    {
      const gr_complex* in = (const gr_complex*) input_items[0];
      gr_complex* out = (gr_complex*) output_items[0];

      dout << "HEADER ninput " << ninput_items[0] << "   noutput " << noutput
           << " state " << d_state << std::endl;

      int ninput = std::min(ninput_items[0], 8192);
      const uint64_t nread = nitems_read(0);
      get_tags_in_range(d_tags, 0, nread, nread + ninput);
      dout << "nread " << nread << std::endl;

      if (d_tags.size()) {
        std::sort(d_tags.begin(), d_tags.end(), gr::tag_t::offset_compare);
        const uint64_t offset = d_tags.front().offset;
        dout << "offset " << offset << std::endl;

        if (offset > nread) {
          ninput = offset - nread;
          d_state = COPY;
        }
        else if (offset == nread) {
          if (d_offset && d_state == COPY){
            d_state = INSERT;
            d_offset = 0;
          }
        }
        else {
          throw std::runtime_error("wtf");
        }
      }
      else {
        dout << "no tags; continue state " << d_state << std::endl;
      }

      int i = 0;
      int o = 0;

      switch (d_state) {

        case INSERT: {

          if (!d_offset) {
            add_item_tag(0,
                         nitems_written(0),
                         pmt::intern(d_length_tag_key),
                         pmt::from_long(d_packet_len + 48),
                         pmt::string_to_symbol(name()));

            bool first_flag_found = false;
            bool alloc_idx_found = false;

            for (int i = 0; i < 2; i++) {
              if (pmt::eq(d_tags[i].key, pmt::mp("first_flag"))) {
                first_flag_found = true;
                d_first_flag = (unsigned)pmt::to_long(d_tags[i].value);
              }

              if (pmt::eq(d_tags[i].key, pmt::mp("alloc_idx"))) {
                alloc_idx_found = true;
                d_alloc_idx = (unsigned)pmt::to_long(d_tags[i].value);
              }
            }

            if ((!first_flag_found) || (!d_alloc_idx)){
              throw std::runtime_error("HEADER did not find tags");
            }

            d_header_bits = (char*)malloc(sizeof(char) * 48);
            // generate_header_field(d_header_bits, d_first_flag, d_alloc_idx);
            generate_header_field_old(d_header_bits, d_first_flag, d_alloc_idx);
          }

          while (o < noutput && d_offset < 48) {
            out[o] = gr_complex(2 * (float)d_header_bits[d_offset] - 1, 0);
            o++;
            d_offset++;
          }

          if (d_offset == 48){
            free(d_header_bits);
            d_offset = 0;
            d_state = COPY;
          }

          dout << "produced " << o << " consumed 0" << std::endl;
          consume(0, 0);
          return o;
        }

        case COPY: {
          while (i < ninput && o < noutput && d_offset < d_packet_len) {
            out[o] = in[i];
            o++;
            i++;
            d_offset++;
          }

          dout << "produced " << o << " consumed " << i << std::endl;
          consume(0, i);
          return o;
        }

      }

      throw std::runtime_error("HEADER undefined state");
      return 0;
    }

    void forecast(int noutput_items, gr_vector_int& ninput_items_required)
    {
      ninput_items_required[0] = 2 * d_packet_len;
    }

    int get_bit(int b, int i) { return (b & (1 << i) ? 1 : 0); }

    int ones(int n)
    {
      int sum = 0;
      for (int i = 0; i < 8; i++) {
        if (n & (1 << i)) {
          sum++;
        }
      }
      return sum;
    }

    void convolutional_encoding(const char* in, char* out)
    {
      int state = 0;

      for (int i = 0; i < 24; i++) {
        assert(in[i] == 0 || in[i] == 1);
        state = ((state << 1) & 0x7e) | in[i];
        out[i * 2] = ones(state & 0155) % 2;
        out[i * 2 + 1] = ones(state & 0117) % 2;
      }
    }

    void generate_header_field(char* out, unsigned first_flag, unsigned alloc_idx)
    {
      dout << "tx cc header bits first_flag " << first_flag << " alloc " << alloc_idx << std::endl;

      char* header_bits = (char*)malloc(sizeof(char) * 24);
      char* encoded_signal_header = (char*)malloc(sizeof(char) * 48);

      // first_flag bits
      dout << "header bits ";
      header_bits[0] = get_bit(first_flag, 0);
      dout << header_bits[0];

      // alloc_idx bits
      for (int i = 1; i < 12; i++) {
        header_bits[i] = get_bit(alloc_idx, i-1);
        dout << header_bits[i];
      }

      // 13-17th are zeros
      for (int i = 12; i < 17; i++) {
        header_bits[i] = 0;
        dout << header_bits[i];
      }

      // 18th bit is parity for the first 17 bits
      int sum = 0;
      for (int i = 0; i < 17; i++) {
        if (header_bits[i]) {
          sum++;
        }
      }
      header_bits[17] = sum % 2;
      dout << header_bits[17];

      // last 6 bits are zeros
      for (int i = 18; i < 24; i++) {
        header_bits[i] = 0;
        dout << header_bits[i];
      }
      dout << std::endl;

      convolutional_encoding(header_bits, encoded_signal_header);
      interleave(encoded_signal_header, out, false);

      for (int k = 0; k < 48; k++) {
        dout << (int)out[k];
      }
      dout << std::endl;

      free(header_bits);
    }

    void generate_header_field_old(char* out, unsigned first_flag, unsigned alloc_idx)
    {
      dout << "tx header bits first_flag " << first_flag << " alloc " << alloc_idx << std::endl;

      for (int i = 0; i < 4; i++){

        out[i * 12] = get_bit(first_flag, 0);
        dout << (int)out[i * 12];

        for (int k = (i * 12 + 1); k < (i * 12 + 12); k++){
          out[k] = get_bit(alloc_idx, (k % 12) - 1);
          dout << (int)out[k];
        }
      }
      dout << std::endl;

    }

    void interleave(const char* in, char* out, bool reverse)
    {

      int n_cbps = 48;
      int first[n_cbps];
      int second[n_cbps];
      int s = std::max(1 / 2, 1);

      for (int j = 0; j < n_cbps; j++) {
        first[j] = s * (j / s) + ((j + int(floor(16.0 * j / n_cbps))) % s);
      }

      for (int i = 0; i < n_cbps; i++) {
        second[i] = 16 * i - (n_cbps - 1) * int(floor(16.0 * i / n_cbps));
      }

      for (int i = 0; i < 1; i++) {
        for (int k = 0; k < n_cbps; k++) {
          if (reverse) {
            out[i * n_cbps + second[first[k]]] = in[i * n_cbps + k];
          } else {
            out[i * n_cbps + k] = in[i * n_cbps + second[first[k]]];
          }
        }
      }
    }


  private:
    enum { INSERT, COPY } d_state = INSERT;
    int d_packet_len;
    std::string d_length_tag_key;
    const bool d_debug;
    std::vector<gr::tag_t> d_tags;
    char* d_header_bits = (char*)malloc(sizeof(char) * 48);

    unsigned d_first_flag;
    unsigned d_alloc_idx;

    int d_offset = 0;
};

header_insert::sptr header_insert::make(int packet_len, std::string length_tag_key, bool debug)
{
  return gnuradio::get_initial_sptr(new header_insert_impl(packet_len, length_tag_key, debug));
}
