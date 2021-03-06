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

#include <gnuradio/sptr_magic.h>
#include <gnuradio/types.h>
#include <stdexcept>
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <gnuradio/fft/fft.h>
#include <gnuradio/filter/fir_filter.h>
#include <gnuradio/io_signature.h>
#include <deepwive_v1/ofdm_sync_long.h>

#include <list>
#include <tuple>

using namespace gr::deepwive_v1;
using namespace std;

bool compare_abs(const std::pair<gr_complex, int>& first,
                 const std::pair<gr_complex, int>& second)
{
    return abs(get<0>(first)) > abs(get<0>(second));
}

class ofdm_sync_long_impl : public ofdm_sync_long
{
  public:
    ofdm_sync_long_impl(unsigned int sync_length,
                        bool log,
                        bool debug)
      : block("ofdm_sync_long",
              gr::io_signature::make(1, 1, sizeof(gr_complex)),
              gr::io_signature::make(1, 1, sizeof(gr_complex))),
        d_fir(gr::filter::kernel::fir_filter_ccc(1, LONG)),
        d_log(log),
        d_debug(debug),
        d_offset(0),
        d_state(SYNC),
        SYNC_LENGTH(sync_length)
    {
      set_tag_propagation_policy(block::TPP_DONT);
      d_correlation = gr::fft::malloc_complex(8192);
    }

    ~ofdm_sync_long_impl()
    {
      gr::fft::free(d_correlation);
    }

    int general_work(int noutput,
                     gr_vector_int& ninput_items,
                     gr_vector_const_void_star& input_items,
                     gr_vector_void_star& output_items)
    {
      const gr_complex* in = (const gr_complex*)input_items[0];
      gr_complex* out = (gr_complex*)output_items[0];
      // float* corr = (float*)output_items[1];

      dout << "LONG ninput[0] " << ninput_items[0]
        << " noutput " << noutput << " state " << d_state << std::endl;

      int ninput = std::min(ninput_items[0], 8192);
      const uint64_t nread = nitems_read(0);
      get_tags_in_range(d_tags, 0, nread, nread + ninput);
      dout << "nread " << nread << std::endl;

      if (d_tags.size()){
        std::sort(d_tags.begin(), d_tags.end(), gr::tag_t::offset_compare);

        const uint64_t offset = d_tags.front().offset;
        dout << "offset " << offset << std::endl;

        if (offset > nread){
          ninput = offset - nread;
          dout << "copy ninput " << ninput << std::endl;
        }
        else {
          if (d_offset && (d_state == SYNC)){
            throw std::runtime_error("wtf");
          }
          if (d_state == COPY){
            dout << "goto reset" << std::endl;
            d_state = RESET;
          }
          d_freq_offset_short = pmt::to_double(d_tags.front().value);
        }

      }else {
        dout << "no tags" << std::endl;
      }

      int i = 0;
      int o = 0;

      switch (d_state) {

        case SYNC: {
          // d_fir.filterN(
          //   d_correlation, in, std::min(SYNC_LENGTH, std::max(ninput - 63, 0)));
          d_fir.filterN(d_correlation, in, SYNC_LENGTH);

          while(i + 63 < ninput){
            d_cor.push_back(pair<gr_complex, int>(d_correlation[i], d_offset));

            i++;
            d_offset++;

            if(d_offset == SYNC_LENGTH){
              search_frame_start();
              // mylog(boost::format("LONG: frame start at %1%") % d_frame_start);
              dout << "LONG: frame start at " << d_frame_start << std::endl;
              d_remainder = SYNC_LENGTH - d_frame_start;
              d_offset = d_frame_start;
              d_count = 0;
              d_state = COPY;
              // for (int k = 0; k < 64; k++){
              //   corr[o] = (float)abs(d_correlation[k]);
              //   o++;
              // }
              break;
            }
          }

          dout << "produced: " << o << " consumed: " << d_frame_start << std::endl;

          consume(0, d_frame_start);
          return o;
        }

        case COPY: {
          while (i < ninput && o < noutput){
            int rel = d_offset - d_frame_start;

            if (rel == 0){
              add_item_tag(0,
                           nitems_written(0),
                           pmt::string_to_symbol("data_start"),
                           pmt::from_double(d_freq_offset_short - d_freq_offset),
                           pmt::string_to_symbol(name()));
            }

            if (rel >= 0 && (rel < 128 || ((rel - 128) % 80) > 15)){
              out[o] = in[i] * exp(gr_complex(0, d_offset * d_freq_offset));
              o++;
              d_count++;
            }

            i++;
            d_offset++;
          }

          dout << "produced: " << o << " consumed: " << i << std::endl;

          consume(0, i);
          return o;
        }

        case RESET: {
          while (o < noutput){
            if ((d_count % 64) == 0){
              d_offset = 0;
              d_state = SYNC;
              break;
            }
            else{
              out[o] = 0;
              o++;
              d_count++;
            }
          }

          dout << "produced: " << o << " consumed: " << i << std::endl;

          consume(0, 0);
          return o;
        }
      }

      throw std::runtime_error("sync long: unknown state");
      return 0;
    }

    void forecast(int noutput_items, gr_vector_int& ninput_items_required)
    {
      // ninput_items_required[0] = std::max(SYNC_LENGTH, noutput_items);

      // in sync state we need at least a symbol to correlate with the pattern
      if (d_state == SYNC){
        ninput_items_required[0] = SYNC_LENGTH;
      }else{
        ninput_items_required[0] = noutput_items;
      }
    }

    void search_frame_start()
    {

        // sort list (highest correlation first)
        assert(d_cor.size() == SYNC_LENGTH);
        d_cor.sort(compare_abs);

        // copy list in vector for nicer access
        vector<pair<gr_complex, int>> vec(d_cor.begin(), d_cor.end());
        d_cor.clear();

        // in case we don't find anything use SYNC_LENGTH
        d_frame_start = SYNC_LENGTH;

        for (int i = 0; i < 3; i++) {
            for (int k = i + 1; k < 4; k++) {
                gr_complex first;
                gr_complex second;
                if (get<1>(vec[i]) > get<1>(vec[k])) {
                    first = get<0>(vec[k]);
                    second = get<0>(vec[i]);
                } else {
                    first = get<0>(vec[i]);
                    second = get<0>(vec[k]);
                }
                int diff = abs(get<1>(vec[i]) - get<1>(vec[k]));
                dout << "sync diff " << diff << endl;
                if (diff == 64) {
                    d_frame_start = min(get<1>(vec[i]), get<1>(vec[k]));
                    d_freq_offset = arg(first * conj(second)) / 64;
                    dout << "found nice match" << std::endl;
                    return;

                } else if (diff == 63) {
                    d_frame_start = min(get<1>(vec[i]), get<1>(vec[k]));
                    d_freq_offset = arg(first * conj(second)) / 63;
                } else if (diff == 65) {
                    d_frame_start = min(get<1>(vec[i]), get<1>(vec[k]));
                    d_freq_offset = arg(first * conj(second)) / 65;
                }
            }
        }

    }

  private:
    enum { SYNC, COPY, RESET } d_state;
    int d_count;
    int d_offset;
    int d_frame_start;
    float d_freq_offset;
    double d_freq_offset_short;
    int d_remainder;

    gr_complex* d_correlation;
    list<pair<gr_complex, int>> d_cor;
    std::vector<gr::tag_t> d_tags;
    gr::filter::kernel::fir_filter_ccc d_fir;

    const bool d_log;
    const bool d_debug;
    const int SYNC_LENGTH;

    static const std::vector<gr_complex> LONG;
};

ofdm_sync_long::sptr ofdm_sync_long::make(unsigned int sync_length, bool log, bool debug)
{
  return gnuradio::get_initial_sptr(new ofdm_sync_long_impl(sync_length, log, debug));
}

const std::vector<gr_complex> ofdm_sync_long_impl::LONG = {
gr_complex(-0.0455, -1.0679), gr_complex(0.3528, -0.9865),
gr_complex(0.8594, 0.7348),   gr_complex(0.1874, 0.2475),
gr_complex(0.5309, -0.7784),  gr_complex(-1.0218, -0.4897),
gr_complex(-0.3401, -0.9423), gr_complex(0.8657, -0.2298),
gr_complex(0.4734, 0.0362),   gr_complex(0.0088, -1.0207),
gr_complex(-1.2142, -0.4205), gr_complex(0.2172, -0.5195),
gr_complex(0.5207, -0.1326),  gr_complex(-0.1995, 1.4259),
gr_complex(1.0583, -0.0363),  gr_complex(0.5547, -0.5547),
gr_complex(0.3277, 0.8728),   gr_complex(-0.5077, 0.3488),
gr_complex(-1.1650, 0.5789),  gr_complex(0.7297, 0.8197),
gr_complex(0.6173, 0.1253),   gr_complex(-0.5353, 0.7214),
gr_complex(-0.5011, -0.1935), gr_complex(-0.3110, -1.3392),
gr_complex(-1.0818, -0.1470), gr_complex(-1.1300, -0.1820),
gr_complex(0.6663, -0.6571),  gr_complex(-0.0249, 0.4773),
gr_complex(-0.8155, 1.0218),  gr_complex(0.8140, 0.9396),
gr_complex(0.1090, 0.8662),   gr_complex(-1.3868, -0.0000),
gr_complex(0.1090, -0.8662),  gr_complex(0.8140, -0.9396),
gr_complex(-0.8155, -1.0218), gr_complex(-0.0249, -0.4773),
gr_complex(0.6663, 0.6571),   gr_complex(-1.1300, 0.1820),
gr_complex(-1.0818, 0.1470),  gr_complex(-0.3110, 1.3392),
gr_complex(-0.5011, 0.1935),  gr_complex(-0.5353, -0.7214),
gr_complex(0.6173, -0.1253),  gr_complex(0.7297, -0.8197),
gr_complex(-1.1650, -0.5789), gr_complex(-0.5077, -0.3488),
gr_complex(0.3277, -0.8728),  gr_complex(0.5547, 0.5547),
gr_complex(1.0583, 0.0363),   gr_complex(-0.1995, -1.4259),
gr_complex(0.5207, 0.1326),   gr_complex(0.2172, 0.5195),
gr_complex(-1.2142, 0.4205),  gr_complex(0.0088, 1.0207),
gr_complex(0.4734, -0.0362),  gr_complex(0.8657, 0.2298),
gr_complex(-0.3401, 0.9423),  gr_complex(-1.0218, 0.4897),
gr_complex(0.5309, 0.7784),   gr_complex(0.1874, -0.2475),
gr_complex(0.8594, -0.7348),  gr_complex(0.3528, 0.9865),
gr_complex(-0.0455, 1.0679),  gr_complex(1.3868, -0.0000),
};
