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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <gnuradio/io_signature.h>
#include <deepwive_v1/ofdm_sync_short.h>
#include <iostream>

using namespace gr::deepwive_v1;

static const int MIN_GAP = 480;
static const int MAX_SAMPLES = 540 * 80;

class ofdm_sync_short_impl : public ofdm_sync_short
{

public:
    ofdm_sync_short_impl(double threshold, unsigned int min_plateau, bool log, bool debug)
        : block("ofdm_sync_short",
                gr::io_signature::make3(3, 3, sizeof(gr_complex), sizeof(gr_complex), sizeof(float)),
                gr::io_signature::make(1, 1, sizeof(gr_complex))),
          d_log(log),
          d_debug(debug),
          d_state(SEARCH),
          d_plateau(0),
          d_freq_offset(0),
          d_copied(0),
          MIN_PLATEAU(min_plateau),
          d_threshold(threshold)
    {
        set_tag_propagation_policy(block::TPP_DONT);
    }

    int general_work(int noutput_items,
                     gr_vector_int& ninput_items,
                     gr_vector_const_void_star& input_items,
                     gr_vector_void_star& output_items)
    {

        const gr_complex* in = (const gr_complex*)input_items[0];
        const gr_complex* in_abs = (const gr_complex*)input_items[1];
        const float* in_cor = (const float*)input_items[2];
        gr_complex* out = (gr_complex*)output_items[0];

        int noutput = noutput_items;
        int ninput =
            std::min(std::min(ninput_items[0], ninput_items[1]), ninput_items[2]);

        // dout << "SHORT noutput : " << noutput << " ninput: " << ninput_items[0] <<
        // std::endl;

        switch (d_state) {

        case SEARCH: {
            int i;

            for (i = 0; i < ninput; i++) {
                if (in_cor[i] > d_threshold) {
                    if (d_plateau < MIN_PLATEAU) {
                        d_plateau++;

                    } else {
                        d_state = COPY;
                        d_copied = 0;
                        d_freq_offset = arg(in_abs[i]) / 16;
                        d_plateau = 0;
                        insert_tag(nitems_written(0), d_freq_offset, nitems_read(0) + i);
                        // dout << "SHORT Frame! " << nitems_read(0) + i << std::endl;
                        break;
                    }
                } else {
                    d_plateau = 0;
                }
            }

            consume_each(i);
            return 0;
        }

        case COPY: {

            int o = 0;
            while (o < ninput && o < noutput && d_copied < MAX_SAMPLES) {
                if (in_cor[o] > d_threshold) {
                    if (d_plateau < MIN_PLATEAU) {
                        d_plateau++;

                        // there's another frame
                    } else if (d_copied > MIN_GAP) {
                        dout << "SHORT copied " << d_copied << std::endl;
                        d_copied = 0;
                        d_plateau = 0;
                        d_freq_offset = arg(in_abs[o]) / 16;
                        insert_tag(nitems_written(0) + o, d_freq_offset, nitems_read(0) + o);
                        // dout << "SHORT Frame! " << nitems_read(0) + o << std::endl;
                        break;
                    }

                } else {
                    d_plateau = 0;
                }

                out[o] = in[o] * exp(gr_complex(0, -d_freq_offset * d_copied));
                o++;
                d_copied++;
            }

            if (d_copied == MAX_SAMPLES) {
                dout << "max samples reached" << std::endl;
                d_state = SEARCH;
            }


            consume_each(o);
            return o;
        }
        }

        throw std::runtime_error("sync short: unknown state");
        return 0;
    }

    // void forecast(int noutput_items, gr_vector_int& ninput_items_required)
    // {
    //     ninput_items_required[0] = noutput_items;
    //     ninput_items_required[1] = noutput_items;
    //     ninput_items_required[2] = noutput_items;
    // }

    void insert_tag(uint64_t item, double freq_offset, uint64_t input_item)
    {
        mylog(boost::format("frame start at in: %2% out: %1%") % item % input_item);
        // dout << "symbol start " << item << std::endl;

        const pmt::pmt_t key = pmt::string_to_symbol("symbol_start");
        const pmt::pmt_t value = pmt::from_double(freq_offset);
        const pmt::pmt_t srcid = pmt::string_to_symbol(name());
        add_item_tag(0, item, key, value, srcid);
    }

private:
    enum { SEARCH, COPY } d_state;
    int d_copied;
    int d_plateau;
    float d_freq_offset;
    const double d_threshold;
    const bool d_log;
    const bool d_debug;
    const unsigned int MIN_PLATEAU;
};

ofdm_sync_short::sptr
ofdm_sync_short::make(double threshold, unsigned int min_plateau, bool log, bool debug)
{
    return gnuradio::get_initial_sptr(
        new ofdm_sync_short_impl(threshold, min_plateau, log, debug));
}
