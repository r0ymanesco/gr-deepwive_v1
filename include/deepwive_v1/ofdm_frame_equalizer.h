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

#ifndef INCLUDED_DEEPWIVE_V1_OFDM_FRAME_EQUALIZER_H
#define INCLUDED_DEEPWIVE_V1_OFDM_FRAME_EQUALIZER_H

#include <deepwive_v1/api.h>
#include <gnuradio/block.h>

namespace gr {
  namespace deepwive_v1 {

    /*!
     * \brief <+description of block+>
     * \ingroup deepwive_v1
     *
     */
    class DEEPWIVE_V1_API ofdm_frame_equalizer : virtual public gr::block
    {
     public:
      typedef boost::shared_ptr<ofdm_frame_equalizer> sptr;

      /*!
       * \brief Return a shared_ptr to a new instance of deepwive_v1::ofdm_frame_equalizer.
       *
       * To avoid accidental use of raw pointers, deepwive_v1::ofdm_frame_equalizer's
       * constructor is in a private implementation
       * class. deepwive_v1::ofdm_frame_equalizer::make is the public interface for
       * creating new instances.
       */
      static sptr make(double freq, double bw, int packet_len, bool log, bool debug);
      virtual void set_bandwidth(double bw) = 0;
      virtual void set_frequency(double freq) = 0;
    };

  } // namespace deepwive_v1
} // namespace gr

#endif /* INCLUDED_DEEPWIVE_V1_OFDM_FRAME_EQUALIZER_H */

