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

#ifndef INCLUDED_DEEPWIVE_V1_OFDM_SYNC_LONG_H
#define INCLUDED_DEEPWIVE_V1_OFDM_SYNC_LONG_H

#include <deepwive_v1/api.h>
#include <gnuradio/block.h>

#define dout d_debug&& std::cout
#define mylog(msg)                      \
    do {                                \
        if (d_log) {                    \
            GR_LOG_INFO(d_logger, msg); \
        }                               \
    } while (0);

namespace gr {
  namespace deepwive_v1 {

    /*!
     * \brief <+description+>
     *
     */
    class DEEPWIVE_V1_API ofdm_sync_long : virtual public block
    {
    public:
      typedef boost::shared_ptr<ofdm_sync_long> sptr;
      static sptr make (unsigned int sync_length, bool log = false, bool debug = false);
    private:
    };

  } // namespace deepwive_v1
} // namespace gr

#endif /* INCLUDED_DEEPWIVE_V1_OFDM_SYNC_LONG_H */

