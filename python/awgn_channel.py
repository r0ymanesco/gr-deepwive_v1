#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2022 gr-deepwive_v1 author.
#
# This is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3, or (at your option)
# any later version.
#
# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this software; see the file COPYING.  If not, write to
# the Free Software Foundation, Inc., 51 Franklin Street,
# Boston, MA 02110-1301, USA.
#


import numpy as np
from gnuradio import gr


class awgn_channel(gr.sync_block):
    """
    docstring for block awgn_channel
    """

    def __init__(self, snr):
        gr.sync_block.__init__(self,
                               name="awgn_channel",
                               in_sig=[np.csingle],
                               out_sig=[np.csingle])

        self.snr = snr

    def work(self, input_items, output_items):
        symbols_in = input_items[0]
        out = output_items[0]
        noise_std = np.sqrt(10 ** (-self.snr/10))
        awgn = noise_std * np.random.randn(len(symbols_in))
        # out[:] = symbols_in[:] + awgn

        for i, symbol in enumerate(symbols_in):
            out[i] = symbol + awgn[i]

        return len(output_items[0])
