#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2021 gr-deepwive_v1 author.
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

import torch
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

import ipdb
import cv2
import numpy as np
from gnuradio import gr

class deepwive_v1_source(gr.sync_block):
    """
    docstring for block deepwive_v1_source
    """
    def __init__(self, source_fn, model_fn, snr=None):
        gr.sync_block.__init__(self,
            name="deepwive_v1_source",
            in_sig=None,
            out_sig=[np.complex64, np.uint8])
        self.snr = np.array([[snr]], dtype=np.float32)

        self.test_frame = self._get_video(source_fn)
        self._get_engine(model_fn)
        self._allocate_memory(self.test_frame)

    def _get_engine(self, model_fn):
        f = open(model_fn, 'rb')
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))

        engine = runtime.deserialize_cuda_engine(f.read())
        self.context = engine.create_execution_context()

    def _allocate_memory(self, input_sample):
        self.output = np.empty([1, 240, input_sample.shape[2]//16, input_sample.shape[3]//16], dtype=np.float32)

        self.d_input_img = cuda.mem_alloc(input_sample.nbytes)
        self.d_input_snr = cuda.mem_alloc(self.snr.nbytes)
        self.d_output = cuda.mem_alloc(self.output.nbytes)

        self.bindings = [int(self.d_input_img), int(self.d_input_snr), int(self.d_output)]

        self.stream = cuda.Stream()

    def _get_video(self, source_fn):
        self.vid = cv2.VideoCapture(source_fn)
        flag, frame = self.vid.read()
        frame = np.swapaxes(frame, 0, 1)
        frame = np.swapaxes(frame, 0, 2)
        frame = np.expand_dims(frame, axis=0) / 255.
        frame = np.ascontiguousarray(frame, dtype=np.float32)
        return frame

    def key_frame_encode(self, frame, snr):
        cuda.memcpy_htod_async(self.d_input_img, frame, self.stream)
        cuda.memcpy_htod_async(self.d_input_snr, snr, self.stream)
        self.context.execute_async_v2(self.bindings, self.stream.handle, None)
        cuda.memcpy_dtoh_async(self.output, self.d_output, self.stream)
        self.stream.synchronize()
        return self.output

    def test_work(self):
        codeword = self.key_frame_encode(self.test_frame, self.snr)

    def work(self, input_items, output_items):
        # out = output_items[0]
        # ipdb.set_trace()
        # codeword = self.key_frame_encode(self.test_frame, self.snr)
        # out[:] = whatever
        return len(output_items[0])

