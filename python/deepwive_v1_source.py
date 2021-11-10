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


import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

import pmt
from gnuradio import gr

import cv2
import numpy as np
import ipdb


class deepwive_v1_source(gr.sync_block):
    """
    docstring for block deepwive_v1_source
    """
    def __init__(self, source_fn, model_fn, model_cout, packet_len=96, snr=20,
                 use_fp16=False, test_mode=False):
        gr.sync_block.__init__(self,
            name="deepwive_v1_source",
            in_sig=None,
            out_sig=[np.csingle, np.uint8])

        if use_fp16:
            self.target_dtype = np.float16
        else:
            self.target_dtype = np.float32

        self.source_fn = source_fn
        self.model_fn = model_fn
        self.model_cout = model_cout
        self.packet_len = packet_len
        self.snr = np.array([[snr]], dtype=self.target_dtype)
        self.use_fp16 = use_fp16

        if not test_mode:
            self._get_runtime(trt.Logger(trt.Logger.WARNING))
            self.key_encoder = self._get_context(model_fn)

            self.video_frames = self._get_video_frames(source_fn)
            self.n_frames = len(self.video_frames)
            self.frame_idx = -1
            self.pair_idx = 0
            self.packet_idx = 0

    def _get_runtime(self, logger):
        self.runtime = trt.Runtime(logger)
        self.stream = cuda.Stream()

    def _get_context(self, model_fn):
        f = open(model_fn, 'rb')
        engine = self.runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()
        return context

    def _allocate_memory(self, samples):
        allocations = []
        bindings = []
        for sample in samples:
            alloc = cuda.mem_alloc(sample.nbytes)
            allocations.append(alloc)
            bindings.append(int(alloc))
        return allocations, bindings

    def _get_video_frames(self, source_fn):
        self.video = cv2.VideoCapture(source_fn)
        flag, frame = self.video.read()
        assert flag
        frames = []
        while flag:
            frame = np.swapaxes(frame, 0, 1)
            frame = np.swapaxes(frame, 0, 2)
            frame = np.expand_dims(frame, axis=0) / 255.
            frame = np.ascontiguousarray(frame, dtype=self.target_dtype)
            frames.append(frame)

            flag, frame = self.video.read()

        self.codeword_shape = [1, self.model_cout, frame.shape[2]//16, frame.shape[3]//16]
        self.ch_uses = np.prod(self.codeword_shape[1:]) // 2
        self.n_padding = self.packet_len - (self.ch_uses % self.packet_len)
        return frames

    def key_frame_encode(self, frame, snr):
        output = np.empty(self.codeword_shape, dtype=self.target_dtype)

        input_allocations, input_bindings = self._allocate_memory((frame, snr))
        output_allocations, output_bindings = self._allocate_memory((output))

        d_input_img = input_allocations[0]
        d_input_snr = input_allocations[1]
        d_output = output_allocations[0]
        bindings = input_bindings + output_bindings

        cuda.memcpy_htod_async(d_input_img, frame, self.stream)
        cuda.memcpy_htod_async(d_input_snr, snr, self.stream)
        self.key_encoder.execute_async_v2(bindings, self.stream.handle, None)
        cuda.memcpy_dtoh_async(output, d_output, self.stream)

        self.stream.synchronize()

        # TODO add block control to send fewer blocks
        ch_codeword = self.power_normalize(output)
        zero_pad = np.zeros((self.n_padding, 2), dtype=self.target_dtype)
        ch_input = np.concatenate((ch_codeword, zero_pad), axis=0, dtype=self.target_dtype)
        return ch_input

    def power_normalize(self, codeword):
        codeword = codeword.reshape(1, -1)
        ch_uses = codeword.shape[1]
        ch_input = (codeword / np.linalg.norm(codeword, ord=2, axis=1, keepdims=True)) * np.sqrt(ch_uses)
        ch_input = ch_input.reshape(-1, 2)
        return ch_input.astype(self.target_dtype)

    def _get_test_frame(self, source_fn):
        self.vid = cv2.VideoCapture(source_fn)
        flag, frame = self.vid.read()
        assert flag
        frame = np.swapaxes(frame, 0, 1)
        frame = np.swapaxes(frame, 0, 2)
        frame = np.expand_dims(frame, axis=0) / 255.
        frame = np.ascontiguousarray(frame, dtype=self.target_dtype)
        return frame

    def test_key_frame_encode(self, frame, snr):
        f = open(self.model_fn[0], 'rb')
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))

        engine = self.runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        output = np.empty([1, 240, self.test_frame.shape[2]//16, self.test_frame.shape[3]//16],
                          dtype=self.target_dtype)

        d_input_img = cuda.mem_alloc(self.test_frame.nbytes)
        d_input_snr = cuda.mem_alloc(self.snr.nbytes)
        d_output = cuda.mem_alloc(output.nbytes)

        bindings = [int(d_input_img), int(d_input_snr), int(d_output)]

        self.stream = cuda.Stream()

        cuda.memcpy_htod_async(d_input_img, frame, self.stream)
        cuda.memcpy_htod_async(d_input_snr, snr, self.stream)
        context.execute_async_v2(bindings, self.stream.handle, None)
        cuda.memcpy_dtoh_async(output, d_output, self.stream)
        self.stream.synchronize()
        return output

    def test_key_frame_decode(self, codeword, snr):
        f = open(self.model_fn[1], 'rb')

        engine = self.runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        output = np.empty(self.test_frame.shape, dtype=self.target_dtype)

        d_input_codeword = cuda.mem_alloc(codeword.nbytes)
        d_input_snr = cuda.mem_alloc(self.snr.nbytes)
        d_output = cuda.mem_alloc(output.nbytes)

        bindings = [int(d_input_codeword), int(d_input_snr), int(d_output)]

        cuda.memcpy_htod_async(d_input_codeword, codeword, self.stream)
        cuda.memcpy_htod_async(d_input_snr, snr, self.stream)
        context.execute_async_v2(bindings, self.stream.handle, None)
        cuda.memcpy_dtoh_async(output, d_output, self.stream)
        self.stream.synchronize()
        return output

    def test_work(self):
        self.test_frame = self._get_test_frame(self.source_fn)
        codeword = self.test_key_frame_encode(self.test_frame, self.snr)
        codeword_shape = codeword.shape
        codeword = codeword.reshape(1, -1)
        ch_uses = codeword.shape[1]
        ch_input = (codeword / np.linalg.norm(codeword, ord=2, axis=1, keepdims=True)) * np.sqrt(ch_uses)
        noise_stddev = np.sqrt(10**(-self.snr/10))
        awgn = np.random.randn(*ch_input.shape) * noise_stddev
        ch_output = ch_input + awgn.astype(self.target_dtype)
        decoder_input = ch_output.reshape(codeword_shape)
        decoded_frame = self.test_key_frame_decode(decoder_input, self.snr)
        mse = np.mean((decoded_frame - self.test_frame)**2)
        return self.test_frame, decoded_frame, mse

    def work(self, input_items, output_items):
        payload_out = output_items[0]
        header_out = output_items[1]

        for payload_idx in range(len(payload_out)):
            if self.pair_idx % self.ch_uses == 0:
                self.frame_idx = (self.frame_idx + 1) % self.n_frames
                self.pair_idx = 0
                self.packet_idx = 0

                self.curr_codeword = self.key_frame_encode(self.video_frames[self.frame_idx],
                                                           self.snr)

            if self.pair_idx % self.packet_len == 0:
                self.add_item_tag(0, payload_idx + self.nitems_written(0), pmt.intern('packet_len'), pmt.from_long(self.packet_len))
                self.add_item_tag(1, payload_idx + self.nitems_written(1), pmt.intern('packet_len'), pmt.from_long(self.packet_len))

                self.add_item_tag(0, payload_idx + self.nitems_written(0), pmt.intern('frame_idx'), pmt.from_long(self.frame_idx))
                self.add_item_tag(1, payload_idx + self.nitems_written(1), pmt.intern('frame_idx'), pmt.from_long(self.frame_idx))

                self.add_item_tag(0, payload_idx + self.nitems_written(0), pmt.intern('packet_idx'), pmt.from_long(self.packet_idx))
                self.add_item_tag(1, payload_idx + self.nitems_written(1), pmt.intern('packet_idx'), pmt.from_long(self.packet_idx))

                self.packet_idx += 1

            payload_out[payload_idx] = self.curr_codeword[self.pair_idx, 0] + self.curr_codeword[self.pair_idx, 1]*1j
            byte_out[payload_idx] = np.uint8(7)

            self.pair_idx += 1

        return len(output_items[0])
