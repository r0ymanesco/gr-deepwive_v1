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


import cv2
import time
import numpy as np
from drawnow import drawnow
import matplotlib.pyplot as plt
import ipdb

import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import threading

import pmt
from gnuradio import gr


def to_np_img(img):
    img = np.squeeze(img, axis=0)
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 0, 1)
    # img = img[:, :, [2, 1, 0]]
    return img.astype(np.float32)


class deepwive_v1_sink(gr.basic_block):
    """
    docstring for block deepwive_v1_sink
    """

    def __init__(self, source_fn, model_fn, model_cout, packet_len=96, snr=20, use_fp16=False):
        gr.sync_block.__init__(self,
                               name="deepwive_v1_sink",
                               in_sig=None,
                               out_sig=None)

        self.cfx = cuda.Device(0).make_context()

        in_port_name = 'pdu_in'
        self.message_port_register_in(pmt.intern(in_port_name))
        self.set_msg_handler(pmt.intern(in_port_name), self.msg_handler)

        if use_fp16:
            self.target_dtype = np.float16
            assert 'fp16' in model_fn
        else:
            self.target_dtype = np.float32

        self.source_fn = source_fn
        self.model_fn = model_fn
        self.model_cout = model_cout
        self.packet_len = packet_len
        self.snr = np.array([[snr]], dtype=self.target_dtype)
        self.use_fp16 = use_fp16

        self._get_runtime(trt.Logger(trt.Logger.WARNING))
        self.key_decoder = self._get_context(model_fn)

        self.video_frames = self._get_video_frames(source_fn)
        self.video_frames = self.video_frames[:125]  # FIXME remove this in final version
        self.n_frames = len(self.video_frames)
        self.window_name = 'video_stream'
        self._open_window(self.frame_shape[3], self.frame_shape[2], self.window_name)

        self._allocate_memory()

        self._reset()

        self.prev_packet_idx = -1

    def _reset(self):
        self.curr_frame_packets = [None] * self.n_packets

    def _get_runtime(self, logger):
        self.runtime = trt.Runtime(logger)
        self.stream = cuda.Stream()

    def _get_context(self, model_fn):
        f = open(model_fn, 'rb')
        engine = self.runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()
        return context

    def _allocate_memory(self):
        self.codeword_addr = cuda.mem_alloc(
            np.empty(self.codeword_shape, dtype=self.target_dtype).nbytes
        )

        self.frame_addr = cuda.mem_alloc(
            np.empty(self.frame_shape, dtype=self.target_dtype).nbytes
        )

        self.interp_output_addr = cuda.mem_alloc(
            np.empty(self.interp_output_shape, dtype=self.target_dtype).nbytes
        )

        self.snr_addr = cuda.mem_alloc(
            self.snr.nbytes
        )

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
            frame_shape = frame.shape

            flag, frame = self.video.read()

        self.frame_shape = frame_shape
        self.interp_output_shape = (1, 12, frame_shape[2], frame_shape[3])
        self.codeword_shape = (1, self.model_cout, frame_shape[2]//16, frame_shape[3]//16)

        self.ch_uses = np.prod(self.codeword_shape[1:]) // 2
        self.n_packets = self.ch_uses // self.packet_len
        self.n_padding = (self.packet_len - (self.ch_uses % self.packet_len)) % self.packet_len
        return frames

    def _open_window(self, width, height, window_name):
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, width, height)
        cv2.moveWindow(window_name, 0, 0)
        cv2.setWindowTitle(window_name, window_name)

    def key_frame_decode(self, codeword, snr):
        threading.Thread.__init__(self)
        self.cfx.push()

        output = np.empty(self.frame_shape, dtype=self.target_dtype)

        bindings = [int(self.codeword_addr),
                    int(self.snr_addr),
                    int(self.frame_addr)]

        cuda.memcpy_htod_async(self.codeword_addr, codeword, self.stream)
        cuda.memcpy_htod_async(self.snr_addr, snr, self.stream)
        self.key_decoder.execute_async_v2(bindings, self.stream.handle, None)
        cuda.memcpy_dtoh_async(output, self.frame_addr, self.stream)

        # TODO can probably synchronize many frames at once; depending on optimization
        self.stream.synchronize()

        self.cfx.pop()
        return output

    def msg_handler(self, msg_pmt):
        # ipdb.set_trace()
        tags = pmt.to_python(pmt.car(msg_pmt))
        payload_in = pmt.to_python(pmt.cdr(msg_pmt))

        frame_idx = tags['frame_idx']
        packet_idx = tags['packet_idx']
        # curr_frame = self.video_frames[frame_idx]

        # print('Rx frame_idx: {}, packet_idx: {}'.format(frame_idx, packet_idx))

        # if packet_idx - self.prev_packet_idx != 1:
        #     ipdb.set_trace()

        # self.prev_packet_idx = packet_idx

        # if packet_idx > 64:
        #     ipdb.set_trace()

        received_IQ = [[pair.real, pair.imag] for pair in payload_in]
        received_IQ = np.array(received_IQ)
        self.curr_frame_packets[packet_idx] = received_IQ

        if (packet_idx == self.n_packets - 1) and not any([v is None for v in self.curr_frame_packets]):
            start_time = time.time()

            codeword = np.concatenate(self.curr_frame_packets, axis=0)[:self.ch_uses-self.n_padding]
            codeword = np.ascontiguousarray(codeword, dtype=self.target_dtype).reshape(self.codeword_shape)
            decoded_frame = self.key_frame_decode(codeword, self.snr)
            # mse = np.mean((curr_frame - decoded_frame) ** 2)
            frame = to_np_img(decoded_frame)

            # drawnow(plt.imshow, False, False, True, frame)

            cv2.imshow(self.window_name, frame)

            if cv2.waitKey(1) == 13:
                cv2.destroyAllWindows()

            end_time = time.time()
            print('decode time: {}'.format(end_time - start_time))

        elif (packet_idx == self.n_packets - 1) and any([v is None for v in self.curr_frame_packets]):
            self._reset()
