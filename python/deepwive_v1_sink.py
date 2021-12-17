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
from collections import Counter
from itertools import combinations
import ipdb

import torch
import torch.nn as nn
import torch.nn.functional as F

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


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(self, channels, kernel_size, sigma, dim=2):
        super().__init__()
        self.padding = kernel_size // 2
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32)
                for size in kernel_size])
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups, padding=self.padding)


def ss_warp(x, flo):
    """
    warp an scaled space volume (x) back to im1, according to scale space flow
    x: [B, C, D, H, W]
    flo: [B, 3, 1, H, W] ss flow
    """
    B, C, D, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    zz = torch.zeros_like(xx)
    grid = torch.cat((xx, yy, zz), 1).float()
    grid = grid.unsqueeze(2)

    if x.is_cuda:
        grid = grid.to(x.device)
    # vgrid = Variable(grid) + flo
    grid.requires_grad = True
    vgrid = grid + flo

    # scale grid to [-1,1]
    vgrid[:, 0, :, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W-1, 1) - 1.0
    vgrid[:, 1, :, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H-1, 1) - 1.0
    vgrid[:, 2, :, :, :] = 2.0 * vgrid[:, 2, :, :].clone() - 1.0

    vgrid = vgrid.permute(0, 2, 3, 4, 1)
    output = nn.functional.grid_sample(x, vgrid, align_corners=True)
    return output.squeeze(2)


def generate_ss_volume(x, sigma, kernel_size, M):
    B, C, H, W = x.size()
    out = [x]
    for i in range(M):
        kernel = GaussianSmoothing(C, kernel_size, (2**i) * sigma).to(x.device)
        out.append(kernel(x))
    out = torch.stack(out, dim=2)
    return out


def perms_without_reps(s):
    partitions = list(Counter(s).items())

    def _helper(idxset, i):
        if len(idxset) == 0:
            yield ()
            return
        for pos in combinations(idxset, partitions[i][1]):
            for res in _helper(idxset - set(pos), i+1):
                yield (pos,) + res

    n = len(s)
    for poses in _helper(set(range(n)), 0):
        out = [None] * n
        for i, pos in enumerate(poses):
            for idx in pos:
                out[idx] = partitions[i][0]
        yield out


def split_list_by_val(x, s):
    size = len(x)
    idx_list = [idx + 1 for idx, val in enumerate(x) if val == s]
    res = [x[i:j] for i, j in zip([0] + idx_list, idx_list + [size])]
    return res


class deepwive_v1_sink(gr.basic_block):
    """
    docstring for block deepwive_v1_sink
    """

    def __init__(self, source_fn, model_fn, model_cout, packet_len=96,
                 snr=20, num_chunks=20, gop_size=5,
                 use_fp16=False):
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
        (self.key_decoder,
         self.interp_decoder) = self._get_context(model_fn)
        # self.key_decoder = self._get_context(model_fn)

        self.num_chunks = num_chunks
        self.chunk_size = model_cout // num_chunks
        self.gop_size = gop_size
        self.ssf_sigma = 0.01
        self.ssf_levels = 5

        self.video_frames = self._get_video_frames(source_fn)
        self.video_frames = self.video_frames[:125]  # FIXME remove this in final version
        self.n_frames = len(self.video_frames)
        self.window_name = 'video_stream'
        self._open_window(self.frame_shape[3], self.frame_shape[2], self.window_name)

        self._allocate_memory()
        self._reset()

        self.prev_packet_idx = -1

    def _get_runtime(self, logger):
        self.runtime = trt.Runtime(logger)
        self.stream = cuda.Stream()

    def _get_context(self, model_fns):
        # f = open(model_fn, 'rb')
        # engine = self.runtime.deserialize_cuda_engine(f.read())
        # context = engine.create_execution_context()
        # return context

        contexts = []
        for fn in model_fns:
            f = open(fn, 'rb')
            engine = self.runtime.deserialize_cuda_engine(f.read())
            ctx = engine.create_execution_context()
            contexts.append(ctx)
        return contexts

    def _allocate_memory(self):
        self.codeword_addr = cuda.mem_alloc(np.empty(self.codeword_shape, dtype=self.target_dtype).nbytes)
        self.frame_addr = cuda.mem_alloc(np.empty(self.frame_shape, dtype=self.target_dtype).nbytes)
        self.interp_output_addr = cuda.mem_alloc(np.empty(self.interp_output_shape, dtype=self.target_dtype).nbytes)
        self.snr_addr = cuda.mem_alloc(self.snr.nbytes)

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

    def _get_bw_set(self):
        bw_set = [1] * self.num_chunks + [0] * (self.gop_size-2)
        bw_set = perms_without_reps(bw_set)
        bw_set = [split_list_by_val(bw, 0) for bw in bw_set]
        self.bw_set = [[(sum(bw)*self.chunk_size) for bw in alloc] for alloc in bw_set]

    def _key_frame_decode(self, codeword, snr):
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

    def _interp_decode(self, codeword, ref_left, ref_right, snr):
        threading.Thread.__init__(self)
        self.cfx.push()

        output = np.empty(self.interp_output_shape, dtype=self.target_dtype)

        bindings = [int(self.codeword_addr),
                    int(self.snr_addr),
                    int(self.interp_output_addr)]

        cuda.memcpy_htod_async(self.codeword_addr, codeword, self.stream)
        cuda.memcpy_htod_async(self.snr_addr, snr, self.stream)
        self.key_decoder.execute_async_v2(bindings, self.stream.handle, None)
        cuda.memcpy_dtoh_async(output, self.interp_output_addr, self.stream)

        # TODO can probably synchronize many frames at once; depending on optimization
        self.stream.synchronize()

        interp_decoder_out = torch.from_numpy(output)
        f1, f2, a, r = torch.chunk(interp_decoder_out, chunks=4, dim=1)

        a = F.softmax(a, dim=1)
        a1, a2, a3 = torch.chunk(a, chunks=3, dim=1)
        r = torch.sigmoid(r)

        a1 = a1.repeat_interleave(3, dim=1)
        a2 = a2.repeat_interleave(3, dim=1)
        a3 = a3.repeat_interleave(3, dim=1)

        pred_vol1 = generate_ss_volume(torch.from_numpy(ref_left), args.ss_sigma, 3, args.ss_levels)
        pred_vol2 = generate_ss_volume(torch.from_numpy(ref_right), args.ss_sigma, 3, args.ss_levels)
        pred_1 = ss_warp(pred_vol1, f1.unsqueeze(2))
        pred_2 = ss_warp(pred_vol2, f2.unsqueeze(2))

        pred = (a1 * pred_1 + a2 * pred_2 + a3 * r).numpy()
        self.cfx.pop()
        return pred

    def _decode_gop(self, codewords_vec, bw_allocation_idx, init_frame=None, first=False):
        codewords = codewords_vec.reshape(self.codeword_shape)

        if first:
            frames = [
                self._key_frame_decode(codewords, self.snr)
            ]
        else:
            frames = [init_frame] + [None] * (self.gop_size-1)
            bw_allocation = self.bw_set[bw_allocation_idx]
            split_idxs = [sum(bw_allocation[:i]) for i in range(1, self.gop_size-1)]
            codewords = np.split(codewords, split_idxs)

            last_frame_codeword = codewords[0]
            zero_pad = np.zeros((1, self.model_cout-last_frame_codeword.shape[1], self.codeword_shape[2], self.codeword_shape[3]))
            last_frame_codeword = np.concatenate((last_frame_codeword, zero_pad), dtype=self.target_dtype)
            last_frame = self._key_frame_decode(last_frame_codeword, self.snr)
            frames[-1] = last_frame

            for pred_idx in (2, 1, 3):
                if pred_idx == 2:
                    dist = 2
                else:
                    dist = 1

                codeword = codewords[pred_idx]
                zero_pad = np.zeros((1, self.model_cout-codeword.shape[1], self.codeword_shape[2], self.codeword_shape[3]))
                codeword = np.concatenate((codeword, zero_pad), dtype=self.target_dtype)
                decoded_frame = self._interp_decode(codeword, frames[pred_idx-dist], frames[pred_idx+dist], self.snr)
                frames[pred_idx] = decoded_frame

        return frames

    def msg_handler(self, msg_pmt):
        tags = pmt.to_python(pmt.car(msg_pmt))
        payload_in = pmt.to_python(pmt.cdr(msg_pmt))

        first_flag = int(tags['first'])
        self.first_buffer.append(first_flag)

        alloc_idx = int(tags['alloc_idx'])
        self.alloc_buffer.append(alloc_idx)

        '''
        TODO need to do majority decoding of first flag and alloc_idx over all packets of gop codeword
        if gop codeword is not complete need to still be able to differentiate gops
        '''

        # print('Rx frame_idx: {}, packet_idx: {}'.format(frame_idx, packet_idx))

        received_IQ = [[pair.real, pair.imag] for pair in payload_in]
        received_IQ = np.array(received_IQ)
        self.curr_frame_packets.append(received_IQ)

        is_first = (first_flag == 1)
        if is_first and len(self.curr_frame_packets) == self.n_packets:
            # start_time = time.time()

            codeword = np.concatenate(self.curr_frame_packets, axis=0)[:self.ch_uses-self.n_padding]
            codeword = np.ascontiguousarray(codeword, dtype=self.target_dtype).reshape(self.codeword_shape)

            if is_first:
                decoded_frames = self._decode_gop(codeword, alloc_idx, first=True)
                self.prev_last = decoded_frames[0]
            else:
                decoded_frames = self._decode_gop(codeword, alloc_idx, init_frame=self.prev_last)
                self.prev_last = decoded_frames[-1]

            self.frame_buffer.extend(decoded_frames)
            # decoded_frame = self._key_frame_decode(codeword, self.snr)
            # mse = np.mean((curr_frame - decoded_frame) ** 2)

            # drawnow(plt.imshow, False, False, True, frame)

            for frame in decoded_frames:
                cv2.imshow(self.window_name, to_np_img(frame))

                if cv2.waitKey(1) == 13:
                    cv2.destroyAllWindows()

            # end_time = time.time()
            # print('decode time: {}'.format(end_time - start_time))
            self._reset()

        # elif (packet_idx == self.n_packets - 1) and any([v is None for v in self.curr_frame_packets]):
        #     self._reset()

    def _reset(self):
        # TODO update this to reflect full scheme
        self.curr_frame_packets = []
        self.frame_buffer = []
        self.first_buffer = []
        self.alloc_buffer = []
