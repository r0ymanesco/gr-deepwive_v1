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
import math
import numbers
import numpy as np
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


class deepwive_v1_source(gr.sync_block):
    """
    docstring for block deepwive_v1_source
    """

    def __init__(self, source_fn, model_fn, model_cout, packet_len=96,
                 snr=20, num_chunks=20, gop_size=5,
                 use_fp16=False):
        gr.sync_block.__init__(self,
                               name="deepwive_v1_source",
                               in_sig=None,
                               out_sig=[np.csingle])

        test_mode = False
        self.cfx = cuda.Device(0).make_context()

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

        if not test_mode:
            self._get_runtime(trt.Logger(trt.Logger.WARNING))
            (self.key_encoder,
             self.interp_encoder,
             self.ssf_net,
             self.bw_allocator) = self._get_context(model_fn)

            self.video_frames = self._get_video_frames(source_fn)
            self.video_frames = self.video_frames[:125]  # FIXME remove this in final version
            self.n_frames = len(self.video_frames)
            self.frame_idx = -1
            self.pair_idx = 0
            self.packet_idx = 0

            self.num_chunks = num_chunks
            self.chunk_size = model_cout // num_chunks
            self.gop_size = gop_size
            self.ssf_sigma = 0.01
            self.ssf_levels = 5

            self._get_bw_set()

    def _get_runtime(self, logger):
        self.runtime = trt.Runtime(logger)
        self.stream = cuda.Stream()

    def _get_context(self, model_fns):
        contexts = []
        for fn in model_fns:
            f = open(fn, 'rb')
            engine = self.runtime.deserialize_cuda_engine(f.read())
            ctx = engine.create_execution_context()
            contexts.append(ctx)
        return contexts

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
            frame_shape = frame.shape

            flag, frame = self.video.read()

        self.frame_shape = frame_shape
        self.codeword_shape = [1, self.model_cout, frame_shape[2]//16, frame_shape[3]//16]
        self.ch_uses = np.prod(self.codeword_shape[1:]) // 2
        self.n_padding = (self.packet_len - (self.ch_uses % self.packet_len)) % self.packet_len
        return frames

    def _get_bw_set(self):
        bw_set = [1] * self.num_chunks + [0] * (self.gop_size-2)
        bw_set = perms_without_reps(bw_set)
        bw_set = [split_list_by_val(bw, 0) for bw in bw_set]
        self.bw_set = [[sum(bw) for bw in alloc] for alloc in bw_set]

    def key_frame_encode(self, frame, snr):
        threading.Thread.__init__(self)
        self.cfx.push()

        output = np.empty(self.codeword_shape, dtype=self.target_dtype)
        input_allocations, input_bindings = self._allocate_memory((frame, snr))
        output_allocations, output_bindings = self._allocate_memory((output))

        # TODO factor execution code for generalisation
        d_input_frame = input_allocations[0]
        d_input_snr = input_allocations[1]
        d_output = output_allocations[0]
        bindings = input_bindings + output_bindings

        cuda.memcpy_htod_async(d_input_frame, frame, self.stream)
        cuda.memcpy_htod_async(d_input_snr, snr, self.stream)
        self.key_encoder.execute_async_v2(bindings, self.stream.handle, None)
        cuda.memcpy_dtoh_async(output, d_output, self.stream)

        self.stream.synchronize()

        # TODO add block control to send fewer blocks
        ch_codeword = self.power_normalize(output)
        zero_pad = np.zeros((self.n_padding, 2), dtype=self.target_dtype)
        ch_input = np.concatenate((ch_codeword, zero_pad), axis=0).astype(self.target_dtype)

        self.cfx.pop()
        return ch_input

    def interp_frame_encode(self, frame, ref_left, ref_right, snr):
        threading.Thread.__init__(self)
        self.cfx.push()

        # TODO rewrite functions without torch
        vol1 = generate_ss_volume(torch.from_numpy(ref_left), self.ssf_sigma, 3, self.ssf_levels)
        vol2 = generate_ss_volume(torch.from_numpy(ref_right), self.ssf_sigma, 3, self.ssf_levels)

        flow1 = torch.from_numpy(self.ssf_estimate(frame, ref_left))
        flow2 = torch.from_numpy(self.ssf_estimate(frame, ref_right))

        w1 = ss_warp(vol1, flow1.unsqueeze(2))
        w2 = ss_warp(vol2, flow2.unsqueeze(2))

        r1 = torch.from_numpy(frame) - w1
        r2 = torch.from_numpy(frame) - w2

        interp_input = torch.cat((
            torch.from_numpy(frame),
            w1, w2,
            r1, r2,
            flow1, flow2
        ), dim=1).numpy()

        output = np.empty(self.codeword_shape, dtype=self.target_dtype)
        input_allocations, input_bindings = self._allocate_memory((interp_input, snr))
        output_allocations, output_bindings = self._allocate_memory((output))

        d_interp_input = input_allocations[0]
        d_input_snr = input_allocations[1]
        d_output = output_allocations[0]
        bindings = input_bindings + output_bindings

        cuda.memcpy_htod_async(d_interp_input, interp_input, self.stream)
        cuda.memcpy_htod_async(d_input_snr, snr, self.stream)
        self.key_encoder.execute_async_v2(bindings, self.stream.handle, None)
        cuda.memcpy_dtoh_async(output, d_output, self.stream)

        self.stream.synchronize()

        ch_codeword = self.power_normalize(output)
        zero_pad = np.zeros((self.n_padding, 2), dtype=self.target_dtype)
        ch_input = np.concatenate((ch_codeword, zero_pad), axis=0).astype(self.target_dtype)

        self.cfx.pop()
        return ch_input

    def ssf_estimate(self, frame, ref_frame):
        threading.Thread.__init__(self)
        self.cfx.push()

        output = np.empty(frame.shape, dtype=self.target_dtype)
        input = np.concatenate((frame, ref_frame), axis=1)
        input_allocations, input_bindings = self._allocate_memory((input))
        output_allocations, output_bindings = self._allocate_memory((output))

        d_input = input_allocations[0]
        d_output = output_allocations[0]
        bindings = input_bindings + output_bindings

        cuda.memcpy_htod_async(d_input, input, self.stream)
        self.ssf_net.execute_async_v2(bindings, self.stream.handle, None)
        cuda.memcpy_dtoh_async(output, d_output, self.stream)

        self.stream.synchronize()

        self.cfx.pop()
        return output

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
        # encoded_symbols = output_items[1]

        payload_idx = 0
        # encoded_symbol_idx = 0

        while payload_idx < len(payload_out):
            if self.pair_idx % self.ch_uses == 0:
                self.frame_idx = (self.frame_idx + 1) % self.n_frames
                self.pair_idx = 0
                self.packet_idx = 0

                # start_time = time.time()
                self.curr_codeword = self.key_frame_encode(self.video_frames[self.frame_idx],
                                                           self.snr)
                # end_time = time.time()
                # print('encode time: {}'.format(end_time - start_time))

            if self.pair_idx % self.packet_len == 0:
                # print('packet_len {}'.format(self.packet_len))
                # print('frame_idx {}'.format(self.frame_idx))
                # print('packet_idx {}'.format(self.packet_idx))

                self.add_item_tag(0, payload_idx + self.nitems_written(0), pmt.intern('packet_len'), pmt.from_long(144))

                frame_idx_bits = '{0:07b}'.format(self.frame_idx)
                frame_idx_bits = [np.float(b) for b in frame_idx_bits]
                frame_idx_bits = frame_idx_bits[::-1]

                # if self.packet_idx > 64:
                #     ipdb.set_trace()
                packet_idx_bits = '{0:09b}'.format(self.packet_idx)
                packet_idx_bits = [np.float(b) for b in packet_idx_bits]
                packet_idx_bits = packet_idx_bits[::-1]

                # print('Tx frame_idx: {}, packet_idx: {}'.format(self.frame_idx, self.packet_idx))

                header_bits = (frame_idx_bits + packet_idx_bits) * 3
                # TODO can use better FEC methods

                for bit in header_bits:
                    # TODO add method for different header modulations
                    payload_out[payload_idx] = (2 * bit - 1) + 0*1j
                    payload_idx += 1
                    if payload_idx >= len(payload_out):
                        break

                if payload_idx >= len(payload_out):
                    break

                self.packet_idx += 1

            payload_out[payload_idx] = (self.curr_codeword[self.pair_idx, 0]
                                        + self.curr_codeword[self.pair_idx, 1]*1j)
            # encoded_symbols[encoded_symbol_idx] = (self.curr_codeword[self.pair_idx, 0]
            #                                        + self.curr_codeword[self.pair_idx, 1]*1j)

            self.pair_idx += 1
            payload_idx += 1
            # encoded_symbol_idx += 1

        return len(output_items[0])
