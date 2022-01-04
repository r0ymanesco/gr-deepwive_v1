#!/usr/bin/env python3

import deepwive_v1
import numpy as np
import matplotlib.pyplot as plt
import ipdb


def to_np_img(img):
    img = np.squeeze(img, axis=0)
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 0, 1)
    img = img[:, :, [2, 1, 0]]
    return img.astype(np.float32)


# source_block = deepwive_v1.deepwive_v1_source('test_files/v_YoYo_g25_c05.avi',
#                                               key_encoder_fn='test_files/key_encoder.trt',
#                                               interp_encoder_fn='test_files/key_decoder.trt',
#                                               ssf_net_fn='test_files/ssf_net.trt',
#                                               bw_allocator_fn='test_files/bw_allocator.trt',
#                                               packet_len=96,
#                                               num_chunks=20,
#                                               gop_size=5,
#                                               model_cout=240,
#                                               snr=20.,
#                                               use_fp16=False)
# _ = source_block.test_work()

sink_block = deepwive_v1.deepwive_v1_sink('test_files/v_YoYo_g25_c05.avi',
                                          key_decoder_fn='test_files/key_decoder.trt',
                                          interp_decoder_fn='test_files/interp_decoder.trt',
                                          packet_len=96,
                                          num_chunks=20,
                                          gop_size=5,
                                          model_cout=240,
                                          snr=20.,
                                          use_fp16=False)

sink_block.test_work()

# self.deepwive_v1_deepwive_v1_source_0 = deepwive_v1.deepwive_v1_source(
#     '/home/tt2114/workspace/gr-deepwive_v1/examples/test_files/v_YoYo_g25_c05.avi',
#     key_encoder_fn='/home/tt2114/workspace/gr-deepwive_v1/examples/test_files/key_encoder.trt',
#     interp_encoder_fn='/home/tt2114/workspace/gr-deepwive_v1/examples/test_files/key_decoder.trt',
#     ssf_net_fn='/home/tt2114/workspace/gr-deepwive_v1/examples/test_files/ssf_net.trt',
#     bw_allocator_fn='/home/tt2114/workspace/gr-deepwive_v1/examples/test_files/bw_allocator.trt',
#     packet_len=self.packet_len,
#     num_chunks=20,
#     gop_size=5,
#     model_cout=240,
#     snr=20.,
#     use_fp16=False)
# ipdb.set_trace()
# plt.imshow(to_np_img(output_frame))
# plt.imshow(to_np_img(input_frame))
