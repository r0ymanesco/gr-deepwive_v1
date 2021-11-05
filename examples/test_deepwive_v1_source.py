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

source_block = deepwive_v1.deepwive_v1_source('test_vid/v_CleanAndJerk_g24_c02.avi',
                                              ['test_model/key_encoder_fp16_graph.trt',
                                               'test_model/key_decoder_fp16_graph.trt'],
                                              snr=20.,
                                              use_fp16=True,
                                              test_mode=True)
input_frame, output_frame, mse = source_block.test_work()

ipdb.set_trace()
plt.imshow(to_np_img(output_frame))
plt.imshow(to_np_img(input_frame))
