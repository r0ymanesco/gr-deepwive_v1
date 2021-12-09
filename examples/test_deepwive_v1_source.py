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


source_block = deepwive_v1.deepwive_v1_source('test_files/v_YoYo_g25_c05.avi',
                                              ['test_files/key_encoder.trt',
                                               'test_files/key_decoder.trt'],
                                              model_cout=240,
                                              snr=20.,
                                              use_fp16=False)
input_frame, output_frame, mse = source_block.test_work()

ipdb.set_trace()
plt.imshow(to_np_img(output_frame))
plt.imshow(to_np_img(input_frame))
