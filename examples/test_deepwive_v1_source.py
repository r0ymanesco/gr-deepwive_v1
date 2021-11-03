#!/usr/bin/env python3

import deepwive_v1

source_block = deepwive_v1.deepwive_v1_source('test_vid/v_CleanAndJerk_g24_c02.avi',
                                              'test_model/key_encoder.trt',
                                              20.)
source_block.test_work()
