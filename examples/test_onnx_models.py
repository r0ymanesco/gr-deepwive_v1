import onnxruntime as ort
import numpy as np
import cv2
import ipdb
import matplotlib.pyplot as plt


def get_video(source_fn):
    vid = cv2.VideoCapture(source_fn)
    flag, frame = vid.read()
    frame = np.swapaxes(frame, 0, 1)
    frame = np.swapaxes(frame, 0, 2)
    frame = np.expand_dims(frame, axis=0) / 255.
    frame = np.ascontiguousarray(frame, dtype=np.float32)
    return frame


def to_np_img(img):
    img = np.squeeze(img, axis=0)
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 0, 1)
    img = img[:, :, [2, 1, 0]]
    return img

encoder_session = ort.InferenceSession('onnx_output/key_encoder_simp.onnx')
decoder_session = ort.InferenceSession('onnx_output/key_decoder_simp.onnx')

frame = get_video('test_vid/v_CleanAndJerk_g24_c02.avi')
snr = np.array([[20.]], dtype=np.float32)

codeword = encoder_session.run(None, {'input': frame,
                                      'snr': snr})[0]
codeword_shape = codeword.shape
codeword = codeword.reshape(1, -1)
ch_uses = codeword.shape[1]
ch_input = (codeword / np.linalg.norm(codeword, ord=2, axis=1, keepdims=True)) * np.sqrt(ch_uses)
noise_stddev = np.sqrt(10**(-snr/10))
awgn = np.random.randn(*ch_input.shape) * noise_stddev
ch_output = ch_input + awgn.astype(np.float32)
decoder_input = ch_output.reshape(codeword_shape)
prediction = decoder_session.run(None, {'codeword': decoder_input,
                                        'snr': snr})[0]

ipdb.set_trace()
# plt.imshow(to_np_img(frame))
plt.imshow(to_np_img(prediction))
