import pytest
import sudio.rateshift as rateshift
import numpy as np
from sudio.testing import psnr


CONVERTER_TYPES = {
    rateshift.ConverterType.sinc_best: 70,
    rateshift.ConverterType.sinc_medium: 65,
    rateshift.ConverterType.sinc_fastest: 60,
    rateshift.ConverterType.zero_order_hold: 16,
    rateshift.ConverterType.linear:15 
}

@pytest.fixture(scope="module", params=[1, 2])
def data(request, nsamples=1000, min=0, max=10):
    num_channels = request.param
    periods = np.linspace(min, max, nsamples)
    input_data = [
        np.sin(2 * np.pi * periods) for _ in range(num_channels)
    ]
    input_data = np.array(input_data)
    return (
        (num_channels, input_data[0])
        if num_channels == 1
        else (num_channels, input_data)
    )

@pytest.fixture(params=CONVERTER_TYPES.keys())
def converter_type(request):
    return request.param




def test_simple(data, converter_type, ratio=2.0):
    _, input_data = data
    out_data = rateshift.resample(input_data, ratio, converter_type)
    out_data = rateshift.resample(out_data, 1.0 / ratio, converter_type)
    assert psnr(input_data, out_data, mean_mlc=True) > CONVERTER_TYPES[converter_type], (
        f"PSNR: {psnr(input_data, out_data, mean_mlc=True)}"
    )

def test_process(data, converter_type, ratio=2.0):
    num_channels, input_data = data
    
    # phase 1
    src = rateshift.Resampler(converter_type, num_channels)
    tinput_data = input_data.T
    shape = tinput_data.shape
    out_data = src.process(tinput_data.flatten(), ratio, end_of_input=True)
    
    # phase 2
    src = rateshift.Resampler(converter_type, num_channels)
    out_data = src.process(out_data, 1 / ratio, end_of_input=True)
    out_data = out_data.reshape(shape)
    out_data = out_data.T

    assert psnr(input_data, out_data, mean_mlc=True) > CONVERTER_TYPES[converter_type], (
        f"PSNR: {psnr(input_data, out_data, mean_mlc=True)}"
    )

def test_match(data, converter_type, ratio=2.0):
    num_channels, input_data = data
    output_simple = rateshift.resample(input_data, ratio, converter_type)

    resampler = rateshift.Resampler(converter_type, channels=num_channels)
    tinput_data = input_data.T
    output_full = resampler.process(tinput_data.flatten(), ratio, end_of_input=True)
    if num_channels > 1:
        output_full = output_full.reshape((-1, num_channels))
    output_full = output_full.T
    
    assert np.allclose(output_simple, output_full)

def test_callback(data, converter_type, ratio=2.0):
    num_channels, input_data = data
    data_sent = False

    def producer(dd):
        nonlocal data_sent
        if not data_sent:
            data_sent = True
            return dd
        return []  

    input_data = input_data.T.flatten()
    input_callback = lambda: producer(input_data)
    resampler = rateshift.CallbackResampler(input_callback, ratio, converter_type, num_channels)
    frames = int(ratio * input_data.size)
    result = resampler.read(frames)
    assert np.abs(result.size - frames) < 10


def test_Resampler_set_ratio():
    resampler = rateshift.Resampler(rateshift.ConverterType.sinc_best, 1)
    resampler.set_ratio(1.5)

def test_Resampler_reset():
    resampler = rateshift.Resampler(rateshift.ConverterType.sinc_best, 1)
    resampler.reset()

def test_CallbackResampler_set_starting_ratio():
    def callback():
        return np.zeros(1000, dtype=np.float32)
    resampler = rateshift.CallbackResampler(callback, 1.0, rateshift.ConverterType.sinc_best, 1)
    resampler.set_starting_ratio(1.5)

def test_CallbackResampler_reset():
    def callback():
        return np.zeros(1000, dtype=np.float32)
    resampler = rateshift.CallbackResampler(callback, 1.0, rateshift.ConverterType.sinc_best, 1)
    resampler.reset()

