import numpy as np
import pytest
import rateshift

@pytest.fixture(scope="module", params=[1, 2])
def data(request):
    num_channels = request.param
    periods = np.linspace(0, 10, 1000)
    input_data = [
        np.sin(2 * np.pi * periods + i * np.pi / 2) for i in range(num_channels)
    ]
    return (
        (num_channels, input_data[0])
        if num_channels == 1
        else (num_channels, np.transpose(input_data))
    )

@pytest.fixture(params=[
    rateshift.ConverterType.sinc_best,
    rateshift.ConverterType.sinc_medium,
    rateshift.ConverterType.sinc_fastest,
    rateshift.ConverterType.zero_order_hold,
    rateshift.ConverterType.linear
])

def converter_type(request):
    return request.param

def test_simple(data, converter_type, ratio=2.0):
    _, input_data = data
    rateshift.resample(input_data, ratio, converter_type, input_data.shape[-1] if len(input_data.shape) > 1 else 1)

def test_process(data, converter_type, ratio=2.0):
    num_channels, input_data = data
    src = rateshift.Resampler(converter_type, num_channels)
    src.process(input_data, ratio, end_of_input=True)

def test_match(data, converter_type, ratio=2.0):
    num_channels, input_data = data
    output_simple = rateshift.resample(input_data, ratio, converter_type, num_channels)
    resampler = rateshift.Resampler(converter_type, channels=num_channels)
    output_full = resampler.process(input_data, ratio, end_of_input=True)
    assert np.allclose(output_simple, output_full)

def test_callback(data, converter_type, ratio=2.0):
    num_channels, input_data = data
    data_sent = False
    
    def producer():
        nonlocal data_sent
        if not data_sent:
            data_sent = True
            # Ensure input_data is a 2D array
            if input_data.ndim == 1:
                input_data_2d = input_data.reshape(-1, 1)
            else:
                input_data_2d = input_data
            # Convert the NumPy array to a list of floats
            return input_data_2d.ravel().tolist()
        return []  # Return an empty list instead of an empty NumPy array

    callback = lambda: producer()
    
    resampler = rateshift.CallbackResampler(callback, ratio, converter_type, num_channels)
    result = resampler.read(int(ratio * input_data.shape[0]))
    
    assert result.size > 0, "Expected non-empty result from CallbackResampler"

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

def test_resample_ndarray():
    input_data = np.random.randn(1000, 2).astype(np.float32)
    output = rateshift.resample(input_data, 2.0, rateshift.ConverterType.sinc_best, 2)
    assert isinstance(output, np.ndarray)
    assert output.shape[0] == int(input_data.shape[0] * 4)


def generate_sine_wave(frequency, duration, sample_rate):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return np.sin(2 * np.pi * frequency * t).astype(np.float32)

@pytest.mark.parametrize("original_sr, target_sr", [
    (44100, 22050),  # downsampling
    (22050, 44100),  # upsampling
])
@pytest.mark.parametrize("converter_type", [
    rateshift.ConverterType.sinc_best,
    rateshift.ConverterType.sinc_medium,
    rateshift.ConverterType.sinc_fastest,
    rateshift.ConverterType.zero_order_hold,
    rateshift.ConverterType.linear,
])
def test_resampling_quality(original_sr, target_sr, converter_type):
    # Generate a 1-second sine wave at 440 Hz
    duration = 1.0
    frequency = 440.0
    original_signal = generate_sine_wave(frequency, duration, original_sr)

    # Resample the signal
    ratio = target_sr / original_sr
    resampled_signal = rateshift.resample(original_signal, ratio, converter_type, 1)

    # Generate the ideal signal at the target sample rate
    ideal_signal = generate_sine_wave(frequency, duration, target_sr)

    # Trim signals to the same length if necessary
    min_length = min(len(resampled_signal), len(ideal_signal))
    resampled_signal = resampled_signal[:min_length]
    ideal_signal = ideal_signal[:min_length]

    # Calculate the root mean square error
    rmse = np.sqrt(np.mean((resampled_signal - ideal_signal) ** 2))

    # Define error thresholds for each converter type
    error_thresholds = {
        rateshift.ConverterType.sinc_best: 1e-4,
        rateshift.ConverterType.sinc_medium: 1e-3,
        rateshift.ConverterType.sinc_fastest: 1e-2,
        rateshift.ConverterType.zero_order_hold: 1.3e-1,
        rateshift.ConverterType.linear: 1e-1,
    }

    # Assert that the error is below the threshold for the given converter type
    assert rmse < error_thresholds[converter_type], f"RMSE {rmse} exceeds threshold {error_thresholds[converter_type]} for {converter_type}"

    # Calculate and print the signal-to-noise ratio (SNR) in dB
    signal_power = np.mean(ideal_signal ** 2)
    noise_power = np.mean((resampled_signal - ideal_signal) ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    print(f"SNR for {converter_type} ({original_sr} -> {target_sr} Hz): {snr:.2f} dB")


if __name__ == "__main__":
    pytest.main([__file__])