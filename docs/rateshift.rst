RateShift Module
================


.. raw:: html

   <script src='https://storage.ko-fi.com/cdn/scripts/overlay-widget.js'></script>
   <script>
     kofiWidgetOverlay.draw('mrzahaki', {
       'type': 'floating-chat',
       'floating-chat.donateButton.text': 'Support me',
       'floating-chat.donateButton.background-color': '#2980b9',
       'floating-chat.donateButton.text-color': '#fff'
     });
   </script>



The `sudio.rateshift` module provides Python bindings for high-quality audio resampling using various interpolation techniques. This module allows users to resample audio data, either through simple function calls or using classes that handle more complex use cases.

Module Contents
---------------

Enums
-----

>>> sudio.rateshift.ConverterType

An enum representing different types of audio resampling converters.

- ``sinc_best``: Highest quality sinc-based resampler.
- ``sinc_medium``: Medium quality sinc-based resampler.
- ``sinc_fastest``: Fastest sinc-based resampler.
- ``zero_order_hold``: Zero-order hold (nearest-neighbor interpolation).
- ``linear``: Linear interpolation resampler.

Classes
-------

>>> sudio.rateshift.Resampler

A class that provides functionality for resampling audio data.

**Methods**:
- ``__init__(converter_type, channels=1)``: Initializes the `Resampler` object.

  **Arguments**:
    - ``converter_type`` (ConverterType): Type of resampler to use.
    - ``channels`` (int, optional): Number of channels. Defaults to 1.

- ``process(input, sr_ratio, end_of_input=False)``: Processes audio data through the resampler.

  **Arguments**:
    - ``input`` (numpy.ndarray): Input audio data as a NumPy array.
    - ``sr_ratio`` (float): Ratio of output sample rate to input sample rate.
    - ``end_of_input`` (bool, optional): Whether this is the final batch of input data. Defaults to False.

  **Returns**:
    - numpy.ndarray: Resampled audio data.

- ``set_ratio(ratio)``: Sets a new resampling ratio.

  **Arguments**:
    - ``ratio`` (float): New resampling ratio.

- ``reset()``: Resets the internal state of the resampler.

>>> sudio.rateshift.CallbackResampler

A class that resamples audio data using a callback function to provide input data.

**Methods**:
- ``__init__(callback, ratio, converter_type, channels)``: Initializes the `CallbackResampler` object.

  **Arguments**:
    - ``callback`` (callable): Function that supplies input audio data.
    - ``ratio`` (float): Initial resampling ratio.
    - ``converter_type`` (ConverterType): Type of resampler to use.
    - ``channels`` (int): Number of channels in the audio.

- ``read(frames)``: Reads and resamples the specified number of audio frames.

  **Arguments**:
    - ``frames`` (int): Number of frames to read.

  **Returns**:
    - numpy.ndarray: Resampled audio data.

- ``set_starting_ratio(ratio)``: Sets a new starting ratio for resampling.

  **Arguments**:
    - ``ratio`` (float): New starting ratio for resampling.

- ``reset()``: Resets the internal state of the resampler.

Functions
---------

>>> sudio.rateshift.resample(input, sr_ratio, converter_type, channels=1)

Resamples audio data.

This function provides a simplified interface for resampling audio data with specified parameters.

**Arguments**:
- ``input`` (numpy.ndarray): Input audio data as a NumPy array.
- ``sr_ratio`` (float): Ratio of output sample rate to input sample rate.
- ``converter_type`` (ConverterType): Type of resampling converter to use.
- ``channels`` (int, optional): Number of channels in the audio. Defaults to 1.

**Returns**:
- numpy.ndarray: Resampled audio data.

