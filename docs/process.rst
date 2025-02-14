Process Module
==============


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



.. automodule:: sudio.process.audio_wrap
   :members:
   :undoc-members:
   :show-inheritance:


.. automodule:: sudio.process.fx.tempo
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: sudio.process.fx.gain
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: sudio.process.fx.fx
   :members:
   :undoc-members:
   :show-inheritance:


.. automodule:: sudio.process.fx.channel_mixer
   :members:
   :undoc-members:
   :show-inheritance:


.. automodule:: sudio.process.fx.pitch_shifter
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: sudio.process.fx.normalize
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: sudio.process.fx.fade_envelope
   :members:
   :undoc-members:
   :show-inheritance:

The Fade Envelope module offers a rich set of predefined envelopes to shape audio dynamics. Each preset provides a unique way to modify the amplitude characteristics of an audio signal.

Preset Catalog
--------------

.. image:: /_static/fade_envelope_presets.png
   :alt: Fade Envelope Presets Visualization
   :align: center

Available Presets
^^^^^^^^^^^^^^^^^

1. **Smooth Ends**

2. **Bell Curve**

3. **Keep Attack Only**

4. **Linear Fade In**

5. **Linear Fade Out**

6. **Pulse**

7. **Remove Attack**

8. **Smooth Attack**

9. **Smooth Fade In**

10. **Smooth Fade Out**

11. **Smooth Release**

12. **Tremors**

13. **Zigzag Cut**

Usage Example
-------------

.. code-block:: python

    from sudio.process.fx import FadeEnvelope, FadePreset

    # Apply a smooth fade in to an audio signal
    fx = FadeEnvelope()
    processed_audio = fx.process(
        audio_data, 
        preset=FadePreset.SMOOTH_FADE_IN
    )


Making Custom Presets
---------------------

Custom presets in Sudio let you shape your audio's dynamics in unique ways. Pass a numpy array to the FadeEnvelope effect, 
and Sudio's processing transforms it into a smooth, musically coherent envelope using interpolation, and gaussian filter. 
You can create precise sound manipulations, control the wet/dry mix, and adjust the output gain. in this mode sawtooth_freq, fade_release, and fade_attack parameters are unavailable.

.. code-block:: python

   from sudio.process.fx import FadeEnvelope

   s = song[10:30]
   custom_preset = np.array([0.0, 0.0, 0.1, 0.2, 0.3, 0.7, 0.1, 0.0])
   s.afx(
      FadeEnvelope, 
      preset=custom_preset, 
      enable_spline=True,
      start=10.5, 
      stop=25, 
      output_gain_db=-5, 
      wet_mix=.9
      )
   su.echo(s)

.. image:: /_static/fade_envelope_custom_preset.png
   :alt: Fade Envelope Custom Preset Visualization
   :align: center
