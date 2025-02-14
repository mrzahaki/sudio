import os
import sys
from sphinx.ext.autodoc import importer

sys.path.insert(0, os.path.abspath('..'))

project = 'sudio'
copyright = '2024, mrzahaki'
author = 'Hossein Zahak'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints',
    'breathe',
    'exhale'
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
autodoc_member_order = 'bysource'
autoclass_content = 'both'

breathe_projects = {
    "sudio": "./doxyoutput/xml"
}
breathe_default_project = "sudio"

exhale_args = {
    "containmentFolder":     "./api",
    "rootFileName":          "io_root.rst",
    "doxygenStripFromPath":  '..',
    "rootFileTitle":         "API",
    "createTreeView":        True,
    "exhaleExecutesDoxygen": True,
    "exhaleDoxygenStdin":    """
        INPUT                  = ../sudio/io ../sudio/rateshift
        EXCLUDE_PATTERNS       = */miniaudio/* */dr_libs/* */portaudio/* */libsamplerate/* */*.py */libmp3lame-CMAKE/* */lame/* */flac/* */ogg/* */vorbis/*
        FILTER_PATTERNS        = *.cpp=python doxyfilter.py
        RECURSIVE              = YES 
        GENERATE_XML           = YES
        GENERATE_HTML          = NO
        XML_OUTPUT             = xml
        INCLUDE_PATH           = ../cache/pybind11
    """,
}
breathe_implementation_filename_extensions = ['.c', '.cc', '.cpp', '.hpp']
primary_domain = 'cpp'
highlight_language = 'cpp'
html_show_sourcelink = False
autodoc_mock_imports = [
    'sudio._suio', 
    'sudio._rateshift',
    'sudio.rateshift',
    'sudio.io', 
    'scipy', 
    # 'numpy',
    'sudio.process.fx._tempo',
    'sudio.process.fx._fade_envelope',
    'sudio.process.fx._channel_mixer',
    "sudio.process.fx._pitch_shifter",
    "sudio.utils.math",
    "sudio.generator.tone.sinewave",
    # "sudio.utils.functional",
    ]
html_context = {
    "google_analytics_id": "G-RLP20V08DB",
}
html_css_files = [
    'style.css',
]

