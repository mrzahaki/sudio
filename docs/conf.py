import os
import sys
from sphinx.ext.autodoc import importer

sys.path.insert(0, os.path.abspath('../sudio'))

project = 'sudio'
copyright = '2024, mrzahaki'
author = 'Hossein Zahak'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints',
    'breathe',
    'exhale',
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
        FILTER_PATTERNS        = *.cpp=doxyfilter.py
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

def custom_import(self, modname):
    try:
        return importer._builtin_import(modname)
    except ImportError:
        return None

importer._builtin_import = custom_import
