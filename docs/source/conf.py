import os
import sys
import datetime
import sphinx_rtd_theme

sys.path.insert(0, os.path.abspath('../..'))

from torch_scatter import __version__

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
]

source_suffix = '.rst'
master_doc = 'index'

project = 'pytorch_scatter'
copyright = '{}, Matthias Fey'.format(datetime.datetime.now().year)
author = 'Matthias Fey'

version = __version__
release = __version__

html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
