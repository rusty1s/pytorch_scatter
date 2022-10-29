import datetime
import doctest

import sphinx_rtd_theme
import torch_scatter

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

source_suffix = '.rst'
master_doc = 'index'

author = 'Matthias Fey'
project = 'pytorch_scatter'
copyright = '{}, {}'.format(datetime.datetime.now().year, author)

version = torch_scatter.__version__
release = torch_scatter.__version__

html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

doctest_default_flags = doctest.NORMALIZE_WHITESPACE
intersphinx_mapping = {'python': ('https://docs.python.org/', None)}
