import datetime
import sphinx_rtd_theme
import doctest

from torch_scatter import __version__  # noqa

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
]

source_suffix = '.rst'
master_doc = 'index'

author = 'Matthias Fey'
project = 'pytorch_scatter'
copyright = '{}, {}'.format(datetime.datetime.now().year, author)

version = 'master ({})'.format(__version__)
release = 'master'

html_theme = 'sphinx_rtd_theme'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

doctest_default_flags = doctest.NORMALIZE_WHITESPACE
intersphinx_mapping = {'python': ('https://docs.python.org/', None)}
