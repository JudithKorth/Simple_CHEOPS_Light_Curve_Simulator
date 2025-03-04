import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import clcsim

project = 'Simple CHEOPS light curve simulator'
copyright = '2025, Judith Korth and Hannu Parviainen'
author = 'Judith Korth and Hannu Parviainen'

version = ".".join(clcsim.__version__.split('.')[:2])
release = clcsim.__version__

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'sphinx.ext.imgmath',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosectionlabel',
    'sphinx_copybutton',
    'nbsphinx',
    'numpydoc'
]

autosummary_generate = True

templates_path = ['_templates']
exclude_patterns = ["**/.ipynb_checkpoints"]
numpydoc_show_class_members = False

pygments_style = 'sphinx'

html_theme = 'furo'
html_title = f'Simple CHEOPS Light Curve Simulator v{version}'
html_theme_options = {
    'sidebar_hide_name': False,
}
html_static_path = ['_static']

default_role = 'py:obj'

intersphinx_mapping = {'python': ('https://docs.python.org/3/', None),
                       'numpy': ('https://docs.scipy.org/doc/numpy/', None),
                       'scipy': ('https://docs.scipy.org/doc/scipy/reference', None),
                       'matplotlib': ('https://matplotlib.org', None),
                       'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
                       'astropy': ('https://docs.astropy.org/en/stable/', None),
                       'uncertainties': ('https://uncertainties.readthedocs.io/en/latest/', None)}
