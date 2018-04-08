:github_url: https://github.com/rusty1s/pytorch_scatter

PyTorch Scatter documentation
===============================

This package consists of a small extension library of highly optimized sparse update (scatter) operations for the use in `PyTorch <http://pytorch.org/>`_, which are missing in the main package.
Scatter operations can be roughly described as reduce operations based on a given "group-index" tensor.

All included operations work on varying data types, are implemented both for CPU and GPU and include a backwards implementation.

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Package reference

   functions/*

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
