:github_url: https://github.com/rusty1s/pytorch_scatter

PyTorch Scatter Documentation
=============================

This package consists of a small extension library of highly optimized sparse update (scatter and segment) operations for the use in `PyTorch <http://pytorch.org/>`_, which are missing in the main package.
Scatter and segment operations can be roughly described as reduce operations based on a given "group-index" tensor.
Segment operations require the "group-index" tensor to be sorted, whereas scatter operations are not subject to these requirements.

All included operations are broadcastable, work on varying data types, and are implemented both for CPU and GPU with corresponding backward implementations.

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Package reference

   functions/*
   composite/*

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
