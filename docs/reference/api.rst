.. meta::
  :description: rocFFT documentation and API reference library
  :keywords: rocFFT, ROCm, API, documentation

.. _api-usage:

********************************************************************
API Usage
********************************************************************

This section describes usage of the rocFFT library API.

Types
=====

There are a few data structures that are internal to the library. The pointer types to these
structures are given below. The user would need to use these types to create handles and pass them
between different library functions.

.. doxygentypedef:: rocfft_plan

.. doxygentypedef:: rocfft_plan_description

.. doxygentypedef:: rocfft_execution_info

Library Setup and Cleanup
=========================

The following functions deal with initialization and cleanup of the library.

.. doxygenfunction:: rocfft_setup

.. doxygenfunction:: rocfft_cleanup

Plan
====

The following functions are used to create and destroy plan objects.

.. doxygenfunction:: rocfft_plan_create

.. doxygenfunction:: rocfft_plan_destroy

The following functions are used to query for information after a plan is created.

.. doxygenfunction:: rocfft_plan_get_work_buffer_size

.. doxygenfunction:: rocfft_plan_get_print

Plan description
================

Most of the time, :cpp:func:`rocfft_plan_create` is able to fully
specify a transform.  Advanced plan details such as strides and
offsets require creation of a plan description object, which is
configured and passed to the :cpp:func:`rocfft_plan_create` function.

The plan description object can be safely destroyed after it is given
to the :cpp:func:`rocfft_plan_create` function.

.. doxygenfunction:: rocfft_plan_description_create

.. doxygenfunction:: rocfft_plan_description_destroy

.. doxygenfunction:: rocfft_plan_description_set_scale_factor

.. doxygenfunction:: rocfft_plan_description_set_data_layout

Execution
=========

After a plan has been created, it can be executed using the
:cpp:func:`rocfft_execute` function,
to compute a transform on specified data. Aspects of the execution can be controlled and any useful
information returned to the user.

.. doxygenfunction:: rocfft_execute

Execution info
-=============

:cpp:func:`rocfft_execute` takes an optional :cpp:type:`rocfft_execution_info` parameter. This parameter encapsulates
information such as the work buffer and compute stream for the transform.

.. doxygenfunction:: rocfft_execution_info_create

.. doxygenfunction:: rocfft_execution_info_destroy

.. doxygenfunction:: rocfft_execution_info_set_work_buffer

.. comment doxygenfunction:: rocfft_execution_info_set_mode

.. doxygenfunction:: rocfft_execution_info_set_stream

.. comment doxygenfunction:: rocfft_execution_info_get_events

HIP graph Support for rocFFT
============================

rocFFT supports capturing kernels launched by
:cpp:func:`rocfft_execute` into HIP graph nodes.  This way, users can
capture FFT execution, along with other work, into a HIP graph and
launch the work in the graph multiple times.

Graph capture is only supported for single-process transforms.
Multi-process transforms such as those that use Message Passing
Interface cannot use graph capture, as rocFFT performs inter-process
communication in addition to launching kernels.

Note that each launch of a HIP graph will provide the same arguments
to the kernels in the graph.  In particular, this implies that all of
the parameters to :cpp:func:`rocfft_execute` remain valid while the
HIP graph is in use:

* The rocFFT plan

* The input and output buffers

* The :cpp:type:`rocfft_execution_info` object, if one was provided

rocFFT does not support capturing work performed by other API
functions aside from :cpp:func:`rocfft_execute` into HIP graphs.

Enumerations
============

This section provides all the enumerations used.

.. doxygenenum:: rocfft_status

.. doxygenenum:: rocfft_transform_type

.. doxygenenum:: rocfft_precision

.. doxygenenum:: rocfft_result_placement

.. doxygenenum:: rocfft_array_type

.. comment doxygenenum:: rocfft_execution_mode

