################
Installation
################

.. contents:: Table of Contents


*********************
Package Requirements
*********************

.. literalinclude:: ../../../requirements/requirements.in

****************
CI Requirements
****************

.. literalinclude:: ../../../requirements/requirements_ci.in

*************
Installation
*************

From pypi server

.. code-block:: bash

    pip install cgnal-core

From source

.. code-block:: bash

    git clone https://github.com/CGnal/cgnal-core
    cd "$(basename "https://github.com/CGnal/cgnal-core" .git)"
    make install
