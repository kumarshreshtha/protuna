{% set title = module.rsplit('.', 1)[-1] ~ '.' ~ objname if module.startswith('protuna.data.datasets.') else objname %}
{{ title | escape | underline }}

.. currentmodule:: {{ module }}

.. rst-class:: api-module

   {{ module }}

.. autofunction:: {{ objname }}
