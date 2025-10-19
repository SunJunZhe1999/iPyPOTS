"""
The package including the modules of UniFormTSV, cloned from MOMENT.

Refer to the paper
`Mononito Goswami, Konrad Szafer, Arjun Choudhry, Yifu Cai, Shuo Li, and Artur Dubrawski.
"MOMENT: A Family of Open Time-series Foundation Models".
In ICML, 2024.
<https://proceedings.mlr.press/v235/goswami24a.html>`_

Notes
-----
This implementation mirrors the official MOMENT release
https://github.com/moment-timeseries-foundation-model/moment-research
and is kept here under the UniFormTSV alias so the original code base
remains untouched.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from .backbone import BackboneUniFormTSV, SUPPORTED_HUGGINGFACE_MODELS

__all__ = [
    "BackboneUniFormTSV",
    "SUPPORTED_HUGGINGFACE_MODELS",
]
