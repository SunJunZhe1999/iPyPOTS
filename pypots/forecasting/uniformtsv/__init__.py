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
and is exposed here under the UniFormTSV alias to keep the upstream code untouched.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause


from .model import UniFormTSV

__all__ = [
    "UniFormTSV",
]
