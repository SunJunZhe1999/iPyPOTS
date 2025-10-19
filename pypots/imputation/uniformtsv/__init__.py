"""
The package of the partially-observed time-series imputation model UniFormTSV.

Refer to the paper
`Wenjie Du, David Cote, and Yan Liu.
MOMENT: Self-Attention-based Imputation for Time Series.
Expert Systems with Applications, 219:119619, 2023.
<https://arxiv.org/pdf/2202.08516>`_

Notes
-----
This implementation mirrors the official MOMENT release https://github.com/WenjieDu/MOMENT
and is provided under the UniFormTSV alias so the upstream implementation stays untouched.

"""

# Created by Wenjie Du <wenjay.du@gmail.com>
# License: BSD-3-Clause

from .model import UniFormTSV

__all__ = [
    "UniFormTSV",
]
