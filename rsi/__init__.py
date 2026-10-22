"""RSI 实验内核。功能实现位于子包，旧模块导入保留兼容。"""

from ._compat import install_legacy_imports as _install_legacy_imports

__version__ = "0.7.0"

_install_legacy_imports()
del _install_legacy_imports
