"""源码与仓库位置。移动功能模块时，不再按各文件的层级推断根目录。"""

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
