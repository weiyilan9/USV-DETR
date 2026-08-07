# -*- coding: utf-8 -*-
"""Run USV-DETR detection from a cloned repository.

    python detect.py recording.wav --config configs/USV-DETR.yml \
        --checkpoint USV-DETR.pth

After pip install the same interface is available as the usvdetr-detect
command. Run with --help to see every option.
"""

import sys

from usvdetr.cli import main

if __name__ == "__main__":
    sys.exit(main())
