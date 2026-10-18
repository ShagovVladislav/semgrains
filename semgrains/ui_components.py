import base64
from io import BytesIO
from pathlib import Path
import numpy as np
from PIL import Image
import streamlit.components.v1 as components

_editor = components.declare_component("sem_grain_editor", path=str(Path(__file__).parent / "editor"))


def editor(rgb, labels=None, key="editor", revision="0", polygons=None, ignored=None):
    buf = BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    image = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    encoded = base64.b64encode(np.asarray(labels, dtype="<u4").tobytes()).decode() if labels is not None else None
    return _editor(image=image, labels=encoded, width=rgb.shape[1], height=rgb.shape[0],
                   annotation=labels is None, revision=revision, polygons=polygons or [], ignored=ignored or [],
                   key=key, default=None)
