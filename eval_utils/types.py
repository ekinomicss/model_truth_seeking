import numpy as np
from typing import Any, Literal, TypedDict
from numpy.typing import NDArray

NDFloatArray = NDArray[np.floating[Any]]
NDIntArray = NDArray[np.integer[Any]]

class ChatMessage(TypedDict):
    role: Literal["user", "assistant", "system"]
    content: str