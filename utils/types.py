from typing import Any, Literal, TypedDict
import numpy as np
from numpy.typing import NDArray

NDFloatArray = NDArray[np.floating[Any]]
NDIntArray = NDArray[np.integer[Any]]


class ChatMessage(TypedDict):
    role: Literal["user", "assistant", "system"]
    content: str