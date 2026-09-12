import numpy as np


class Display:
    def show(self, fb: np.ndarray) -> None:
        raise NotImplementedError

    def should_quit(self) -> bool:
        return False

    def close(self) -> None:
        pass
