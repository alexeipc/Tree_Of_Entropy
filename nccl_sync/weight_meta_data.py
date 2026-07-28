from __future__ import annotations

from typing import Any, Iterator
import torch

class WeightMetadata:
    """
    Serializable description of tensors that will be sent to vLLM.
    """

    def __init__(
        self,
        names: list[str],
        shapes: list[list[int]],
        dtype_names: list[str],
    ) -> None:
        if not (len(names) == len(shapes) == len(dtype_names)):
            raise ValueError(
                "names, shapes, and dtype_names must have equal lengths."
            )

        self.names = names
        self.shapes = shapes
        self.dtype_names = dtype_names

    def to_dict(self) -> dict[str, Any]:
        return {
            "names": self.names,
            "shapes": self.shapes,
            "dtype_names": self.dtype_names,
        }

    @classmethod
    def from_named_parameters(
        cls,
        named_parameters: Iterator[tuple[str, torch.Tensor]],
    ) -> "WeightMetadata":
        names: list[str] = []
        shapes: list[list[int]] = []
        dtype_names: list[str] = []

        for name, parameter in named_parameters:
            names.append(name)
            shapes.append(list(parameter.shape))
            dtype_names.append(
                str(parameter.dtype).removeprefix("torch.")
            )

        return cls(
            names=names,
            shapes=shapes,
            dtype_names=dtype_names,
        )