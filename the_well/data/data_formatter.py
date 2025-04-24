from abc import ABC, abstractmethod
from typing import Dict, Tuple

import torch
from einops import rearrange

from .datasets import WellMetadata


class AbstractDataFormatter(ABC):
    def __init__(self, metadata: WellMetadata):
        self.metadata = metadata

    @abstractmethod
    def process_input(self, data: Dict) -> Tuple:
        raise NotImplementedError

    @abstractmethod
    def process_output_channel_last(self, output: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def process_output_expand_time(self, output: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    

class SineNetFormatter(AbstractDataFormatter):
    """
    Formatter voor het SineNet-model.

    Houdt tijd als aparte dimensie en gebruikt channels-first formaat:
    (B, T, H, W, C) → (B, T, C, H, W)
    """

    def process_input(self, data: Dict) -> Tuple:
        x = data["input_fields"]  # (B, Ti, H, W, C)
        x = rearrange(x, "b t h w c -> b t c h w")  # (B, Ti, C, H, W)

        y = data["output_fields"]  # (B, To, H, W, C)
        y = rearrange(y, "b t h w c -> b t c h w")  # (B, To, C, H, W)

        return (torch.nan_to_num(x),), torch.nan_to_num(y)

    def process_output_channel_last(self, output: torch.Tensor) -> torch.Tensor:
        return rearrange(output, "b t c h w -> b t h w c")

    def process_output_after_denomalize(self, output: torch.Tensor) -> torch.Tensor:
        return rearrange(output, "b t h w c -> b t c h w") # terug naar original format

    def process_output_denormalize(self, output: torch.Tensor) -> torch.Tensor:
        return rearrange(output, "b t c h w -> b t h w c")

    def process_output_expand_time(self, output: torch.Tensor) -> torch.Tensor:
        return output  # tijdsdimensie is al aanwezig



class DefaultChannelsFirstFormatter(AbstractDataFormatter):
    """
    Default preprocessor for data in channels first format.

    Stacks time as individual channel.
    """

    def process_input(self, data: Dict) -> Tuple:
        x = data["input_fields"]
        x = rearrange(x, "b t ... c -> b (t c) ...")
        if "constant_fields" in data:
            flat_constants = rearrange(data["constant_fields"], "b ... c -> b c ...")
            x = torch.cat(
                [
                    x,
                    flat_constants,
                ],
                dim=1,
            )
        y = data["output_fields"]
        # TODO - Add warning to output if nan has to be replaced
        # in some cases (staircase), its ok. In others, it's not.
        return (torch.nan_to_num(x),), torch.nan_to_num(y)

    def process_output_channel_last(self, output: torch.Tensor) -> torch.Tensor:
        return rearrange(output, "b c ... -> b ... c")

    def process_output_expand_time(self, output: torch.Tensor) -> torch.Tensor:
        return rearrange(output, "b ... c -> b 1 ... c")


class DefaultChannelsLastFormatter(AbstractDataFormatter):
    """
    Default preprocessor for data in channels last format.

    Stacks time as individual channel.
    """

    def process_input(self, data: Dict) -> Tuple:
        x = data["input_fields"]
        x = rearrange(x, "b t ... c -> b ... (t c)")
        if "constant_fields" in data:
            flat_constants = data["constant_fields"]
            x = torch.cat(
                [
                    x,
                    flat_constants,
                ],
                dim=-1,
            )
        y = data["output_fields"]
        # TODO - Add warning to output if nan has to be replaced
        # in some cases (staircase), its ok. In others, it's not.
        return (torch.nan_to_num(x),), torch.nan_to_num(y)

    def process_output_channel_last(self, output: torch.Tensor) -> torch.Tensor:
        return output

    def process_output_expand_time(self, output: torch.Tensor) -> torch.Tensor:
        return rearrange(output, "b ... c -> b 1 ... c")
