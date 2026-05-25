"""Decoder registry: maps representation name → decoder class.

To add a new variant, add one import and one entry to DECODER_REGISTRY.
No other files need to change.
"""

from typing import Any

from models.decoders.tsdf_decoder import TSDFDecoder
# from models.decoders.occupancy_decoder import OccupancyDecoder  ← add here
# from models.decoders.triplane_decoder import TriplaneDecoder    ← add here

DECODER_REGISTRY: dict[str, type] = {
    "tsdf": TSDFDecoder,
    # "occupancy": OccupancyDecoder,
    # "triplane": TriplaneDecoder,
}


def get_decoder(name: str, **kwargs: Any) -> Any:
    """Instantiate a decoder by its registry name."""
    if name not in DECODER_REGISTRY:
        raise KeyError(
            f"Unknown decoder '{name}'. Available: {list(DECODER_REGISTRY.keys())}"
        )
    return DECODER_REGISTRY[name](**kwargs)
