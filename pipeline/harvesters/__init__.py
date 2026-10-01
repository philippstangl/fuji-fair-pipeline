"""Repository harvesters, dispatched by `type`."""
from __future__ import annotations

import logging

from pipeline.harvesters.base import MANIFEST_COLUMNS, BaseHarvester
from pipeline.harvesters.chemotion import ChemotionHarvester
from pipeline.harvesters.dataverse import DataverseHarvester
from pipeline.harvesters.zenodo import ZenodoHarvester

_HARVESTERS: dict[str, type[BaseHarvester]] = {
    "chemotion": ChemotionHarvester,
    "dataverse": DataverseHarvester,
    "zenodo": ZenodoHarvester,
}


def get_harvester(repo: dict, log: logging.Logger) -> BaseHarvester:
    """Instantiate the harvester for `repo['type']`."""
    rtype = repo.get("type")
    cls = _HARVESTERS.get(rtype)
    if cls is None:
        raise ValueError(
            f"unknown repository type {rtype!r}; known types: {sorted(_HARVESTERS)}"
        )
    return cls(repo, log)


__all__ = ["get_harvester", "BaseHarvester", "MANIFEST_COLUMNS"]
