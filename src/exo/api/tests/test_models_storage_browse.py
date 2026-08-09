"""Wire-format tests for the Shared Model Storage browse response.

The dashboard reads this payload directly, and :class:`FrozenModel` serialises
through a camelCase alias generator — so the JSON keys are *not* the Python
field names. A silent rename here strands the folder picker (an unread
``parentPath`` simply made the Up button fall back to the shortcut list), which
is exactly the kind of break a type checker cannot see across the boundary.
"""

from typing import cast

from exo.api.types import (
    ModelsStorageBrowseEntry,
    ModelsStorageBrowseResponse,
    ModelsStorageNetworkVolume,
)


def _response() -> ModelsStorageBrowseResponse:
    return ModelsStorageBrowseResponse(
        path="/mnt",
        parent_path="/",
        entries=[ModelsStorageBrowseEntry(name="models", path="/mnt/models")],
        network_volumes=[
            ModelsStorageNetworkVolume(
                path="/mnt/models",
                source="10.0.10.44:/export/models",
                filesystem="nfs",
                reachable=True,
            )
        ],
    )


def _payload() -> dict[str, object]:
    return cast(dict[str, object], _response().model_dump(by_alias=True))


def _first_item(payload: dict[str, object], key: str) -> dict[str, object]:
    return cast(list[dict[str, object]], payload[key])[0]


class TestBrowseResponseWireFormat:
    def test_keys_the_dashboard_reads_are_camel_case(self) -> None:
        assert set(_payload()) == {
            "path",
            "parentPath",
            "entries",
            "error",
            "truncated",
            "networkVolumes",
        }

    def test_parent_path_survives_serialisation(self) -> None:
        assert _payload()["parentPath"] == "/"

    def test_network_volume_fields_are_camel_case(self) -> None:
        volume = _first_item(_payload(), "networkVolumes")
        assert set(volume) == {"path", "source", "filesystem", "reachable"}
        assert volume["source"] == "10.0.10.44:/export/models"

    def test_entry_fields_are_stable(self) -> None:
        assert set(_first_item(_payload(), "entries")) == {"name", "path", "hidden"}
