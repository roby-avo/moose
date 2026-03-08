from __future__ import annotations

from moose.legal import normalize_legal_ref


def test_normalize_legal_ref_supports_adaptcentre_gdprtext_iris() -> None:
    assert (
        normalize_legal_ref("gdprtext", "http://purl.org/adaptcentre/resources/GDPRtEXT#article5")
        == "gdprtext:article5"
    )
    assert (
        normalize_legal_ref("gdprtext", "http://purl.org/adaptcentre/ontologies/GDPRtEXT#Article")
        == "gdprtext:Article"
    )
