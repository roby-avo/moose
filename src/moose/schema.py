from __future__ import annotations

COARSE_TYPES = [
    "NER:PERSON",
    "NER:ORGANIZATION",
    "NER:LOCATION",
    "NER:EVENT",
    "NER:WORK_OF_ART",
    "NER:PRODUCT",
    "NER:DATE_TIME",
    "NER:NUMBER",
    "NER:MONEY",
    "NER:PERCENT",
    "NER:LAW_OR_REGULATION",
    "NER:OTHER",
]

FINE_TYPES = [
    "NER:PERSON.ENTREPRENEUR",
    "NER:PERSON.POLITICIAN",
    "NER:ORG.COMPANY",
    "NER:ORG.UNIVERSITY",
    "NER:ORG.RESEARCH_INSTITUTE",
    "NER:LOC.CITY",
    "NER:LOC.COUNTRY",
    "NER:PRODUCT.DEVICE",
    "NER:EVENT.CONFERENCE",
    "NER:OTHER",
]

FINE_TO_COARSE = {
    "NER:PERSON.ENTREPRENEUR": "NER:PERSON",
    "NER:PERSON.POLITICIAN": "NER:PERSON",
    "NER:ORG.COMPANY": "NER:ORGANIZATION",
    "NER:ORG.UNIVERSITY": "NER:ORGANIZATION",
    "NER:ORG.RESEARCH_INSTITUTE": "NER:ORGANIZATION",
    "NER:LOC.CITY": "NER:LOCATION",
    "NER:LOC.COUNTRY": "NER:LOCATION",
    "NER:PRODUCT.DEVICE": "NER:PRODUCT",
    "NER:EVENT.CONFERENCE": "NER:EVENT",
    "NER:OTHER": "NER:OTHER",
}


def get_types(schema: str) -> list[str]:
    if schema == "fine":
        return list(FINE_TYPES)
    return list(COARSE_TYPES)


def get_coarse_type(type_id: str) -> str | None:
    if type_id in COARSE_TYPES:
        return type_id
    return FINE_TO_COARSE.get(type_id)
