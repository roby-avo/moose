from __future__ import annotations

import json

DEFAULT_TEXT_SAMPLE = (
    "We collect email addresses and IP addresses for account creation and fraud prevention, "
    "and we share them with our payment processor."
)

DEFAULT_TABLE_SAMPLE = json.dumps(
    [
        {"name": "Alice Smith", "email": "alice@example.com", "age": "29", "ip_address": "192.168.0.1", "country": "DE"},
        {"name": "Bob Jones", "email": "bob@example.com", "age": "41", "ip_address": "10.0.0.5", "country": "DE"},
    ],
    ensure_ascii=True,
    indent=2,
)

DEFAULT_CPA_TABLE_SAMPLE = json.dumps(
    [
        {"BookName": "A Handbook for Morning Time", "Language": "English", "Date": "01-01-2016"},
        {"BookName": "The Intentional Brain", "Language": "English", "Date": "15-06-2016"},
        {"BookName": "The Comeback", "Language": "English", "Date": "03-08-2020"},
    ],
    ensure_ascii=True,
    indent=2,
)