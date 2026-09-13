# Copyright 2026 Mathieu Fourment
# SPDX-License-Identifier: MIT

from cpplink_studio import guess, stats


def test_names_map_to_roles():
    cases = {"surname": "surname", "LastName": "surname", "family_name": "surname",
             "nom": "surname", "first_name": "first_name", "given_name": "first_name",
             "dob": "dob", "date_of_birth": "dob", "DateBirth": "dob", "birth_date": "dob",
             "email_address": "email", "phone_number": "phone", "zip": "postcode",
             "post_code": "postcode", "sex": "gender", "lat": "latitude",
             "lng": "longitude", "record_id": "id", "cust_first_name": "first_name",
             "weight_kg": None}
    for name, role in cases.items():
        assert guess.guess_from_name(name) == role, name


def test_roles_on_sample(dataset):
    types = {c: t for c, (t, _) in dataset.proposed_types().items()}
    expected = {"id": "id", "first_name": "first_name", "last_name": "surname",
                "dob": "dob", "email": "email", "phone": "phone", "postcode": "postcode",
                "gender": "gender", "latitude": "latitude", "longitude": "longitude"}
    for column, role in expected.items():
        st = stats.column_stats(dataset, column)
        probe = stats.probe_content(dataset, column)
        got = guess.guess_role(column, probe, st, types[column])
        assert got[0] == role, (column, got)


def test_content_alone_finds_email_and_dates(dataset):
    st = stats.column_stats(dataset, "email")
    probe = stats.probe_content(dataset, "email")
    assert guess.guess_role("column_7", probe, st, "string")[0] == "email"
    st = stats.column_stats(dataset, "dob")
    probe = stats.probe_content(dataset, "dob")
    assert guess.guess_role("column_3", probe, st, "date")[0] == "dob"
