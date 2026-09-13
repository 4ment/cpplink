# Copyright 2026 Mathieu Fourment
# SPDX-License-Identifier: MIT
"""Guess what a column holds, from its name and from its values.

A role is what the schema templates key on: ``surname`` gets a name
comparison with term frequency and rare-value blocking, ``dob`` an exact and
date-within comparison with exact-value blocking. Two signals are read
independently, the column's name against a synonym table and its values
against the content probes, and combined: agreement is confident, a name with
no content evidence is a guess, and content that contradicts the name is
reported as such so the page can show both. Nothing here is more than a
starting point for the dropdown.
"""

from .stats import normalise_name

ROLES = ["none", "id", "first_name", "surname", "full_name", "dob", "date", "email",
         "phone", "postcode", "gender", "address", "city", "state", "country",
         "latitude", "longitude", "number", "category", "list"]

# Synonyms are matched on the name reduced to letters, so `date_of_birth`,
# `DateBirth` and `dob` all read as the same key.
SYNONYMS = {
    "id": ["id", "uid", "uuid", "recordid", "rowid", "key", "identifier", "pk",
           "uniqueid", "personid", "customerid", "patientid", "clientid"],
    "first_name": ["firstname", "givenname", "forename", "fname", "first", "prenom",
                   "given", "firstnames", "name1", "nombre"],
    "surname": ["surname", "lastname", "familyname", "lname", "last", "nom", "apellido",
                "family", "secondname", "sname", "cognome", "nachname"],
    "full_name": ["name", "fullname", "personname", "firstandsurname", "completename",
                  "displayname", "nomcomplet"],
    "dob": ["dob", "birthdate", "dateofbirth", "datebirth", "birthday", "born", "birth",
            "dateofbirt", "datenaissance", "naissance", "bdate", "birthdt"],
    "date": ["date", "dt", "eventdate", "recorddate", "registered", "created",
             "registrationdate", "datedeath", "deathdate", "dod"],
    "email": ["email", "emailaddress", "mail", "courriel", "emailaddr"],
    "phone": ["phone", "telephone", "tel", "mobile", "cell", "phonenumber", "mobilenumber",
              "cellphone", "homephone", "workphone", "fax", "telefono"],
    "postcode": ["postcode", "zip", "zipcode", "postalcode", "postal", "pincode", "cp",
                 "codepostal", "plz"],
    "gender": ["gender", "sex", "sexe", "genre"],
    "address": ["address", "street", "addr", "address1", "streetaddress", "adresse",
                "addressline", "addressline1", "streetname", "streetnumber"],
    "city": ["city", "town", "suburb", "locality", "ville", "municipality", "place"],
    "state": ["state", "province", "region", "county", "territory", "etat"],
    "country": ["country", "pays", "nation", "countrycode"],
    "latitude": ["latitude", "lat", "y"],
    "longitude": ["longitude", "lon", "lng", "long", "x"],
}


def guess_from_name(column):
    key = normalise_name(column)
    for role, names in SYNONYMS.items():
        if key in names:
            return role
    # A suffix or prefix match catches `cust_first_name` and `birth_date_1`.
    for role, names in SYNONYMS.items():
        for name in names:
            if len(name) >= 4 and (key.endswith(name) or key.startswith(name)):
                return role
    return None


def guess_from_content(probe, stats, cpplink_type):
    """The role the values themselves argue for, or None."""
    if probe.get("kind") == "list":
        return "list"
    if cpplink_type == "boolean":
        return "category"
    if cpplink_type == "date":
        return "dob"
    if cpplink_type == "double":
        return "number"
    n = probe.get("sampled") or 0
    if n == 0:
        return None
    if stats["distinct"] == stats["rows"] - stats["nulls"] and stats["nulls"] == 0 \
            and stats["rows"] > 1:
        return "id"
    if probe["email"] > 0.8:
        return "email"
    if probe["date"] > 0.9:
        return "dob"
    if probe["gender"] > 0.9 and stats["distinct"] <= 6:
        return "gender"
    if probe["phone"] > 0.8 and probe["digits"] < 0.98 or \
            (probe["digits"] > 0.9 and 7 <= (probe["mean_length"] or 0) <= 15):
        return "phone"
    if probe["postcode"] > 0.8 and (probe["mean_length"] or 0) <= 10:
        return "postcode"
    if probe["numeric"] > 0.95:
        return "number"
    if probe["name"] > 0.9 and (probe["mean_length"] or 0) < 20:
        return "first_name"
    if stats["distinct"] <= 50:
        return "category"
    return None


# What each role is compatible with, so a content guess of `first_name` on a
# column named `surname` is not read as a contradiction: both are names.
NAME_ROLES = {"first_name", "surname", "full_name", "city", "state", "country",
              "address"}
DATE_ROLES = {"dob", "date"}


def guess_role(column, probe, stats, cpplink_type):
    """(role, confidence, reason). Confidence is one of `high`, `medium`, `low`."""
    by_name = guess_from_name(column)
    by_content = guess_from_content(probe, stats, cpplink_type)
    if by_name and by_content:
        if by_name == by_content:
            return by_name, "high", "name and values agree"
        if by_name in NAME_ROLES and by_content in NAME_ROLES:
            return by_name, "high", f"name says {by_name}, values look like a name"
        if by_name in DATE_ROLES and by_content in DATE_ROLES:
            return by_name, "high", f"name says {by_name}, values parse as dates"
        if by_content == "number" and by_name in ("latitude", "longitude", "number",
                                                   "postcode", "phone", "id"):
            return by_name, "high", f"name says {by_name}, values are numeric"
        if by_content == "list" and by_name != "list":
            return by_name, "medium", f"name says {by_name}, held as a list"
        if by_content in ("category", "number") and by_name in (
                "gender", "state", "country", "postcode", "city"):
            return by_name, "medium", f"name says {by_name}, values are categorical"
        return by_name, "low", f"name says {by_name}, values look like {by_content}"
    if by_name:
        return by_name, "medium", "from the name only"
    if by_content:
        return by_content, "low", "from the values only"
    return "none", "low", "no signal"


def type_for_role(role, current):
    """The cpplink type a role implies where it implies one, else the current."""
    if role in ("dob", "date"):
        return "date"
    if role in ("latitude", "longitude", "number"):
        return "double"
    if role == "list":
        return "string_list"
    return current
