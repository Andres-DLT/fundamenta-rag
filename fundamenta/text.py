"""Módulo 03: normalización y tokenización de texto en español.

Convierte un texto en la lista de términos con la que trabajan los índices de búsqueda:
minúsculas, sin tildes, sin palabras vacías (stopwords) y con un stemming ligero que
agrupa variantes de una misma palabra ("vacación", "vacaciones" -> "vacacio").
"""

from __future__ import annotations

import re
import unicodedata

_STOPWORDS_RAW = """
a al algo algun alguna algunas alguno algunos ante antes aqui asi aun cada como con
contra cual cuales cualquier cuando cuanta cuantas cuanto cuantos de del desde donde
dos e el ella ellas ello ellos en entre era eran es esa esas ese eso esos esta estan
estas este esto estos fue fueron ha hace hacer han hasta hay la las le les lo los mas
me mi mis mucho muy nada ni no nos nosotros o os otra otras otro otros para pero poco
por porque puede pueden que quien quienes se sea segun ser si sido sin sobre solo son
su sus tal tambien tanto te tener tengo tiene tienen todo todos tu tus un una uno unos
usted ustedes y ya yo
"""

SPANISH_STOPWORDS = frozenset(_STOPWORDS_RAW.split())


def strip_accents(text: str) -> str:
    """Elimina tildes y diéresis: "información" -> "informacion", "ñ" -> "n"."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def light_stem(word: str, max_length: int = 7) -> str:
    """Stemming ligero para español.

    1. Quita la "s" final del plural ("permisos" -> "permiso").
    2. Quita la vocal final en palabras largas ("permiso" -> "permis").
    3. Trunca a `max_length` caracteres ("vacacion" -> "vacacio").

    Es deliberadamente simple: agrupa la mayoría de variantes de género, número y
    algunas formas verbales, a cambio de alguna colisión ocasional.
    """
    if len(word) > 3 and word.endswith("s"):
        word = word[:-1]
    if len(word) > 4 and word[-1] in "aeo":
        word = word[:-1]
    return word[:max_length]


def tokenize(
    text: str, stopwords: frozenset[str] = SPANISH_STOPWORDS, stem: bool = True
) -> list[str]:
    """Convierte un texto en una lista de términos normalizados."""
    words = re.findall(r"[a-z0-9]+", strip_accents(text.lower()))
    terms = [word for word in words if word not in stopwords]
    return [light_stem(term) for term in terms] if stem else terms
