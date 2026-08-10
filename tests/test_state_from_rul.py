import sys
import os

# Permettre l'import du module api
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))

from grafana_bridge import state_from_rul


def test_moteur_critique():
    """Un RUL sous 30 doit etre classe CRITIQUE."""
    assert state_from_rul(0) == "CRITIQUE"
    assert state_from_rul(15) == "CRITIQUE"
    assert state_from_rul(29.9) == "CRITIQUE"


def test_moteur_a_surveiller():
    """Un RUL entre 30 et 60 doit etre classe A_SURVEILLER."""
    assert state_from_rul(30) == "A_SURVEILLER"
    assert state_from_rul(45) == "A_SURVEILLER"
    assert state_from_rul(59.9) == "A_SURVEILLER"


def test_moteur_ok():
    """Un RUL de 60 ou plus doit etre classe OK."""
    assert state_from_rul(60) == "OK"
    assert state_from_rul(100) == "OK"
    assert state_from_rul(125) == "OK"


def test_bornes_exactes():
    """Verifie le comportement precis aux seuils."""
    # Exactement 30 -> plus critique, devient A_SURVEILLER
    assert state_from_rul(30) == "A_SURVEILLER"
    # Exactement 60 -> plus a surveiller, devient OK
    assert state_from_rul(60) == "OK"