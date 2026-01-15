"""
Tests unitaires pour valider les contraintes sur les données d'événements.

Ce fichier teste que les données respectent les contraintes suivantes :
1. Zone géographique : Île-de-France uniquement
2. Fenêtre temporelle : événements de moins d'un an (365 jours)
3. Schéma minimal : présence des champs obligatoires
4. Cohérence : pas de doublons, dataset non vide

Source de vérité : data/raw/evenements_publics_openagenda.csv

Exécution :
    python -m pytest tests/test_data_constraints.py -v
    OU
    python tests/test_data_constraints.py
"""

import csv
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import Counter


# ============================================================================
# CONSTANTES DE CONTRAINTE (Étape 5.2)
# ============================================================================

# Zone géographique sélectionnée (valeur exacte utilisée lors du filtering)
SELECTED_GEOGRAPHIC_ZONE = "Île-de-France"

# Règle temporelle : événements de moins d'un an
TEMPORAL_WINDOW_DAYS = 365

# Date de référence "aujourd'hui" - peut être figée via variable d'environnement
# pour rendre le test déterministe (Étape 5.10)
def get_reference_date() -> datetime:
    """
    Retourne la date de référence pour les tests.

    Peut être figée via la variable d'environnement TEST_REFERENCE_DATE (format: YYYY-MM-DD)
    pour rendre le test stable dans le temps.

    Returns:
        Date de référence (datetime avec timezone UTC)
    """
    from datetime import timezone

    env_date = os.environ.get('TEST_REFERENCE_DATE')
    if env_date:
        try:
            # Parse la date et ajoute timezone UTC
            dt = datetime.fromisoformat(env_date)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            raise ValueError(
                f"TEST_REFERENCE_DATE invalide: {env_date}. "
                f"Format attendu: YYYY-MM-DD"
            )
    # Retourne datetime avec timezone UTC
    return datetime.now(timezone.utc)


# ============================================================================
# CHARGEMENT DES DONNÉES (Étape 5.3)
# ============================================================================

def load_events_under_test() -> List[Dict[str, str]]:
    """
    Charge la source de vérité des événements depuis le fichier CSV.

    Source: data/raw/evenements_publics_openagenda.csv

    Returns:
        Liste de dictionnaires représentant les événements

    Raises:
        FileNotFoundError: Si le fichier source n'existe pas
        ValueError: Si le dataset est vide
    """
    # Déterminer le chemin du fichier
    project_root = Path(__file__).parent.parent
    csv_path = project_root / "data" / "raw" / "evenements_publics_openagenda.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Fichier source introuvable: {csv_path}\n"
            f"Le fichier de données doit être présent pour exécuter les tests."
        )

    # Charger le CSV
    events = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f, delimiter=';')
        events = list(reader)

    # Vérifier que le dataset n'est pas vide
    if not events:
        raise ValueError(
            f"Le dataset est vide: {csv_path}\n"
            f"Au moins un événement doit être présent."
        )

    return events


# ============================================================================
# NORMALISATION DES DATES (Étape 5.5)
# ============================================================================

def parse_event_date(event: Dict[str, str]) -> datetime:
    """
    Convertit la date d'un événement en objet datetime.

    Règle : utilise le champ "Première date - Début" comme date de référence.
    Ce champ représente la date de début du premier horaire de l'événement.

    Args:
        event: Dictionnaire représentant un événement

    Returns:
        Date normalisée (datetime)

    Raises:
        ValueError: Si la date est absente ou non parsable
    """
    date_field = "Première date - Début"
    date_str = event.get(date_field, '').strip()

    if not date_str:
        raise ValueError(
            f"Date absente pour l'événement '{event.get('Titre', 'N/A')}' "
            f"(ID: {event.get('Identifiant', 'N/A')})\n"
            f"Champ requis: {date_field}"
        )

    try:
        # Parse ISO 8601 format (ex: 2026-05-12T19:00:00+02:00)
        # Replace 'Z' with '+00:00' for compatibility
        normalized_str = date_str.replace('Z', '+00:00')
        return datetime.fromisoformat(normalized_str)
    except (ValueError, AttributeError) as e:
        raise ValueError(
            f"Date non parsable pour l'événement '{event.get('Titre', 'N/A')}' "
            f"(ID: {event.get('Identifiant', 'N/A')})\n"
            f"Valeur: {date_str}\n"
            f"Format attendu: ISO 8601 (ex: 2026-05-12T19:00:00+02:00)\n"
            f"Erreur: {e}"
        )


# ============================================================================
# NORMALISATION DE LA LOCALISATION (Étape 5.7)
# ============================================================================

def extract_event_geo(event: Dict[str, str]) -> str:
    """
    Extrait le champ de localisation permettant de vérifier l'appartenance à la zone.

    Règle : utilise le champ "Région" comme critère d'appartenance géographique.
    Cette règle doit être identique à celle utilisée lors du pré-processing.

    Args:
        event: Dictionnaire représentant un événement

    Returns:
        Nom de la région (str)

    Raises:
        ValueError: Si la localisation est absente ou non exploitable
    """
    geo_field = "Région"
    region = event.get(geo_field, '').strip()

    if not region:
        raise ValueError(
            f"Localisation absente pour l'événement '{event.get('Titre', 'N/A')}' "
            f"(ID: {event.get('Identifiant', 'N/A')})\n"
            f"Champ requis: {geo_field}\n"
            f"Autres infos: Ville={event.get('Ville', 'N/A')}, "
            f"CP={event.get('Code postal', 'N/A')}"
        )

    return region


# ============================================================================
# TESTS UNITAIRES
# ============================================================================

def test_schema_minimal():
    """
    Test 5.4 : Vérifie que chaque événement dispose des champs minimaux nécessaires.

    Champs requis :
    - Identifiant (identifiant unique)
    - Titre (nom de l'événement)
    - Première date - Début (date exploitable)
    - Région (information de localisation)
    """
    events = load_events_under_test()

    required_fields = [
        'Identifiant',
        'Titre',
        'Première date - Début',
        'Région'
    ]

    errors = []

    for i, event in enumerate(events, 1):
        missing_fields = []
        for field in required_fields:
            if not event.get(field, '').strip():
                missing_fields.append(field)

        if missing_fields:
            errors.append(
                f"Événement #{i} (ID: {event.get('Identifiant', 'N/A')}, "
                f"Titre: {event.get('Titre', 'N/A')[:30]}...): "
                f"champs manquants = {missing_fields}"
            )

    if errors:
        raise AssertionError(
            f"ÉCHEC du test de schéma minimal.\n"
            f"{len(errors)} événement(s) avec des champs manquants:\n" +
            "\n".join(f"  - {e}" for e in errors)
        )

    print(f"✓ Test schéma minimal réussi : {len(events)} événements validés")


def test_temporal_constraint_less_than_one_year():
    """
    Test 5.6 : Vérifie que tous les événements ont une date de moins d'un an.

    Règle : la date de début de l'événement ne doit pas être strictement antérieure
    à (date_référence - 365 jours).
    """
    events = load_events_under_test()
    reference_date = get_reference_date()
    cutoff_date = reference_date - timedelta(days=TEMPORAL_WINDOW_DAYS)

    out_of_window_events = []

    for event in events:
        try:
            event_date = parse_event_date(event)

            # Vérifier que l'événement n'est pas trop ancien
            if event_date < cutoff_date:
                days_old = (reference_date - event_date).days
                out_of_window_events.append({
                    'id': event.get('Identifiant', 'N/A'),
                    'title': event.get('Titre', 'N/A'),
                    'date': event_date.isoformat(),
                    'days_old': days_old
                })
        except ValueError as e:
            # Le parsing de date échouera dans test_schema_minimal
            # On ne fait que comptabiliser ici
            pass

    if out_of_window_events:
        raise AssertionError(
            f"ÉCHEC de la contrainte temporelle 'moins d'un an'.\n"
            f"Date de référence: {reference_date.date()}\n"
            f"Date limite (référence - {TEMPORAL_WINDOW_DAYS} jours): {cutoff_date.date()}\n"
            f"{len(out_of_window_events)} événement(s) hors fenêtre:\n" +
            "\n".join(
                f"  - ID: {e['id']}, Titre: {e['title'][:40]}...\n"
                f"    Date: {e['date']}, Ancienneté: {e['days_old']} jours"
                for e in out_of_window_events
            )
        )

    print(
        f"✓ Test contrainte temporelle réussi : {len(events)} événements "
        f"dans la fenêtre de {TEMPORAL_WINDOW_DAYS} jours"
    )


def test_geographic_constraint():
    """
    Test 5.8 : Vérifie que tous les événements appartiennent à la zone géographique sélectionnée.

    Règle : le champ "Région" doit correspondre exactement à SELECTED_GEOGRAPHIC_ZONE.
    """
    events = load_events_under_test()

    out_of_zone_events = []

    for event in events:
        try:
            region = extract_event_geo(event)

            if region != SELECTED_GEOGRAPHIC_ZONE:
                out_of_zone_events.append({
                    'id': event.get('Identifiant', 'N/A'),
                    'title': event.get('Titre', 'N/A'),
                    'region': region,
                    'ville': event.get('Ville', 'N/A'),
                    'cp': event.get('Code postal', 'N/A')
                })
        except ValueError as e:
            # L'extraction géo échouera dans test_schema_minimal
            pass

    if out_of_zone_events:
        raise AssertionError(
            f"ÉCHEC de la contrainte géographique.\n"
            f"Zone attendue: {SELECTED_GEOGRAPHIC_ZONE}\n"
            f"{len(out_of_zone_events)} événement(s) hors zone:\n" +
            "\n".join(
                f"  - ID: {e['id']}, Titre: {e['title'][:40]}...\n"
                f"    Région trouvée: {e['region']}, Ville: {e['ville']} ({e['cp']})"
                for e in out_of_zone_events
            )
        )

    print(
        f"✓ Test contrainte géographique réussi : {len(events)} événements "
        f"dans la zone '{SELECTED_GEOGRAPHIC_ZONE}'"
    )


def test_dataset_non_empty_and_coherent():
    """
    Test 5.9 : Vérifie que le dataset est non vide et cohérent.

    Vérifications :
    - Nombre d'événements > 0
    - Pas de doublons sur (Identifiant)
    - Présence d'une description pour chaque événement
    """
    events = load_events_under_test()

    errors = []

    # (a) Vérifier que le dataset contient au moins un événement
    if len(events) == 0:
        raise AssertionError(
            "ÉCHEC : le dataset est vide.\n"
            "Au moins un événement doit être présent."
        )

    # (b) Vérifier l'absence de doublons sur Identifiant
    identifiers = [e.get('Identifiant', '') for e in events]
    identifier_counts = Counter(identifiers)
    duplicates = {id_: count for id_, count in identifier_counts.items() if count > 1}

    if duplicates:
        errors.append(
            f"Doublons détectés sur le champ 'Identifiant':\n" +
            "\n".join(
                f"  - Identifiant '{id_}' : {count} occurrences"
                for id_, count in duplicates.items()
            )
        )

    # (c) Vérifier la présence d'une description
    # On accepte soit "Description" soit "Description longue"
    events_without_description = []

    for event in events:
        desc = event.get('Description', '').strip()
        desc_long = event.get('Description longue', '').strip()

        if not desc and not desc_long:
            events_without_description.append({
                'id': event.get('Identifiant', 'N/A'),
                'title': event.get('Titre', 'N/A')
            })

    if events_without_description:
        errors.append(
            f"{len(events_without_description)} événement(s) sans description:\n" +
            "\n".join(
                f"  - ID: {e['id']}, Titre: {e['title'][:50]}"
                for e in events_without_description[:5]  # Limiter à 5 pour lisibilité
            ) +
            (f"\n  ... et {len(events_without_description) - 5} autres"
             if len(events_without_description) > 5 else "")
        )

    if errors:
        raise AssertionError(
            f"ÉCHEC du test de cohérence.\n" +
            "\n\n".join(errors)
        )

    print(
        f"✓ Test cohérence réussi : {len(events)} événements, "
        f"{len(identifier_counts)} identifiants uniques, "
        f"{len(events) - len(events_without_description)} avec description"
    )


# ============================================================================
# EXÉCUTION DIRECTE (mode standalone)
# ============================================================================

def run_all_tests():
    """
    Exécute tous les tests et affiche les résultats.

    Returns:
        0 si tous les tests passent, 1 sinon
    """
    tests = [
        ("Schéma minimal", test_schema_minimal),
        ("Contrainte temporelle (< 1 an)", test_temporal_constraint_less_than_one_year),
        ("Contrainte géographique", test_geographic_constraint),
        ("Cohérence du dataset", test_dataset_non_empty_and_coherent),
    ]

    print("=" * 70)
    print("TESTS UNITAIRES - Contraintes sur les données d'événements")
    print("=" * 70)
    print(f"Source: data/raw/evenements_publics_openagenda.csv")
    print(f"Zone géographique: {SELECTED_GEOGRAPHIC_ZONE}")
    print(f"Fenêtre temporelle: {TEMPORAL_WINDOW_DAYS} jours")
    print(f"Date de référence: {get_reference_date().date()}")
    print("=" * 70)
    print()

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            print(f"Exécution du test: {test_name}")
            test_func()
            passed += 1
            print()
        except AssertionError as e:
            print(f"✗ ÉCHEC: {test_name}")
            print(str(e))
            print()
            failed += 1
        except Exception as e:
            print(f"✗ ERREUR: {test_name}")
            print(f"Exception inattendue: {type(e).__name__}: {e}")
            print()
            failed += 1

    print("=" * 70)
    print(f"RÉSULTATS: {passed} réussi(s), {failed} échoué(s)")
    print("=" * 70)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
