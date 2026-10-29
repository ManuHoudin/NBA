
import pytest
from pathlib import Path
import sys

# Racine du projet : P10_DSML
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SQL_DIR = PROJECT_ROOT / "sql"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SQL_DIR))

from sql_tool import clean_sql_query

@pytest.mark.parametrize(
    "raw_query, expected",
    [
        # SQL brut
        (
            "SELECT COUNT(*) FROM players",
            "SELECT COUNT(*) FROM players",
        ),

        # Préfixe SQLQuery
        (
            "SQLQuery: SELECT COUNT(*) FROM players",
            "SELECT COUNT(*) FROM players",
        ),

        # Balises Markdown
        (
            "```sql\nSELECT COUNT(*) FROM players;\n```",
            "SELECT COUNT(*) FROM players",
        ),

        # Préambule explicatif
        (
            "Voici la requête : SELECT COUNT(*) FROM players",
            "SELECT COUNT(*) FROM players",
        ),

        # Requête sur plusieurs lignes
        (
            "SELECT player_name\nFROM players\nLIMIT 10;",
            "SELECT player_name\nFROM players\nLIMIT 10",
        ),

        # Requête commençant par WITH
        (
            "WITH top_players AS "
            "(SELECT player_name FROM players) "
            "SELECT * FROM top_players",
            "WITH top_players AS "
            "(SELECT player_name FROM players) "
            "SELECT * FROM top_players",
        ),
    ],
)
def test_clean_sql_query(raw_query, expected):
    assert clean_sql_query(raw_query) == expected


@pytest.mark.parametrize(
    "raw_query",
    [
        "",
        "Je ne peux pas générer de SQL.",
        "UPDATE players SET player_name = 'test'",
        "SELECT * FROM players; DROP TABLE players",
    ],
)
def test_clean_sql_query_rejects_invalid_input(raw_query):
    with pytest.raises(ValueError):
        clean_sql_query(raw_query)
