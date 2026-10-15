from pathlib import Path
import logging
import sqlite3

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


# --------------------------------------------------
# Configuration
# --------------------------------------------------

#BASE_DIR = Path(__file__).resolve().parent

EXCEL_FILE = "inputs/regular NBA.xlsx"
SQL_FILE = "sql/create_table.sql"
DB_FILE = "sql/nba.db"
SHEET_NAME = "Données NBA"

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------
# Correspondance Excel -> modèle Python / SQL
# --------------------------------------------------

COLUMN_MAPPING = {
    "Player": "player_name",
    "Team": "team",
    "Age": "age",
    "GP": "gp",
    "W": "wins",
    "L": "losses",
    "Min": "minutes",
    "PTS": "points",
    "FGM": "fgm",
    "FGA": "fga",
    "FG%": "fg_pct",
    "15:00:00": "minutes_after_15",
    "3PA": "three_pa",
    "3P%": "three_pct",
    "FTM": "ftm",
    "FTA": "fta",
    "FT%": "ft_pct",
    "OREB": "oreb",
    "DREB": "dreb",
    "REB": "reb",
    "AST": "ast",
    "TOV": "tov",
    "STL": "stl",
    "BLK": "blk",
    "PF": "pf",
    "FP": "fantasy_points",
    "DD2": "double_doubles",
    "TD3": "triple_doubles",
    "+/-": "plus_minus",
    "OFFRTG": "off_rtg",
    "DEFRTG": "def_rtg",
    "NETRTG": "net_rtg",
    "AST%": "ast_pct",
    "AST/TO": "ast_to",
    "AST RATIO": "ast_ratio",
    "OREB%": "oreb_pct",
    "DREB%": "dreb_pct",
    "REB%": "reb_pct",
    "TO RATIO": "to_ratio",
    "EFG%": "efg_pct",
    "TS%": "ts_pct",
    "USG%": "usg_pct",
    "PACE": "pace",
    "PIE": "pie",
    "POSS": "poss",
}


# --------------------------------------------------
# Modèle Pydantic : une ligne Excel = un joueur
# --------------------------------------------------

class NBAPlayerRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    player_name: str = Field(min_length=1)
    team: str = Field(min_length=1)
    age: int | None = Field(default=None, ge=0)

    gp: int | None = Field(default=None, ge=0)
    wins: int | None = Field(default=None, ge=0)
    losses: int | None = Field(default=None, ge=0)

    minutes: float | None = Field(default=None, ge=0)
    points: float | None = Field(default=None, ge=0)
    fgm: float | None = Field(default=None, ge=0)
    fga: float | None = Field(default=None, ge=0)
    fg_pct: float | None = Field(default=None, ge=0, le=100)
    minutes_after_15: float | None = Field(default=None, ge=0)

    three_pa: float | None = Field(default=None, ge=0)
    three_pct: float | None = Field(default=None, ge=0, le=100)

    ftm: float | None = Field(default=None, ge=0)
    fta: float | None = Field(default=None, ge=0)
    ft_pct: float | None = Field(default=None, ge=0, le=100)

    oreb: float | None = Field(default=None, ge=0)
    dreb: float | None = Field(default=None, ge=0)
    reb: float | None = Field(default=None, ge=0)
    ast: float | None = Field(default=None, ge=0)
    tov: float | None = Field(default=None, ge=0)
    stl: float | None = Field(default=None, ge=0)
    blk: float | None = Field(default=None, ge=0)
    pf: float | None = Field(default=None, ge=0)
    fantasy_points: float | None = None

    double_doubles: int | None = Field(default=None, ge=0)
    triple_doubles: int | None = Field(default=None, ge=0)
    plus_minus: float | None = None

    off_rtg: float | None = None
    def_rtg: float | None = None
    net_rtg: float | None = None
    ast_pct: float | None = Field(default=None, ge=0, le=100)
    ast_to: float | None = Field(default=None, ge=0)
    ast_ratio: float | None = Field(default=None, ge=0)
    oreb_pct: float | None = Field(default=None, ge=0, le=100)
    dreb_pct: float | None = Field(default=None, ge=0, le=100)
    reb_pct: float | None = Field(default=None, ge=0, le=100)
    to_ratio: float | None = Field(default=None, ge=0)
    efg_pct: float | None = Field(default=None, ge=0, le=100)
    ts_pct: float | None = Field(default=None, ge=0, le=100)
    usg_pct: float | None = Field(default=None, ge=0, le=100)
    pace: float | None = Field(default=None, ge=0)
    pie: float | None = None
    poss: float | None = Field(default=None, ge=0)

    @field_validator("*", mode="before")
    @classmethod
    def normalize_empty_values(cls, value):
        """Convertit les cellules vides en None et les nombres décimaux français."""
        if value is None or pd.isna(value):
            return None

        if isinstance(value, str):
            value = value.strip()

            if value in {"", "-", "—", "N/A", "n/a"}:
                return None

            # Gère les nombres au format français, par exemple "12,5".
            value = value.replace(",", ".")

        return value
# --------------------------------------------------
# Validation complémentaire pour les pourcentages (0-100)
# --------------------------------------------------
def validate_percentage(value):
    if pd.isna(value):
        return None

    value = float(value)

    if not 0.0 <= value <= 100.0:
        print(f"Pourcentage invalide ({value}), remplacé par None")
        return None

    return value

# --------------------------------------------------
# Lecture et validation Excel
# --------------------------------------------------

def load_and_validate_excel() -> list[NBAPlayerRow]:
    if not EXCEL_FILE:
        raise FileNotFoundError(
            f"Fichier Excel introuvable : {EXCEL_FILE}"
        )

    df = pd.read_excel(EXCEL_FILE, sheet_name=SHEET_NAME, skiprows=1)

    # Évite les erreurs dues aux espaces autour des noms de colonnes.
    df.columns = [str(column).strip() for column in df.columns]

    missing_columns = set(COLUMN_MAPPING) - set(df.columns)
    if missing_columns:
        raise ValueError(
            "Colonnes Excel manquantes : "
            + ", ".join(sorted(missing_columns))
        )

    # On conserve uniquement les colonnes attendues et on les renomme.
    df = df[list(COLUMN_MAPPING)].rename(columns=COLUMN_MAPPING)

    validated_rows = []
    errors = []

    for index, row in df.iterrows():
        # Les numéros de lignes Excel commencent à 1.
        excel_row_number = index + 3

        data = row.to_dict()

        # on contrôle les pourcentages avant la validation Pydantic
        PERCENTAGE_COLUMNS = ["fg_pct", "three_pct", "ft_pct", "ast_pct",
        "oreb_pct", "dreb_pct", "reb_pct", "efg_pct", "ts_pct", "usg_pct"]

        for col in PERCENTAGE_COLUMNS:
            if col in row:
                row[col] = validate_percentage(row[col])

        # Validation Pydantic
        try:
            player = NBAPlayerRow.model_validate(data)
            validated_rows.append(player)

        except ValidationError as exc:
            errors.append(
                {
                    "row": excel_row_number,
                    "player": data.get("player_name"),
                    "errors": exc.errors(
                        include_input=False,
                        include_url=False,
                    ),
                }
            )

    if errors:
        for error in errors:
            logger.error(
                "Ligne Excel %s, joueur %r : %s",
                error["row"],
                error["player"],
                error["errors"],
            )

        raise ValueError(
            f"Validation échouée : {len(errors)} ligne(s) invalide(s). "
            "Aucune donnée n'a été insérée."
        )

    # Contrôle complémentaire : un seul enregistrement par joueur.
    names = [player.player_name.strip() for player in validated_rows]

    if len(names) != len(set(names)):
        duplicates = sorted(
            {name for name in names if names.count(name) > 1}
        )
        raise ValueError(
            f"Noms de joueurs en double : {duplicates}"
        )

    logger.info(
        "%s lignes validées par Pydantic.",
        len(validated_rows),
    )

    return validated_rows


# --------------------------------------------------
# Création des tables et insertion en transaction
# --------------------------------------------------

STAT_COLUMNS = [
    "gp", "wins", "losses", "minutes", "points",
    "fgm", "fga", "fg_pct", "minutes_after_15",
    "three_pa", "three_pct", "ftm", "fta", "ft_pct",
    "oreb", "dreb", "reb", "ast", "tov", "stl", "blk",
    "pf", "fantasy_points", "double_doubles", "triple_doubles",
    "plus_minus", "off_rtg", "def_rtg", "net_rtg",
    "ast_pct", "ast_to", "ast_ratio", "oreb_pct",
    "dreb_pct", "reb_pct", "to_ratio", "efg_pct",
    "ts_pct", "usg_pct", "pace", "pie", "poss",
]


def create_tables(conn: sqlite3.Connection) -> None:
    if not SQL_FILE.exists():
        raise FileNotFoundError(
            f"Fichier SQL introuvable : {SQL_FILE}"
        )

    with open(SQL_FILE, "r", encoding="utf-8") as SQL_FILE:
        sql_script = SQL_FILE.read()
    
    conn.executescript(sql_script)

    logger.info("Script create_table.sql exécuté.")


def insert_players(
    conn: sqlite3.Connection,
    players: list[NBAPlayerRow],
) -> None:
    for player in players:
        data = player.model_dump()

        cursor = conn.execute(
            """
            INSERT INTO players (player_name, team, age)
            VALUES (?, ?, ?)
            """,
            (
                data["player_name"].strip(),
                data["team"].strip(),
                data["age"],
            ),
        )

        # SQLite génère le player_id : aucune génération manuelle nécessaire.
        player_id = cursor.lastrowid

        stat_values = [data[column] for column in STAT_COLUMNS]

        columns_sql = ", ".join(["player_id"] + STAT_COLUMNS)
        placeholders = ", ".join(["?"] * (len(STAT_COLUMNS) + 1))

        conn.execute(
            f"""
            INSERT INTO stats ({columns_sql})
            VALUES ({placeholders})
            """,
            [player_id] + stat_values,
        )


def main() -> None:
    # 1. Valider toutes les lignes AVANT l'insertion.
    players = load_and_validate_excel()

    # 2. Créer les tables et insérer toutes les données atomiquement.
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("PRAGMA foreign_keys = ON")

        # En cas d'erreur, les insertions de la transaction sont annulées.
        with conn:
            create_tables(conn)
            insert_players(conn, players)

    logger.info(
        "Import terminé : %s joueurs dans %s",
        len(players),
        DB_FILE,
    )


if __name__ == "__main__":
    main()