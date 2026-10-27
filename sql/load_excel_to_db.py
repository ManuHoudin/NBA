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
SQL_FILE = Path(__file__).resolve().parent / "create_table.sql"
DB_FILE = "sql/nba.db"
SHEET_NAME = "Données NBA"
TEAM_SHEET_NAME = "Equipe"

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
    "GP": "games_played",
    "W": "wins",
    "L": "losses",
    "Min": "average_minutes_match",
    "PTS": "average_points_match",
    "FGM": "field_goals_made",
    "FGA": "field_goals_attempted",
    "FG%": "field_goal_percentage",
    "15:00:00": "minutes_after_15",
    "3PA": "three_points_attempts",
    "3P%": "three_points_percentage",
    "FTM": "free_throws_made",
    "FTA": "free_throws_attempted",
    "FT%": "free_throw_percentage",
    "OREB": "offensive_rebounds",
    "DREB": "defensive_rebounds",
    "REB": "total_rebounds",
    "AST": "assists",
    "TOV": "turnovers",
    "STL": "steals",
    "BLK": "blocks",
    "PF": "personal_fouls",
    "FP": "fantasy_points",
    "DD2": "double_doubles",
    "TD3": "triple_doubles",
    "+/-": "plus_minus",
    "OFFRTG": "offensive_rating",
    "DEFRTG": "defensive_rating",
    "NETRTG": "net_rating",
    "AST%": "assists_percentage",
    "AST/TO": "assists_to_turnovers",
    "AST RATIO": "assists_ratio_percentage",
    "OREB%": "offensive_rebound_percentage",
    "DREB%": "defensive_rebound_percentage",
    "REB%": "total_rebound_percentage",
    "TO RATIO": "turnover_ratio_percentage",
    "EFG%": "effective_field_goal_percentage",
    "TS%": "true_shooting_percentage",
    "USG%": "usage_rate_percentage",
    "PACE": "pace",
    "PIE": "player_impact_estimate",
    "POSS": "total_possessions",
}


# --------------------------------------------------
# Modèle Pydantic : une ligne Excel = un joueur
# --------------------------------------------------

class NBAPlayerRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    player_name: str = Field(min_length=1)
    team: str = Field(min_length=1)
    age: int | None = Field(default=None, ge=0)

    games_played: int | None = Field(default=None, ge=0)
    wins: int | None = Field(default=None, ge=0)
    losses: int | None = Field(default=None, ge=0)

    average_minutes_match: float | None = Field(default=None, ge=0)
    average_points_match: float | None = Field(default=None, ge=0)
    field_goals_made: float | None = Field(default=None, ge=0)
    field_goals_attempted: float | None = Field(default=None, ge=0)
    field_goal_percentage: float | None = Field(default=None, ge=0, le=100)
    minutes_after_15: float | None = Field(default=None, ge=0)

    three_points_attempts: float | None = Field(default=None, ge=0)
    three_points_percentage: float | None = Field(default=None, ge=0, le=100)

    free_throws_made: float | None = Field(default=None, ge=0)
    free_throws_attempted: float | None = Field(default=None, ge=0)
    free_throw_percentage: float | None = Field(default=None, ge=0, le=100)

    offensive_rebounds: float | None = Field(default=None, ge=0)
    defensive_rebounds: float | None = Field(default=None, ge=0)
    total_rebounds: float | None = Field(default=None, ge=0)
    assists: float | None = Field(default=None, ge=0)
    turnovers: float | None = Field(default=None, ge=0)
    steals: float | None = Field(default=None, ge=0)
    blocks: float | None = Field(default=None, ge=0)
    personal_fouls: float | None = Field(default=None, ge=0)
    fantasy_points: float | None = None

    double_doubles: int | None = Field(default=None, ge=0)
    triple_doubles: int | None = Field(default=None, ge=0)
    plus_minus: float | None = None

    offensive_rating: float | None = None
    defensive_rating: float | None = None
    net_rating: float | None = None
    assists_percentage: float | None = Field(default=None, ge=0, le=100)
    assists_to_turnovers: float | None = Field(default=None, ge=0)
    assists_ratio_percentage: float | None = Field(default=None, ge=0, le=100)
    offensive_rebound_percentage: float | None = Field(default=None, ge=0, le=100)
    defensive_rebound_percentage: float | None = Field(default=None, ge=0, le=100)
    total_rebound_percentage: float | None = Field(default=None, ge=0, le=100)
    turnover_ratio_percentage: float | None = Field(default=None, ge=0)
    effective_field_goal_percentage: float | None = Field(default=None, ge=0, le=100)
    true_shooting_percentage: float | None = Field(default=None, ge=0, le=100)
    usage_rate_percentage: float | None = Field(default=None, ge=0, le=100)
    pace: float | None = Field(default=None, ge=0)
    player_impact_estimate: float | None = None
    total_possessions: float | None = Field(default=None, ge=0)

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
            f"Fichier Excel introuvable : {EXCEL_FILE.resolve()}"
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
        PERCENTAGE_COLUMNS = ["field_goal_percentage", "three_points_percentage", "free_throw_percentage", "assists_percentage",
        "net_rating", "effective_field_goal_percentage", "true_shooting_percentage", "usage_rate_percentage"]

        for col in PERCENTAGE_COLUMNS:
            value = data.get(col)

            if value is not None and pd.notna(value):
                if not 0 <= float(value) <= 100:
                    print(f"Correction : {col} = {value} -> None")
                    data[col] = None

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
    "games_played", "wins", "losses", "average_minutes_match", "average_points_match",
    "field_goals_made", "field_goals_attempted", "field_goal_percentage", "minutes_after_15",
    "three_points_attempts", "three_points_percentage", "free_throws_made", "free_throws_attempted", "free_throw_percentage",
    "offensive_rebounds", "defensive_rebounds", "total_rebounds", "assists", "turnovers", "steals", "blocks",
    "personal_fouls", "fantasy_points", "double_doubles", "triple_doubles",
    "plus_minus", "offensive_rating", "defensive_rating", "net_rating",
    "assists_percentage", "assists_to_turnovers", "assists_ratio_percentage", "offensive_rebound_percentage",
    "defensive_rebound_percentage", "total_rebound_percentage", "turnover_ratio_percentage", "effective_field_goal_percentage",
    "true_shooting_percentage", "usage_rate_percentage", "pace", "player_impact_estimate", "total_possessions",
]

def load_teams(conn):
    # Lecture de l'onglet Equipe
    df_teams = pd.read_excel(
        EXCEL_FILE,
        sheet_name=TEAM_SHEET_NAME,
        skiprows=1 
    )

    teams = []

    for _, row in df_teams.iterrows():
        team = str(row.iloc[0]).strip()
        team_label = str(row.iloc[1]).strip()

        # Ignorer les lignes incomplètes
        if not team or not team_label or team == "nan" or team_label == "nan":
            continue

        teams.append((team, team_label))

    # Insertion ou mise à jour des équipes
    conn.executemany(
        """
        INSERT OR REPLACE INTO teams (team, team_label)
        VALUES (?, ?)
        """,
        teams
    )

    print(f"{len(teams)} équipes chargées dans la table teams.")

def create_tables(conn: sqlite3.Connection) -> None:
    if not SQL_FILE.exists():
        raise FileNotFoundError(
            f"Fichier SQL introuvable : {SQL_FILE.resolve()}"
        )

    with SQL_FILE.open("r", encoding="utf-8") as sql_file:
        sql_script = sql_file.read()

    for i, line in enumerate(sql_script.splitlines(), start=1):
        print(f"{i:3}: {line}")

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
            load_teams(conn)
            insert_players(conn, players)

    logger.info(
        "Import terminé : %s joueurs dans %s",
        len(players),
        DB_FILE,
    )


if __name__ == "__main__":
    main()