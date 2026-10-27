PRAGMA foreign_keys = ON;

-- 1. Identité des joueurs
CREATE TABLE IF NOT EXISTS players (
    player_id   INTEGER PRIMARY KEY,
    player_name TEXT NOT NULL UNIQUE,
    team        TEXT NOT NULL,
    age         INTEGER
);

-- 2. Label des équipes
CREATE TABLE IF NOT EXISTS teams (
    team        TEXT PRIMARY KEY,
    team_label   TEXT NOT NULL

);

-- 3. Statistiques agrégées sur la saison
CREATE TABLE IF NOT EXISTS stats (
    player_id INTEGER PRIMARY KEY,
    games_played INTEGER,
    wins      INTEGER,
    losses    INTEGER,
    average_minutes_match   REAL,
    average_points_match    REAL,
    field_goals_made       REAL,
    field_goals_attempted       REAL,
    field_goal_percentage    REAL,
    minutes_after_15 REAL,
    three_points_attempts  REAL,
    three_points_percentage REAL,
    free_throws_made       REAL,
    free_throws_attempted       REAL,
    free_throw_percentage    REAL,
    offensive_rebounds      REAL,
    defensive_rebounds      REAL,
    total_rebounds       REAL,
    assists       REAL,
    turnovers       REAL,
    steals       REAL,
    blocks       REAL,
    personal_fouls        REAL,
    fantasy_points REAL,
    double_doubles INTEGER,
    triple_doubles INTEGER,
    plus_minus REAL,
    offensive_rating   REAL,
    defensive_rating   REAL,
    net_rating   REAL,
    assists_percentage   REAL,
    assists_to_turnovers   REAL,
    assists_ratio_percentage   REAL,
    offensive_rebound_percentage  REAL,
    defensive_rebound_percentage  REAL,
    total_rebound_percentage   REAL,
    turnover_ratio_percentage  REAL,
    effective_field_goal_percentage   REAL,
    true_shooting_percentage    REAL,
    usage_rate_percentage   REAL,
    pace      REAL,
    player_impact_estimate       REAL,
    total_possessions      REAL,


    FOREIGN KEY (player_id)
        REFERENCES players(player_id)
        ON DELETE CASCADE
);

-- Index pour les filtres SQL courants
CREATE INDEX IF NOT EXISTS idx_players_team
    ON players(team);
