PRAGMA foreign_keys = ON;

-- 1. Identité des joueurs
CREATE TABLE IF NOT EXISTS players (
    player_id   INTEGER PRIMARY KEY,
    player_name TEXT NOT NULL UNIQUE,
    team        TEXT NOT NULL,
    age         INTEGER
);

-- 2. Statistiques agrégées sur la saison
CREATE TABLE IF NOT EXISTS stats (
    player_id INTEGER NOT NULL,
    gp        INTEGER,
    wins      INTEGER,
    losses    INTEGER,
    minutes   REAL,
    points    REAL,
    fgm       REAL,
    fga       REAL,
    fg_pct    REAL,
    minutes_after_15 REAL,
    three_pa  REAL,
    three_pct REAL,
    ftm       REAL,
    fta       REAL,
    ft_pct    REAL,
    oreb      REAL,
    dreb      REAL,
    reb       REAL,
    ast       REAL,
    tov       REAL,
    stl       REAL,
    blk       REAL,
    pf        REAL,
    fantasy_points REAL,
    double_doubles INTEGER,
    triple_doubles INTEGER,
    plus_minus REAL,
    off_rtg   REAL,
    def_rtg   REAL,
    net_rtg   REAL,
    ast_pct   REAL,
    ast_to    REAL,
    ast_ratio REAL,
    oreb_pct  REAL,
    dreb_pct  REAL,
    reb_pct   REAL,
    to_ratio  REAL,
    efg_pct   REAL,
    ts_pct    REAL,
    usg_pct   REAL,
    pace      REAL,
    pie       REAL,
    poss      REAL,

    PRIMARY KEY player_id,

    FOREIGN KEY (player_id)
        REFERENCES players(player_id)
        ON DELETE CASCADE
);

-- Index pour les filtres SQL courants
CREATE INDEX IF NOT EXISTS idx_players_team
    ON players(team);

CREATE INDEX IF NOT EXISTS idx_stats_points
    ON stats(points);

CREATE INDEX IF NOT EXISTS idx_stats_assists
    ON stats(ast);