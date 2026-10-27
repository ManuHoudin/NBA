```mermaid
erDiagram
    PLAYERS {
        int player_id pk
        player_name string 
        team        string 
        age         int
    }
    
    TEAMS {
        team      string pk
        team_name string
    }

    STATS {
        player_id int pk
    games_played int
    wins      int
    losses    int
    average_minutes_match   float
    average_points_match    float
    field_goals_made       float
    field_goals_attempted       float
    minutes_after_15 float
    three_points_attempts  float
    three_points_percentage float
    free_throws_made       float
    free_throws_attempted       float
    free_throw_percentage    float
    offensive_rebounds      float
    defensive_rebounds      float
    total_rebounds       float
    assists       float
    turnovers       float
    steals       float
    blocks       float
    personal_fouls        float
    fantasy_points float
    double_doubles int,
    triple_doubles int,
    plus_minus float
    offensive_rating   float
    defensive_rating   float
    net_rating   float
    assists_percentage   float
    assists_to_turnovers   float
    assists_ratio_percentage   float
    offensive_rebound_percentage  float
    defensive_rebound_percentage  float
    total_rebound_percentage   float
    turnover_ratio_percentage  float
    effective_field_goal_percentage   float
    true_shooting_percentage    float
    usage_rate_percentage   float
    pace      float
    player_impact_estimate       float
    total_possessions      float
    }

    PLAYERS ||--|| STATS : "référence"

    
```