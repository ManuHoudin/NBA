# Intégration d'une base SQL dans la ToolChain
Une base de donnée a été créée pour mieux intégrer les données numériques dans la chaîne du RAG.

## Structure de la base
La base nba.db contient les tables players, teams et stats
Voir le MCD dans `schema.md`. Ne pas oublier de mettre à jour ce schéma en cas de modification dans la base.

La clé étrangère permet de s'assurer qu'un enregistrement de stats correspond bien à un player de la table players
Elle permet également, pour un player_id, de supprimer la ligne concernée dans les 2 tables

## Tests
Les tests unitaires se trouvent dans sql/tests. Ils testent:
- la formulation d'une requête par le llm en fonction de la question, son exécution et la récupération du résultat de la requête.
- le nettoyage de la requête renvoyée par le llm avant son exécution dans SQLite.