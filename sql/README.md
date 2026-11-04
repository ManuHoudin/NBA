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
- le nettoyage de la requête renvoyée par le llm avant son exécution dans SQLite (test_clean_sql).

## Tests Ragas
- lancer le test avec le script tests/evaluation/evaluate_rag.py
- le fichier json qui contient les cas de tests est dans tests/evaluation/rag_test_cases.json. Il contient des cas adressant uniquement le traitement des pdf par le llm, d'autres le traitement des données via des requêtes SQL et enfin des cas mixtes.

Les métriques suivantes sont calculées :
- Context Precision : est-ce que le tool (Faiss ou SQL) place les passages utiles en tête des résultats, sans trop de passages hors-sujet ?
- Context Recall : est-ce que le tool (Faiss ou SQL) retrouve suffisamment d'informations pour répondre correctement à la question ?
- Faithfulness : est-ce que la réponse est la même après avoir posé la même question plusieurs fois ?
- Response relevancy : est-ce que la réponse apportée est pertinente par rapport à la question posée ?

Amélioration des résultats:
- nettoyage des requêtes transmises
- interdiction des requêtes qui modifient la base (insert, update, delete, drop, alter, create)
- nettoyage des résultats récupérés
- bien récupérer toutes les réponses retournées notamment lors de comparaisons
- on ne transmet pas le contexte tel que il faut en extraire les éléments pour augmenter Faithfulness
