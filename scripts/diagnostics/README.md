# Auswertungen abgeschlossener Läufe

Schreibgeschützt: Eingaben werden nie verändert, Ausgaben gehen in ein **neues** Verzeichnis außerhalb des
Eingabebaums. Diese Skripte liegen bewusst außerhalb von `scripts/compare/`, damit sie die Quelltext-Hashes
der Läufe nicht ändern. Sie laden Runner und Scoring über `--code` aus dem Worktree, der den Lauf erzeugt hat
(`compare_results.api`). Der Quelltext des Laufs muss mit dessen `plan.json` übereinstimmen.

| Skript | Zweck |
|---|---|
| `compare_results.py` | Gemeinsame Hilfen (`api`, `digest`, `interval`, `mean`) und Auswertung abgeschlossener Vergleichsläufe |
| `role_usage.py` | Welche Worker-Rollen ruft jedes System auf, in welcher Position, wie oft die angegriffene Rolle |
| `pilot_worker_audit.py` | Worker-Antworten und deren Pfad zur Endantwort (direkte/transitive Vorfahren) für ausgewählte Aufgaben |
| `main_sensitivity.py` | Hauptlauf: Effekte je Seed mit/ohne Aufgaben mit Backend-Fehler, Exposition und kompromittierte Kopien im Aggregator-Request |
| `main_deep_dive.py` | Hauptlauf vertieft: Kosten, gepaarte ehrliche Genauigkeit, Rauschniveau aus No-op-Zellen, Effekte je Checkpoint, Strata nach ehrlicher Korrektheit mit Übergängen, DyLAN-Konsens/Ranker |
| `cooperation.py` (+ `.sh`) | Ehrliche Kooperationsdiagnose mit AnyMAC-Prompt, kein Router-Training |
| `graph_probe.py` | CPU-Wahrscheinlichkeitssonde für G-Designer aus dem eingefrorenen Trainings-Worktree |
| `reserve_balanced_main.py` | Reserviert eine themenbalancierte Teststichprobe (Standard: 200 Fragen) |

Beispiel (Ramses, ohne GPU):

```bash
python scripts/diagnostics/role_usage.py --code "$HOME/<Lauf-Worktree>" \
  --input "$RESULTS_DIR/compare/<ausgabe>/pilot" --out "$RESULTS_DIR/analysis/<neuer-ordner>"
```

Die früheren Reproduzierbarkeitsdiagnosen (`diagnose.py`, `extended.py`) liegen unter
`archive/legacy-robustness-pipeline/scripts/diagnostics/`.
