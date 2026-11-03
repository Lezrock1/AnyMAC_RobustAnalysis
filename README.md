# AnyMAC_RobustAnalysis

Masterarbeitscode: Vergleich dynamischer Multi-Agent-Router (AnyMAC normal/efficient, DyLAN, feste
Referenz; G-Designer vorbereitet, im ersten Hauptlauf zurückgestellt) auf Abhängigkeit von
kompromittierten Agenten. Alle Systeme laufen mit demselben lokalen Modell (Gemma 4 12B IT QAT über
Ollama) auf dem Ramses-Cluster (A30). Fork von [SongW-SW/AnyMAC](https://github.com/SongW-SW/AnyMAC).

## Aufbau

| Pfad | Inhalt |
|---|---|
| [`scripts/compare/`](scripts/compare/README.md) | Vergleichsrahmen: Event-Contract, Runner, Training der gelernten Router, Suite (Pilot/Hauptlauf), Angriffe, Scoring. **Quelltext-Hashes binden die Läufe: `*.py` (ohne `test_*`) nicht ändern.** |
| [`scripts/screen/`](scripts/screen/README.md) | Ehrliches Modell-×-Benchmark-Screening und Datenvorbereitung (MMLU-Pro) |
| [`scripts/diagnostics/`](scripts/diagnostics/README.md) | Schreibgeschützte Auswertungen abgeschlossener Läufe |
| [`scripts/training_attack/`](scripts/training_attack/README.md) | Explorative, gepaarte Kurztrainingsstudie mit harter 24-h-Allokationsgrenze |
| `scripts/upstream/` | Minimale Reproduktion der AnyMAC-Router-Indexfehler |
| `scripts/ramses/` | Cluster-Anbindung (`lib.sh`, `setup.sh`, `doctor.sh`) |
| [`docs/`](docs/COMPARATIVE_PROTOCOL.md) | Protokolle, Abweichungen, Audit (`COMPARATIVE_PROTOCOL.md` ist maßgeblich) |
| [`archive/`](archive/README.md) | Frühe Robustheits-Pipeline und Original-Material der AnyMAC-Autoren, eingefroren |

## Tests

```bash
.venv/bin/python -m pytest scripts/compare scripts/diagnostics scripts/screen scripts/training_attack -q
```

`pytest.ini` schließt `archive/` aus. Archivierte Tests laufen nur am Tag `pre-cleanup-2026-11-03`.

## Git-Marken

- `eval-core-targeted-main-ceb6601`: Code-Stand, der den Hauptlauf `core_four_gemma12b_paper_v2_routing_targeted` ausgeführt hat.
- `pre-cleanup-2026-11-03`: Zustand vor dem Aufräumen (enthält die ursprüngliche Verzeichnisstruktur).

## Zitation

Originalarbeit: Wang et al., *AnyMAC: Cascading Flexible Multi-Agent Collaboration via Next-Agent Prediction*, EMNLP 2025.
