# Archiv (Stand 03.11.2026)

Eingefrorener Code, der für den aktuellen Pfad der Arbeit (`scripts/compare`) nicht mehr gebraucht wird.
Nichts davon wurde inhaltlich geändert, nur verschoben (`git mv`, die Historie bleibt erhalten). Die
Archivinhalte sind **nicht an ihrem neuen Ort lauffähig**: Pfade wie `ROOT / "GDesigner/..."` setzen die
alte Struktur voraus. Zum Ausführen den Tag auschecken:

```bash
git checkout pre-cleanup-2026-11-03   # ursprüngliche Struktur, alle archivierten Tests lauffähig
```

Stand der archivierten Tests vor dem Verschieben (alle am Tag `pre-cleanup-2026-11-03` bestanden): `tests/` 14,
`scripts/followup` 9, `diagnose.py`/`extended.py` 11, `scripts/stage0` 12 (218 s, drei End-to-End-Tests
brauchen je rund 65–80 s).

## `legacy-robustness-pipeline/`

Die frühe Pipeline auf dem Original-AnyMAC-Code (Experimente 1–4 zu gelernter Abhängigkeit von
markierten Agenten beim Router-Training, Qwen3). Sie wurde durch den korrigierten Nachbau und den
einheitlichen Vergleichsrahmen in `scripts/compare/` abgelöst und liefert vorläufige Ergebnisse.

| Pfad | Zweck |
|---|---|
| `robust/` | Pipeline, Training, Rollouts, Statistik (Exp. 1–4) |
| `GDesigner/` | **Unsere modifizierte Kopie** des AnyMAC-Basiscodes (Hooks für `robust/control.py`, gekürzte Prompts). Der unveränderte Originalcode liegt im Upstream-Klon `AnyMAC`. Enthält die dokumentierten Router-Fehler (Rollen-Index, Gradientenmaske) |
| `tests/` | Tests der Pipeline und der Router-Gradienten |
| `configs/profiles/` | Smoke-/Pilot-/Full-Profile der Pipeline |
| `scripts/followup/`, `scripts/stage0/` | Folgetests, Positivkontrolle und Rauschmessung |
| `scripts/diagnostics/` | `diagnose.py`, `extended.py`, `run.sh`, README: Reproduzierbarkeitsdiagnosen |
| `scripts/ramses/` | `prepare.sh`, `run.sh`, `submit.sh` der alten Pipeline |

## `upstream-paper-material/`

Unverändertes Material der AnyMAC-Autoren aus dem Initial-Upload. Von keiner Datei der aktuellen Linie
referenziert.

| Pfad | Inhalt |
|---|---|
| `experiments/` | Trainings- und Auswertungsskripte der Autoren (inkl. `run_gsm8k_bk.py`, eine **eigenständige ältere Fassung**, kein Duplikat) |
| `datasets_my/` | Datenlader und kleine Datensätze der Autoren |
| `temp0_qwen/` | Ollama-Modelfile (Temperatur 0) |
| `AnyMAC.yml`, `requirements.txt` | Umgebungsexporte der Autoren (aktuell: `requirements-ramses.txt`) |
| `README.upstream.md` | Original-README |

## Gelöscht

`result/`: 39 GPT-4-Logdateien der Autoren (Mai 2025), nicht aus unseren Läufen. Wiederherstellbar am Tag
`pre-cleanup-2026-11-03` und aus dem Upstream-Repository.

## Pfad-Zuordnung für historische Dokumente

`docs/PROTOCOL.md`, `docs/AUDIT.md` und `docs/RAMSES.md` nennen die alten Pfade. Zuordnung: `robust/`,
`GDesigner/`, `tests/`, `configs/profiles/`, `scripts/followup/`, `scripts/stage0/` →
`archive/legacy-robustness-pipeline/<alter Pfad>`; `experiments/`, `datasets_my/` →
`archive/upstream-paper-material/<alter Pfad>`.
