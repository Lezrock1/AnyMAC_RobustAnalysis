# Robustness-Analyse auf Ramses (Slurm, A30)

Experimente: 1 gelernte Abhängigkeit, 2 Ursache, 3 Beständigkeit, 4 Verteidigung. Definitionen: `PROTOCOL.md`, Audit: `AUDIT.md`.
Nur Router (NAP/NCS) wird trainiert; Agenten-LLM (Ollama, Qwen3-4B-Instruct-2507) bleibt eingefroren.

## Befehle (Login-Knoten)
```bash
git clone https://github.com/Lezrock1/AnyMAC_RobustAnalysis.git && cd AnyMAC_RobustAnalysis
git checkout robust-analysis
cp configs/ramses.env.example configs/ramses.env   # ausfüllen (Partition, GRES, Account, Module, Pfade)
scripts/ramses/doctor.sh                           # Werte prüfen (Slurm, GPU, Verzeichnisse)
scripts/ramses/setup.sh                            # venv + torch 2.3.0 + Ollama (User-Space)
scripts/ramses/prepare.sh --profile pilot          # Modell + MASBench breadth (mit Internet, Login-Knoten)
scripts/ramses/submit.sh --profile smoke --experiment all --dry-run
scripts/ramses/submit.sh --profile smoke --experiment all    # echter Smoke (<=15 min)
scripts/ramses/submit.sh --profile pilot --experiment all    # nur wenn Smoke bestanden
scripts/ramses/submit.sh --profile full  --experiment all    # >=3 Seeds, ohne Zeitlimit
scripts/ramses/submit.sh --profile full  --experiment all --resume   # nach Abbruch/Timeout
```
Gemma-3-4B-IT (Full, Exp. 3) benötigt `prepare.sh --profile full`.

## Ablauf pro Seed
`llm_limit` (Ausgabelimit auf Dev kalibrieren, danach für alle Arme gleich) → `calibrate` → `signal_check`
(Reward-Varianz auf MASBench; Abbruch bei „insufficient“, Override `ALLOW_LOW_SIGNAL=1`, dann im Report markiert) → Warm-up → Zweige A/B/C → Exp 1-4 → Report.

## Ergebnisse / Logs
- Ergebnisse: `$RAMSES_RESULTS_DIR/<profil>/` (`report.md`, `report.json`, `budget.json`, `<modell>/seed<N>/…`, `llm_limit.json`, `signal_check.json`)
- Smoke: `<results>/smoke/smoke/smoke_result.json`
- Logs: `$RAMSES_LOG_DIR` (Slurm-out/err, Ollama-Log, GPU-Monitor-CSV)

## Zwei getrennte Prüfungen
1. **Synthetisch (lokal, kein Forschungsergebnis):** `python -m pytest tests` – Gradientenpfad NAP/NCS mit gestubbtem LLM, zufälligem Reward; Grad-Modus nach parallelen Rollouts inkl. Fehler/Timeout.
2. **Echter Smoke (nur Ramses):** Qwen3-4B-Instruct-2507, einfache `smoke-*`-Aufgaben außerhalb der Splits, echter Router-Schritt, Checkpoint, LLM-Digest.

## Interpretation
- Läuft kein Update (alle übersprungen), ist der Lauf „PIPELINE TEST ONLY“: keine Aussagen zu Angriff, Persistenz, Abwehr.
- Ein Trainingsseed im Pilot → Aussagen nur über diese Checkpoints. Kein Signal im Pilot ≠ Robustheitsnachweis.
- Laufzeiten im Report sind gemessen; `projection_NOT_MEASURED` ist Hochrechnung.
- Offen bis Ramses: CUDA/Slurm, A30-Durchsatz, Ollama-Binary im User-Space, Qwen3-Trunkierung (Limit), echter Lernerfolg auf MASBench, Gemma-Tag.
