# Audit der AnyMAC-Implementierung (für die Experimente)

| Frage | Befund |
|---|---|
| Werden NAP/NCS trainiert? | Ja. Router-Parameter (`routing_transformer`, NAP-/NCS-Köpfe, Rollen-/Aufgaben-Projektionen) sind im AdamW-Optimizer; Smoke prüft Gradientennorm > 0 und Parameteränderung. Agenten-LLM ist ein externer Ollama-Server und kein Teil des Optimizers. |
| Was sieht der Entscheidungsagent? | Aufgabe plus die per NCS-Maske gewählten Hints früherer Agenten ("Agent N: …", Rollen anonymisiert). Nicht gewählte Outputs sind **nicht** im Prompt: direkte NCS-Unterdrückung ist möglich und wird in `smoke` (ncs_controls_decision_context) und im Rollout-Log (`delivered`, `decision_context`) geprüft. |
| Gradientenpfad | Sampling (cos_scaling 1.5, Gumbel-Softmax hard) unter `no_grad`; Gradient wird in `arun_next_agent_prediction_grad` über die aufgezeichnete Trajektorie neu berechnet. Die Schleife läuft über `trace_length-1` Schritte: Die NAP-Wahl „Entscheider“ (letzter Schritt) taucht im Loss nicht auf. Ob die Hint-Maske der letzten aufgezeichneten Auswahl einen Gradienten erhält, wird nicht angenommen, sondern pro Smoke-Lauf empirisch gemessen (`decision_step_gradient`; in lokalen Läufen änderte ein Flip der letzten Maske den Gradienten). Gradienten für NAP und NCS einzeln: `tests/test_router_gradients_synthetic.py` (SYNTHETISCH, kein Forschungsergebnis). Die Implementierung wurde nicht verändert. |
| Reward | `decay_factor**Routinglänge · korrekt`, Advantage pro Aufgabe normiert, Gruppen ohne Variation verworfen. |
| Evaluationsdeterminismus | cos_scaling 1e3 → Router entscheidet deterministisch; „frische Verläufe“ sind bei Temperatur 0 identisch. |
| Umgehungspfade | Letzter erlaubter Schritt (`max_routing`) erzwingt den Entscheider; `max_context` blendet ältere Hints aus; beides ist Teil der „Zustellung“. |
| Eingriffe am Original | Additive, standardmäßig inaktive Controller-Hooks in `GDesigner/graph/graph.py` (ohne Controller identisches Verhalten); `ipdb.set_trace()` im Backward-Except durch `raise` ersetzt; konfigurierbare Ollama-Optionen/Usage-Zählung in `sllm_chat.py`; SentenceTransformer einmal geladen. |
| Lokale Änderungen aus `AnyMAC/` | Keine übernommen (nicht benötigt). Der ursprüngliche Ordner blieb unverändert. |

Sicherheitshinweis: Die Datei `run_llm_benchmark.py` im Original-Ordner `AnyMAC/` enthält einen fest codierten OpenAI-Schlüssel (`sk-proj-…`). Er wurde nicht übernommen; bitte widerrufen/rotieren.
