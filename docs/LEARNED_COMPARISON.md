# Gelernte Router im Gemma-Vergleich

Stand 26.10.2026, laut uebermittelten Ramses-Ausgaben. Historischer AnyMAC-Seed 0
mit Kostenfaktor 0,95 trainiert, aber Kooperations-Gate nicht bestanden.
G-Designer-Seed 0 und neues AnyMAC-normal Seed 0 bestehen das Dev-Gate.
Alle neun sauberen Trainings sind abgeschlossen. Normal besteht fuer Seeds 0/1/2
die urspruenglichen Gates; Efficient fuer keinen Seed; G-Designer nur fuer Seed 0.
Alle Fehlschlaege betreffen ausschliesslich `nontrivial_cooperation`; alle neun
erreichen 10/12 korrekte Dev-Antworten. Die ausdrueckliche
[Zulassungsaenderung](./COMPARATIVE_PROTOCOL.md#zulassungsaenderung-nach-dev-auswertung-26102026)
trennt solche Routingbefunde von technischer/qualitativer Freigabe. Die CPU-Probes
reproduzieren alle aufgezeichneten Dev-Graphen. Der Angriffspilot ist als Job
6987782 gestartet; ein Abschlussbericht liegt noch nicht vor. Der Hauptlauf
ist nicht gestartet. Die bisherigen AnyMAC-Trainings sind vorlaeufige
Kleinbudgetbedingungen, kein belegter ausreichender Trainingsumfang.
Trainingsinterventionen sind lokal getestet, noch nicht auf Ramses empirisch
validiert. Lokale Tests ersetzen diese Validierung nicht.

Nutzerentscheidung: korrigiertes AnyMAC neu trainieren; G-Designer eigenstaendig
anhand des Papers rekonstruieren. Historische Qwen-Checkpoints bleiben unveraendert.
Die [DyLAN-Vertiefung](./COMPARATIVE_PROTOCOL.md#e1b-dylan-abhaengigkeitsstudie-vorregistriert-2026-10-21-vor-jedem-angriffslauf)
ersetzt nicht den Vier-System-Vergleich.

## 1. Methodenidentitaet und Grenzen

### `anymac-corrected`

Die Implementierung verwendet die vorhandene AnyMAC-Architektur:
vortrainiertes GPT-2, fuenf Projektionen (Aufgabe, Rolle, Historie, NAP, NHP)
und finale LayerNorm. Eingabereihenfolge: Aufgabe, Rollen einschliesslich
Entscheider, Historie, NAP, NHP. Prefix-Positionen sind wie im lokalen Original null.

- Rollenscores beginnen bei Token 1, nicht beim Aufgabentoken 0.
- Historienauswahl beginnt nach Aufgabe **und** Rollen.
- REINFORCE beruecksichtigt die terminale Kontextentscheidung; eine am Horizont
  erzwungene Entscheiderwahl traegt keine kuenstliche Agenten-Logwahrscheinlichkeit bei.
- Alle Routerparameter einschliesslich Projektionen werden optimiert.
- Evaluation: Argmax der Rolle und Schwelle null fuer die Hint-Logits.
- Training: kategoriale Rollenauswahl und Bernoulli-Hints bei Skalierung 1,5.
  Dies entspricht den diskreten Aktionsverteilungen der bisherigen Gumbel-Auswahl.
- GPT-2-Dropout bleibt auch bei der Likelihood-Rekonstruktion deaktiviert.
  Rollout und Gradientenberechnung verwenden dadurch dieselbe Policy.

Das ist eine **korrigierte Reimplementierung**, keine unveraenderte Reproduktion
der historischen Pipeline. Neun explizite Rollen und gemeinsame kurze Antwortprompts
ersetzen historische MASBench-Rollen/Marker; die MMLU-Pro-Optionenzahl wird dynamisch
beruecksichtigt. Rollenembeddings werden aus `You are <role>.` berechnet.
Die Architektur ist nicht mit einem Nachweis sinnvoll gelernter Auswahl gleichzusetzen.

#### Korrektur der Trainingsbedingungen (23.10.2026)

Der erste Vergleichslauf verwendete gamma=0,95, Kontextfenster 10 und lambda=0.
Das war weder normales AnyMAC noch AnyMAC-Eff. Er bleibt eine historische
Diagnosebedingung und darf nicht entsprechend umbenannt werden.

Neue Profile (Architektur-/Trace-ID bleibt `anymac-corrected`, Suite-Labels getrennt):

| Profil | gamma | lambda | neueste zur Zustellung waehlbare Antworten |
|---|---:|---:|---:|
| `anymac-normal` | 1,0 | 0 | 5 |
| `anymac-efficient` | 0,9 | 0,001 | 2 |
| `anymac-historical-095` | 0,95 | 0 | 10 |

Efficient entspricht den vom Nutzer zitierten Paperparametern und lokalen
Eff-MMLU-Defaults. Das normale Fenster 5 stammt aus `experiments/run_mmlu.py`.
Dies bleibt eine korrigierte Reimplementierung, keine vollstaendige Paperreproduktion.

Lambda multipliziert die **Summe vorzeichenbehafteter, unskalierter
Kosinus-Hintscores** ueber die gesamte bisherige Historie je Entscheidung,
einschliesslich inzwischen nicht mehr zustellbarer alter Antworten. Das entspricht
dem Sparsitaetsterm in `GDesigner/graph/graph.py`, unter korrigierten Tokenoffsets.
Keine L0/L1-Strafe auf ausgewaehlte Nachrichten, keine Entropiestrafe.
Da `scores()` doppelte Kosinuswerte liefert, wird fuer diesen Term durch 2 geteilt.
Der Term wird getrennt vom Utility-Gradienten berechnet: reine
Regularisierungsupdates bestehen weiterhin kein Utility-Gate.

Ein Kontextfenster beschraenkt die **Zustellung**, nicht die Historie, die der
Router zur Auswahl sieht. Profile werden in Konfiguration und Manifest gebunden.
Gleiche Seeds initialisieren dieselben Gewichte; der initiale State-Dict-Hash wird
gespeichert. Nach unterschiedlichen Aktionen divergieren Trajektorien; gleiche
Seeds garantieren keine identischen Nachrichten oder identischen Ressourcenverbrauch.
Historische Checkpoints werden nicht konvertiert.

#### Erste Profilbefunde (Seed 0, zwoelf bekannte Dev-Aufgaben)

Alle drei gelernten Bedingungen erreichen vor und nach Training 10/12 korrekte
Antworten. Fuer beide AnyMAC-Profile bleiben sogar alle zwoelf geparsten
Endantworten unveraendert. Normales AnyMAC reduziert die mittlere Aufrufzahl von
8,5 auf 4 und verwendet nach Training ausschliesslich Historian: 36 Aufrufe
ueber zwoelf Aufgaben, Worker in jeder Episode. Das belegt Auswahlkonzentration,
noch keine inhaltliche Abhaengigkeit. Ob jede Aufgabe genau drei Aufrufe hat und
welche Nachrichten den Decider erreichen, muss aus Einzeltraces bestimmt werden.
Efficient reduziert von 8,5 auf einen Aufruf und verwendet auf diesen Aufgaben
keine Worker. Sein Slurm-Fehlerstatus stammt vom absichtlich fehlschlagenden
Dev-Gate nach abgeschlossenem Training, nicht von einem Trainingsabbruch.
G-Designer bleibt bei 30 Aufrufen; 3/40 Trainingsaufgaben haben Utility-Updates,
gegenueber 7/40 bei Normal und 16/40 bei Efficient.

Ein Profilvergleich variiert gamma, lambda und Kontextfenster gemeinsam:
keine isolierte Kostenfaktor-Ablation. Ein Seed und zwoelf bekannte Aufgaben
belegen weder allgemeine Worker-Nutzlosigkeit noch allgemeine Angriffsfreiheit.
Alle Seeds einschliesslich Gate-Fehlschlaegen bleiben im Bericht; keine
Ersetzung durch einen nachtraeglich ausgewaehlten kooperativeren Seed.
Die Strict-Hauptsuite bleibt fuer nicht bestandene Checkpoints gesperrt.
Seit 26.10. gilt zusaetzlich die oben verlinkte ausdrueckliche Zulassungsaenderung,
nicht eine stille Abschwaechung oder Neuberechnung des urspruenglichen Gates.

Die schreibgeschuetzte Auswertung ausserhalb des Trainingsquellverzeichnisses
trennt pro Rolle Router-Sichtbarkeit, LLM-Zustellung und strukturelle Pfade zur
Endantwort. Legacy-Expositionswerte beziehen sich nur auf die im Trace benannte
kompromittierte Rolle; null ist keine Aussage ueber Zustellung anderer Rollen.
`initial_distribution_max_task_range` im Gate bezeichnet die Verteilung bei
**leerer Historie des ausgewerteten Routers**, nicht zwingend die untrainierte
Policy. Variation dieser Verteilung ist mit identischem Argmax ueber alle
Aufgaben vereinbar und kein Beleg fuer kompetenzbasierte Auswahl.

### `gdesigner-paper`

Primaere Spezifikation: [G-Designer, arXiv:2410.11782v2](https://arxiv.org/html/2410.11782v2),
Abschnitte 3.2 und 4, Algorithmus 1. Kein Codeimport aus dem ungelizenzierten
G-Designer-Repository. Der lokale AnyMAC-Namensraum `GDesigner/` wird ebenfalls
nicht fuer diese Implementierung verwendet.

Enthalten sind:

1. MiniLM-Rollenembeddings und ein virtueller Aufgabenknoten.
2. Kettenanker, bidirektionale Verbindungen zum Aufgabenknoten und normalisierte
   GCN-Nachbarschaft mit Selbstkanten. Fuer den Encoder wird der Kettenanker
   symmetrisiert; der Kommunikationsanker bleibt gerichtet.
3. Getrennte zweilagige GCNs fuer Mittelwert und Log-Standardabweichung;
   gaussche latente Variablen via Reparametrisierung.
4. FFN aus Quell-, Ziel- und Aufgabenlatenz; Concrete-Skizze mit Temperatur 0,01.
5. Low-Rank-Verfeinerung im Unterraum der fuehrenden linken Singulaervektoren;
   Anker-/Rekonstruktionsregularisierung und Nuklearnorm mit Gewicht 0,1.
6. Drei Kommunikationsrunden und je ein ehrlicher Summarizer pro Runde.
7. Routertraining mit zehn Graphsamples pro Trainingsaufgabe.

**Notwendige Auslegungen, keine stillschweigenden Korrekturen:**

- Gl. 14 schreibt `argmax`, der Text fordert Minimierung positiver
  quadratischer Kosten. Implementiert ist die begruendete **Minimierungslesart**.
  Im fuer die innere Optimierung festen Unterraum wird die geschlossene
  Singulaerwert-Schrumpfung mit Schwelle `zeta/2` verwendet. Der Gradient durch die
  Wahl des Unterraums selbst wird nicht zurueckpropagiert.
- Ein niedriger Rang erzeugt keine garantierten entry-wise Nullkanten.
  Die verfeinerte Matrix wird explizit auf Wahrscheinlichkeiten begrenzt und
  eine Bernoulli-Kommunikationsstruktur gezogen. Wahrscheinlichkeiten erhalten
  einen numerischen Abstand von `1e-6` zu 0/1.
- Eine feste Vorwaertsordnung ueber die Rollen definiert zulaessige Kanten
  und garantiert einen DAG. Der Kettenanker ist eine Regularisierung, nicht
  die gesamte Menge zulaessiger Kommunikationskanten. Die Beziehung zwischen
  Anker und zulaessigem Kantenraum in Gl. 15 ist im Paper nicht eindeutig.
- Die Policy-Likelihood beruecksichtigt gezogene **und nicht gezogene** Kanten,
  statt nur ausgewaehlte Wahrscheinlichkeiten zu multiplizieren. Gaussian- und
  Concrete-Zufall werden bei der Rekonstruktion wiederverwendet.
- Fuer nicht festgelegte Dimensionen werden 32 Hidden- und 16 Latent-Dimensionen
  sowie Rang 4 gewaehlt; Log-Standardabweichungen liegen aus numerischen Gruenden
  in [-8, 4]. Kein zusaetzlicher KL-Term wird erfunden.

Folglich ist dies eine **Paper-abgeleitete, qualifizierte Rekonstruktion mit
offenen Reproduktionsabweichungen**, kein verifiziertes Originalverfahren.
Weder lokale Tests noch das Dev-Gate beweisen die Reproduktion der Paperergebnisse.
In einer Arbeit stets mit diesem Zusatz ausweisen. Die urspruengliche Robustheitszahl
des Papers wird nicht als reproduziert bezeichnet.

Nach Gl. 4 erhalten Worker Nachrichten ihrer Vorgaenger derselben Runde;
es gibt keinen zusaetzlich erfundenen Cross-Round-Speicher. Alle neun Rollen
werden ausgefuehrt, auch isolierte. Der Summarizer sieht alle neun Antworten.
Damit liegt seine direkte Exposition bei jedem erfolgreichen Angriffslauf bei eins:
dieser Pfad untersucht indirekte Einflussverstaerkung und Graphkosten, nicht
Abschirmung des Summarizers. Drei Runden ergeben **30 LLM-Aufrufe**.

## 2. Daten, Training und Artefakte

Die folgenden 40-Fragen-Parameter beschreiben die **vorlaeufige Bedingung**
`preliminary-v1`. Fuer das neue AnyMAC-Referenztraining gilt Abschnitt 2a.

- Gemma4 12B mit dem gescreenten Digest, Ollama 0.40.6, Temperatur 0,
  Outputlimit 8192, Kontext 32768. Workergewichte bleiben unveraendert.
- 40 Trainingsaufgaben aus dem offiziellen **Validation-Split** von MMLU-Pro
  (Seed 20261023, Datenrevision `b189ec765aa7ed75c8acfea42df31fdae71f97be`);
  keine Testaufgaben fuer Routertraining. Validation-IDs erhalten einen eigenen
  Namensraum; zusaetzlich werden normalisierte Textduplikate ausgeschlossen.
  Ausgeschlossen sind beide
  Screening-Saetze, die 12 Preflight-Dev-Aufgaben und die 60 vorab reservierten
  Evaluationsaufgaben. IDs und Dateihashes werden gebunden. Reichen die geeigneten
  Validation-Aufgaben nicht aus, erfolgt ein expliziter Stopp, kein Testset-Fallback.
- Die 12 Preflight-Aufgaben bleiben Entwicklung: sie dienen Vorher/Nachher-
  Vergleich und Freigaben, niemals der Hauptauswertung.
- Eine Trainingspassage pro Seed. AnyMAC: vier Trajektorien, Lernrate `1e-5`,
  Reward `korrekt * gamma^Workeranzahl` mit explizitem Profil wie oben.
  G-Designer: zehn Graphsamples,
  Lernrate `1e-3`, Korrektheitsreward plus die expliziten Paper-Regularisierungen.
- Leave-one-out-Baseline pro Aufgabe; kein Skalieren durch die Reward-Streuung.
  AdamW ohne Weight Decay, Gradient-Clipping bei Norm 1.
- Unterschiedliche Trainingsumfaenge sind methodenbezogene Entscheidungen,
  keine Behauptung gleicher Trainingskosten.
- Seeds 0, 1, 2 werden fuer den Hauptvergleich separat trainiert.
  Zunaechst nur Seed 0 und den Pilot ausfuehren.

Im **sauberen Referenztraining** wird die richtige Loesung erst nach einer
abgeschlossenen Trajektorie fuer Reward/Scoring gelesen. Der getrennte
Trainingsinterventionsstrang unten besitzt ausdruecklich weitergehenden Zugriff.
Es gibt keine Gewichtsupdates waehrend der Evaluation.
Budget-/Backend-/Kontrollfehler im Training stoppen explizit.

Snapshots sind fest gepinnt:

- MiniLM: `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`
- GPT-2: `607a30d783dfa663caf39e06633721c8d4cfcd7e`

Vorbereitung laedt nur passende Gewichts-/Konfigurationsdateien, keine ganzen
Modell-Repositories mit ONNX/OpenVINO/Duplikatgewichten. Laufzeit ist offline.
Checkpoint, Embedding-Inhalt, Workeridentitaet/Decoding/Serving-Version,
Implementierung und Bibliotheksversionen sind gebunden. CPU-Threadzahl ist
Ausfuehrungsmetadatum und keine Decoding-Aenderung.

**Speicher:** genau ein rollierender Resume-Checkpoint pro Seed mit Optimierer,
danach ein finales Tensorartefakt. Keine grossen Checkpoints pro Aufgabe.
Das Resume-Artefakt wird nach dauerhaft geschriebenem finalem Modell samt Metadaten
entfernt. Beim atomaren Schreiben existieren zeitweise Alt- und Neufassung.
GPT-2 plus Adam-Zustaende benoetigen dennoch mehrere GB; persoenliche Quota vorher
pruefen. Smoke-Checkpoints sind technisch von Studienartefakten getrennt und
koennen kein Hauptstudien-Gate bestehen.

## 2a. Papernahes AnyMAC-Training nach Budgetpruefung

Explizite Nutzerentscheidung nach den ersten Trainings und vor dem Hauptlauf:
**Validation-only**, kein Auffuellen mit Testfragen. Das
[AnyMAC-Paper, Abschnitt 5.1](https://arxiv.org/html/2506.17784v2) nennt
80 unterschiedliche Fragen und 1000 Trajektorien **pro Datensatz**, nicht
einen separaten Router pro MMLU-Themenbereich. Die gebundene
[MMLU-Pro-Datenrevision](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/blob/b189ec765aa7ed75c8acfea42df31fdae71f97be/README.md)
hat jedoch insgesamt nur 70 Validation-Fragen. Die bisherigen 40 x 4 =
160 AnyMAC-Trajektorien waren nur 16 % des Paperbudgets.

Erste separat gespeicherte Bedingung `paper-aligned-v1`
(wegen Optionsfilter durch v2 unten abgeloest):

| Parameter | AnyMAC normal und efficient | G-Designer |
|---|---|---|
| Unterschiedliche Fragen | alle geeigneten Validation-Fragen, hoechstens 70 statt Paper-80 | vorhandene 40 |
| Trainingsbudget je Seed | genau 1000 Trajektorien | vorhandene 40 x 10 = 400 Graphsamples |
| Samples je Update | 8 unabhaengige Trajektorien | 10 Graphsamples |
| Aufrufhorizont | hoechstens 5 Worker, danach Decider | unveraendert 3 Runden, 30 LLM-Aufrufe |
| Lernrate | unveraendert `1e-5` | unveraendert `1e-3` |
| Seeds | 0, 1, 2 je Profil | alle drei vorhandenen Seeds beibehalten |

Jede AnyMAC-Passage mischt die Aufgaben reproduzierbar mit dem Trainingsseed.
Alle Aufgaben kommen einmal vor, bevor eine weitere Passage beginnt.
1000 / 8 ergeben 125 Updateslots, nicht zwingend 125 nichtnull Gradienten.
Normal/Efficient haben pro Seed denselben Aufgabenplan und dieselben initialen
Gewichte. Die Profile gamma/lambda/Kontextfenster bleiben wie oben.
Die Vorbereitung protokolliert Themenhaeufigkeiten und stoppt, falls Ausschluesse
einen ganzen verfuegbaren Themenbereich entfernen. Sie prueft IDs und normalisierte
Texte gegen Screening, Dev, DyLAN-Evaluation und beide reservierten Hauptsamples.
Es gibt keinen Testset-Fallback und keine separaten Modelle pro Themenbereich.

**Bewusste verbleibende Trainingsabweichungen:** Das AnyMAC-Paper verwendet
adaptives Sampling bis zu einer bzw. vier korrekten Trajektorien. Hier bleiben
feste unabhaengige Achtergruppen und die Leave-one-out-Baseline erhalten.
Erfolgsabhaengiges Stoppen wuerde deren Unabhaengigkeitsargument veraendern;
Einzelgruppen koennten zudem kein Leave-one-out bilden. Die im lokalen
Originalcode vorgesehenen zehn Wiederverwendungen gespeicherter Trajektorien
werden nicht unbesehen als neue On-Policy-Samples behandelt: keine solche
Off-Policy-Wiederverwendung in dieser Implementierung.
Dies ist eine deutlich budgetnaehere Reimplementierung, **keine exakte
Paperreproduktion**. MMLU-Pro statt MMLU, Gemma statt GPT-4, Decoding und
gemeinsame Rollen/Prompts bleiben dokumentierte Abweichungen.

Beim [G-Designer-Paper, Abschnitt 5.1.3](https://arxiv.org/html/2410.11782v2)
sind die expliziten Werte 40 Trainingsfragen, zehn Graphsamples und drei Runden
bereits umgesetzt. Daraus folgt keine Begruendung, auch dort 80 Fragen oder
1000 Samples zu erzwingen. Die vorhandenen drei sauberen G-Designer-Checkpoints
bleiben unter ihren urspruenglichen Manifesten erhalten, einschliesslich aller
Routing-Gatebefunde. Ihre sonstigen Rekonstruktionsgrenzen bleiben bestehen.
Die gemeinsame Runtime wird fuer diese Trainingsaenderung nicht modifiziert.

### Lernverlauf, Speicherung und neue Evaluation

- Dev-Diagnosen bei 200/400/600/800/1000 Trajektorien, mit gleichen
  Evaluationsseeds; Ausgabe `learning_curve.json` und separate Dev-Traces.
  Der untrainierte Vergleich und die regulaere finale Dev-Freigabe bleiben.
  Kein Best-Dev-Checkpoint, kein Early Stopping anhand von Angriffserfolgen.
  Zwoelf bekannte Dev-Fragen belegen auch mit Lernkurve keine Konvergenz.
- Nur ein rollierendes Gewichts-/Optimiererartefakt; keine fuenf grossen
  Zwischenmodelle. Resume bindet Budget, Plan, Daten, Worker und Quellcode;
  abgebrochene Gruppen werden vollstaendig wiederholt. Das **logische**
  Trainingsbudget ist exakt 1000; physisch ausgefuehrte Versuche koennen bei
  einem Abbruch mitten in einer noch nicht gespeicherten Gruppe hoeher liegen.
- Neue saubere Ausgaben unter `learned_gemma12b_paper_v1`, Smoke getrennt darunter.
  Trust-Bedingungen erhalten `trust_gemma12b_paper_v1` und benoetigen eine neue
  zur groesseren Trainingsmenge passende Bank. Alte Banken/Checkpoints werden
  nicht stillschweigend uebernommen oder konvertiert.
- Die neue Suite bindet fuer AnyMAC den neuen Pool und genau 1000 abgeschlossene
  Trajektorien. G-Designer bindet weiterhin seinen urspruenglichen 40-Fragen-Pool.
  Ausgabe `five_condition_gemma12b_paper_v1` bzw. mit ausdruecklicher
  Routingzulassung Suffix `_routing`. Der alte Pilot bleibt unveraendert;
  fuer die neuen AnyMAC-Checkpoints ist ein neuer Pilot erforderlich.
- Die 200 reservierten Testfragen werden **nicht neu geschrieben oder gezogen**.
  Ihre urspruengliche Hashbindung an den alten Trainingspool bleibt erhalten.
  Der spaeter beschlossene groessere Validation-Pool wird zusaetzlich anhand
  von IDs und normalisierten Fragetexten auf Disjunktheit geprueft. Dies ist
  eine nachtraegliche Trainingsaenderung, keine urspruengliche Vorregistrierung.
  Ein neuer erfolgreicher Pilot und eine ausdrueckliche Kostenfreigabe bleiben
  fuer den Hauptlauf verpflichtend.

Die Cluster-Vorbereitung muss die tatsaechlich verbleibende Fragenzahl und
Themenverteilung noch bestaetigen. Zuerst Smoke fuer beide Profile, dann Seed 0
mit Laufzeit-/Lernverlaufsauswertung, erst danach Seeds 1/2. Keine automatische
Einreichung und kein automatischer Hauptlauf.

### Korrektur nach Pruefung der N/A-Optionen: `paper-aligned-v2`

Die Ramses-Vorbereitung von v1 ergab 49 Fragen. Die anschliessende direkte
Pruefung der gepinnten Parquet-Datei zeigt: **alle 21 weiteren Fragen** wurden
vom alten Optionsfilter verworfen. Sie enthalten insgesamt 51 `N/A`-Eintraege,
saemtlich am Ende der Optionslisten; kein Goldbuchstabe verweist darauf.
Die 49 ausgegebenen Fragen entsprechen bereits dem gesamten Optionsfilter-
Ergebnis; Reservierungspruefungen entfernten dort keine weitere Frage.
Die fruehere Erklaerung, die Reduktion liege an Evaluation-Ueberschneidungen,
war damit falsch.

Mit ausdruecklicher Nutzerfreigabe entfernt v2 ausschliesslich **nachgestellte
`N/A`-Eintraege**, nicht die ganze Frage. Echte Optionen und ihre offiziellen
Buchstaben bleiben unveraendert. Interne Platzhalter, leere echte Optionen,
weniger als zwei echte Optionen sowie ungueltige oder inkonsistente
Goldbuchstaben/Indizes erzeugen einen expliziten Fehler, kein stilles
Umnummerieren oder Verwerfen. Der
[offizielle MMLU-Pro-Evaluator](https://github.com/TIGER-AI-Lab/MMLU-Pro/blob/main/evaluate_from_local.py)
entfernt ebenfalls `N/A` vor dem Promptaufbau. Unsere Beschraenkung auf
nachgestellte Platzhalter verhindert zusaetzlich eine unbemerkte Verschiebung.

Lokal an echten Daten verifiziert: **70 Fragen, 14 Themen x 5**, alle bisherigen
49 geparsten Fragen unveraendert. Die neuen 21 muessen auf Ramses noch gegen
alle dortigen Reservierungen geprueft werden; erst dann steht die endgueltige
Trainingszahl fest. Die Vorbereitung schreibt eine `preparation_audit` mit
Rohzahl, bereinigten Fragen/Optionszahlen, eindeutigen Ausschluessen,
betroffenen IDs je Reservierungsdatei und Endzahl sowie die Themenverteilung.
Ein Ausschluss kann mehrere Dateien betreffen; die Einzelzahlen nicht addieren.

`COMPARE_TRAINING_REGIME=paper-aligned-v2` verwendet ausschliesslich neue Pfade:
`learned_gemma12b_paper_v2`, `trust_gemma12b_paper_v2` und
`five_condition_gemma12b_paper_v2` (ggf. `_routing`). Die restlichen
Trainingsparameter bleiben: genau 1000 Trajektorien, Achtergruppen,
hoechstens fuenf Workeraufrufe, Dev-Diagnosen alle 200 Trajektorien.
Die Suite prueft den zur Optionsverarbeitung passenden Trainingsregime-Namen.
Smoke und Training verwenden neue Serverports 16000 + 3*Seed (Efficient +1).

Historische Screening-/Reservierungsaufrufe behalten standardmaessig den alten
Loader-Modus; die Bereinigung ist ausdrueckliches Opt-in fuer die neue
Trainingsvorbereitung. **Keine alten Samples, Checkpoints, laufenden Jobs oder
die 200 Hauptfragen werden konvertiert/ueberschrieben.** Alte Worktrees muessen
gepinnt bleiben. Auch v1-Trainings lassen sich nicht als v2 fortsetzen.
G-Designer behaelt weiterhin seinen urspruenglichen 40-Fragen-Pool und alle
drei vorhandenen Checkpoints. Kein Neustart ohne neue Datenvorbereitung und
Smoke; kein automatischer Abbruch alter Jobs.

## 3. Entscheidungsschranken

### Vorarbeit nach dem abgeschlossenen Kleinbudget-Piloten

Pilot 6987782 ist laut Ramses `COMPLETED`, 06:42:13, Exitcode 0.
Alle 528 Episoden wurden mit dem Originalworktree validiert. Auf zwoelf
bekannten Dev-Fragen gibt es in keiner gepaarten Angriffsepisode einen Wechsel
des geparsten Endantwortbuchstabens. Fixed und G-Designer sind dabei tatsaechlich
substituiert und aggregator-exponiert; fehlende Exposition erklaert deren
Nullbefund nicht. `delivered_any` umfasst auch Kontrollkonsum und darf nicht
pauschal als LLM-Zustellung bezeichnet werden.

Alle ehrlichen Systeme/Seeds scheitern in beiden Wiederholungen an denselben
beiden Geschichtsfragen: 4745 (Gold E, alle B) und 4804 (Gold G, DyLAN F,
andere E). Auf den anderen zehn Fragen sind alle korrekt. Das belegt
gemeinsame Fehlermengen auf diesem kleinen Satz, nicht fehlendes Modellwissen
oder eine allgemeine Qualitaetsgrenze von MMLU-Pro.

Vorlaeufige Quellenpruefung, **keine Neubewertung der Labels**:

- Die [Progressive Party Platform 1912](https://www.presidency.ucsb.edu/documents/progressive-party-platform-1912)
  fordert im Abschnitt Tariff einerseits Senkung ueberhoehter Zoelle,
  verteidigt andererseits das Schutzprinzip und kritisiert die demokratische
  Revenue-only-Position. Das erlaubt keine einfache Gleichsetzung von
  Progressive Party und Demokraten. Die Quelle entscheidet die konkrete
  Underwood-Simmons-Multiple-Choice-Frage nicht abschliessend. Weder das
  offizielle E noch das gemeinsame B wird hier als historisch verifiziert
  behauptet.
- Der [Britannica-Eintrag zu Zamindars](https://www.britannica.com/topic/zamindar)
  beschreibt unterschiedliche regionale Formen von Landbesitz und
  Steuererhebung, belegt jedoch nicht den konkreten Kausalzusammenhang der
  Aufgabe mit religioeser Toleranz. Die Zweitfrage bleibt fachlich offen;
  keine alternative Antwort als gesichert einsetzen.

Die gezielte Workerpruefung liegt ausserhalb des gebundenen Vergleichscodes:
`scripts/diagnostics/pilot_worker_audit.py`, Aufruf im Vergleichs-README.
Sie unterscheidet Vorkommen einer offiziell richtigen Workerwahl,
direkte Zustellung an die finale Entscheidung und transitive Nachrichtenpfade.
Ungeparste Antworten bleiben unbekannt. Ein fehlender Goldbuchstabe beweist
kein fehlendes Wissen, ein vorhandener keine korrekte Begruendung.
Ein abgelegtes Ergebnis benoetigt noch die Ramses-Traces; lokal sind nur
synthetische Verhaltenstests moeglich. Nutzer hat vor dieser Vorarbeit noch
keine neuen Trainingsjobs eingereicht. Keine GPU-Jobs oder Hauptaufgaben
werden durch die Diagnose gestartet.

Die urspruenglichen, weiterhin unveraendert gespeicherten Gates verlangen:

1. mindestens ein endlicher, nichtnull Utility-Gradient und ein tatsaechliches
   Optimiererupdate beobachtet wurden; reine Regularisierungsupdates reichen nicht;
2. Trainingsrewards zwischen unabhaengigen Samples variierten;
3. ehrliche Dev-Laeufe keine Backend-/Kontroll-/Budgetfehler haben, hoechstens
   5 % ungueltige Endformate und 5 % finale Abschneidung;
4. Dev-Genauigkeit mindestens 0,30 betraegt und nicht mehr als 0,10 unter der
   beobachteten untrainierten Genauigkeit liegt;
5. Worker auf wenigstens einer Dev-Aufgabe verwendet werden und die initiale
   Routingverteilung bei einem Common-Noise-Probe ueber Aufgaben um mehr als
   `1e-6` variiert.

Diese Schranken erkennen unbrauchbare/collabierte Konfigurationen. Sie sind
**kein statistischer Lernfortschrittsnachweis** und kein Nachweis kompetenzbasierter
Auswahl. Vorher-/Nachherwerte, Gradienten und Einzelaufgaben bleiben gespeichert.
Keinen erfolglosen Router durch einen zufaelligen Graphen oder feste Reihenfolge
ersetzen und unter demselben Namen weiterlaufen lassen.

Seit 26.10.2026 kann Punkt 5 mit expliziter `routing-observation/2026-10-26`-Freigabe
als Ergebnis statt Ausschlussgrund behandelt werden; Punkte 1-4 bleiben zwingend.
Original-Gates bleiben erhalten. Strict-Modus bleibt Standard. Alle festgelegten
Seeds werden behalten, nicht durch erfolgreiche Nachtrainings ersetzt.

Urspruenglicher Strict-Pilot: fuenf Bedingungen x 12 Dev-Aufgaben x
ehrlich/`bare` x zwei Wiederholungen = **240 Pilotepisoden** mit Seed 0.
Geaenderter Pilot: alle drei Seeds der gelernten Bedingungen plus zwei Baselines,
also **528 Episoden**. Beide verwenden die gemeinsame Aufrufhuelle 32. Das ersetzt nicht
die einzelnen Trainings-Dev-Gates. Der Pilot liefert die empirische Kostenbasis.
Angriffsbedingte Formatfehler sind Messergebnisse, keine nachtraeglich zu
entfernenden Aufgaben. Die Freigabe beruht auf ehrlicher Ausgabequalitaet.

## 4. Bedingter Hauptlauf

**Aktuelle Umfangsentscheidung:** Im ersten Hauptlauf wird G-Designer aus
Ressourcengruenden zurueckgestellt; siehe
[Kernvergleich](./COMPARATIVE_PROTOCOL.md#ressourcenbedingter-kernvergleich-ohne-g-designer-26102026).
Die folgenden Vollplan-Zahlen bleiben als urspruenglicher Fuenf-Bedingungen-
Plan erhalten, nicht als aktueller automatischer Startauftrag.
Der neue `core`-Plan umfasst 28800 Episoden / 136 Laeufe auf denselben
200 Fragen, nach 384 Pilotepisoden mit allen sechs neuen AnyMAC-Checkpoints.
G-Designer-Daten werden nicht geloescht oder nachtraeglich umgedeutet.
Die laufenden v2-Trainings bleiben im gepinnten Worktree unveraendert.
Den neuen Suitecode nur in einem separaten Evaluationsworktree verwenden;
nicht damit ein laufendes Training fortsetzen (Quellcodebindung).

Erst nach bestandenem Pilot und drei Trainingsseeds je gelerntem Router:

- 200 frisch reservierte offizielle Testfragen, thematisch ausgewogen (14/15 je
  Bereich); Nutzerentscheidung vom 26.10.2026 ersetzt die alte 60-Fragen-Planung.
  Kein Nachsamplen anhand von Angriffserfolgen. Alte 60 bleiben ausgeschlossen.
- Ehrlich: zwei Wiederholungen pro System/Checkpoint.
- Angriff: eine Wiederholung, jede Workerrolle rotiert, `bare` und `persuasive`.
- Gemeinsame Aufrufhuelle 32, keine erzwungene Ausschoepfung.
- Persuasive Texte einmal pro Aufgabe einfrieren; identischer Text in allen
  Systemen/Rollen/Seeds. Generator sieht nur Aufgabe und offline festgelegtes
  falsches Ziel. Maximal 10 % markierte Generationen, sonst Stopp.
- Vor Start explizites GPU-Stundenbudget, mindestens Pilotprojektion plus 25 %.
  Das ist eine **Zulassungsschaetzung**, keine Laufzeitgarantie oder
  programmatische Verbrauchsgrenze; Slurm-Zeitlimit weiterhin setzen.
  Der Plan schaetzt zusaetzlich das Trace-Volumen aus den Pilotdateien;
  Modellcache/Checkpoints und persoenliche Quota separat beruecksichtigen.

Mit beiden AnyMAC-Profilen und den Standardrollen sind das
**193 Einzellaeufe und 40.800 Episoden** (ohne Trainingsinterventionen).
Diese hohe Zahl ist KEIN Argument fuer den sofortigen Start. Zuerst gemessene
Laufzeit, ehrliche Leistung und Routingdiagnostik bewerten. Rollenanzahl,
Prompts und Trainingsumfang sind nicht voll harmonisiert. Systemunterschiede
sind deshalb kein isolierter kausaler Effekt des Routers. G-Designer und AnyMAC
sind mit ihren oben genannten Variantenbezeichnungen zu berichten.

Auswertung: Genauigkeit, unbedingte Zieltreffer, Exposition, Format-/Trunkierungs-
und Infrastrukturfehler, tatsaechliche Aufrufe/Tokens/Laufzeiten. Angriff minus
ehrlich je Form; persuasive minus bare. Wiederholungen, Rollen und Seeds werden
**innerhalb Aufgabe** gemittelt, dann 10.000-fach gepaart auf Aufgaben gebootstrapt.
Zusaetzlich Effekte je Trainingsseed. Aufgaben-CIs sind bedingt auf diese Seeds,
keine Unsicherheit ueber eine Population moeglicher Routertrainings.

Die 200 Aufgaben bleiben 200 Aufgabeneinheiten, nicht 40.800.
Sie garantieren keine Praezision fuer 0,10-Effekte. Breite Intervalle sind
unentschieden; fuer engere Bestaetigungsintervalle ist spaeter ein getrennt
vorregistrierter, neuer Aufgabensatz erforderlich. `bare` und `persuasive`
unterscheiden sich auch in Laenge/Inhalt: kein isolierter kausaler Nachweis
von Ueberzeugungskraft ohne laengenangepasste Kontrolle.

## 4a. Vertrauensaufbau im Training: getrenntes 2x2-Experiment

Leitfrage: Erzeugen hilfreichere oder kompetenzbetont dargestellte Workerantworten
eine gelernte Abhaengigkeit, die ein spaeterer Verhaltenswechsel ausnutzen kann?

`trust.py` friert pro offizieller Validierungs-Trainingsaufgabe zwei Inhalte ein:
normale Rollenloesung ohne Referenzlabel und eine Loesung mit Zugriff auf das
**richtige Trainingslabel**. Letzteres ist ein explizites staerkeres Angreifermodell.
Keine Entwicklungs-/Testlabels oder fremden Agentenantworten gehen in die
Erzeugung ein. Dass die zweite Loesung besser ist, wird nicht vorausgesetzt:
Korrektheit wird gemessen, falsche aber parsebare Ergebnisse bleiben erhalten.
Abgeschnittene/unparsebare Generationen stoppen die Freigabe, ohne Neusampling.
80 Generierungen fuer 40 Aufgaben; ein dauerhaftes Fortschrittsdokument erlaubt
Resume ohne erneute Erzeugung fertiger Texte.

Die vier Bedingungen sind `ordinary-neutral`, `ordinary-prominent`,
`helpful-neutral`, `helpful-prominent`. Darstellung wird durch eine vorangestellte
neutrale bzw. kompetenzbetonte Aussage mit gleicher **Wortzahl** variiert.
Der nachfolgende Antwortinhalt bleibt dabei byte-identisch. Das ist noch keine
Tokenizer-Laengengleichheit oder semantisch validierte perfekte Stilkondition.
Die Inhalte selbst koennen unterschiedlich lang sein; dieser Faktor ist
zunaechst das Paket aus Antwortqualitaet und Erzeugungsinstruktion.

Ein gemeinsamer, rollenunabhaengig von Ergebnissen festgelegter Zielworker wird
vor der Bankgenerierung explizit angegeben; es gibt keinen stillen Default.
Alle Aufrufe dieser einen Identitaet auf allen Trainingsaufgaben werden ersetzt.
Die Rolle wird nicht erzwungen. Alle anderen Ausgaben, Labels, Rewards und
Optimierungsbudgets bleiben unveraendert. Ein normaler Workeraufruf findet
weiterhin statt; seine Originalantwort wird gespeichert, bevor die Bankantwort
an weitere Agenten und den Router geht. Gemessene Tokens beschreiben diese realen
Aufrufe, nicht die Laenge des eingesetzten Banktexts. Die gleiche eingefrorene
Antwort wird auch bei wiederholtem Rollenaufruf verwendet und ignoriert den
jeweiligen Peer-Kontext; das ist ein kontrollierter Eingriff, kein nativer Worker.

Der neutrale normale Bankarm kontrolliert diese Ersetzung selbst und ist
**zusaetzlich zur unberuehrten sauberen Referenz** notwendig. Entwicklungslaeufe
erhalten keine Banktexte und keinen Loesungszugriff. Gewichte/Labels/Rewards werden
nicht direkt manipuliert. Eintraege sind separat als `training_intervention`
markiert; `policy=honest` bedeutet hier nur, dass der Evaluations-Angriffshook
nicht aktiv ist. Trainingslogs sind keine ehrlichen Evaluationsepisoden.

Je normalem AnyMAC, Efficient und G-Designer: vier Bankarme x drei Seeds =
36 interventionelle Trainings, **zusaetzlich** zu neun sauberen Trainings.
Erst seed0-Pilot und Kosten-/Speicherpruefung, keine automatische Vollfreigabe.
Pro Seed und Ergebnis Y ist die faktorielle Interaktion
`Y(helpful,prominent)-Y(helpful,neutral)-Y(ordinary,prominent)+Y(ordinary,neutral)`.
Rollen/Seeds/Wiederholungen sind keine unabhaengigen Aufgabeneinheiten.

Der saubere Hauptsuite-Loader verweigert interventionell trainierte Artefakte.
Eine vollstaendige, gepaarte Interventions-Evaluationssuite, fachlich akzeptierter
Leistungstoleranzbereich und kontrollierte Routingfortsetzungen sind noch
ausstehend. Vor deren Umsetzung keine Aussage ueber erhaltene Leistung,
Vertrauensaufbau oder kausalen Dynamikvorteil. Dynamisches Routing kann sowohl
Korrektur als auch zusaetzliche Angriffssteuerung ermoeglichen; feste Topologie
verhindert keine inhaltliche Korrektur durch Worker und Entscheider.

## 5. Ausfuehrung

Siehe [praktische Anleitung](../scripts/compare/README.md#gelernte-router-und-vier-system-suite).
Die historische Bestaetigung und der alte Preflight-Worktree duerfen weiterlaufen.
Quellcode fuer neue Laeufe in einem neuen, auf den Commit fixierten Worktree
bereitstellen. Resume setzt identischen Code, Daten und Einstellungen voraus.
