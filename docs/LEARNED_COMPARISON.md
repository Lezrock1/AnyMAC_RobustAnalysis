# Gelernte Router im Gemma-Vergleich

Stand 21.10.2026. Implementiert und lokal mit kleinen Modellen/simulierten Antworten
getestet; **noch keine Gemma-Router trainiert oder auf Ramses validiert**.
Diese Unterscheidung gilt auch bei gruener lokaler Testsuite.

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
  Reward `korrekt * 0.95^Workeranzahl`. G-Designer: zehn Graphsamples,
  Lernrate `1e-3`, Korrektheitsreward plus die expliziten Paper-Regularisierungen.
- Leave-one-out-Baseline pro Aufgabe; kein Skalieren durch die Reward-Streuung.
  AdamW ohne Weight Decay, Gradient-Clipping bei Norm 1.
- Unterschiedliche Trainingsumfaenge sind methodenbezogene Entscheidungen,
  keine Behauptung gleicher Trainingskosten.
- Seeds 0, 1, 2 werden fuer den Hauptvergleich separat trainiert.
  Zunaechst nur Seed 0 und den Pilot ausfuehren.

Gold wird erst nach einer abgeschlossenen Trajektorie fuer Reward/Scoring gelesen.
Es gibt keine Angriffe, Marker, Withholding-Policies oder Evaluationsupdates im
Routertraining. Budget-/Backend-/Kontrollfehler im Training stoppen explizit.

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

## 3. Entscheidungsschranken

Ein gelernter Router ist erst freigegeben, wenn:

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

Danach: vier Systeme x 12 Dev-Aufgaben x ehrlich/`bare` x zwei Wiederholungen =
**192 Pilotepisoden**, alle mit gemeinsamer Aufrufhuelle 32. Das ersetzt nicht
die einzelnen Trainings-Dev-Gates. Der Pilot liefert die empirische Kostenbasis.
Angriffsbedingte Formatfehler sind Messergebnisse, keine nachtraeglich zu
entfernenden Aufgaben. Die Freigabe beruht auf ehrlicher Ausgabequalitaet.

## 4. Bedingter Hauptlauf

Erst nach bestandenem Pilot und drei Trainingsseeds je gelerntem Router:

- 60 vorab reservierte Aufgaben, kein Nachsamplen anhand von Angriffserfolgen.
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

Mit den Standardrollen sind das **136 Einzellaeufe und 8.640 Episoden**.
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

Die 60 Aufgaben bleiben 60 unabhaengige Aufgabeneinheiten, nicht 8.640.
Sie garantieren keine Praezision fuer 0,10-Effekte. Breite Intervalle sind
unentschieden; fuer engere Bestaetigungsintervalle ist spaeter ein getrennt
vorregistrierter, neuer Aufgabensatz erforderlich. `bare` und `persuasive`
unterscheiden sich auch in Laenge/Inhalt: kein isolierter kausaler Nachweis
von Ueberzeugungskraft ohne laengenangepasste Kontrolle.

## 5. Ausfuehrung

Siehe [praktische Anleitung](../scripts/compare/README.md#gelernte-router-und-vier-system-suite).
Die historische Bestaetigung und der alte Preflight-Worktree duerfen weiterlaufen.
Quellcode fuer neue Laeufe in einem neuen, auf den Commit fixierten Worktree
bereitstellen. Resume setzt identischen Code, Daten und Einstellungen voraus.
