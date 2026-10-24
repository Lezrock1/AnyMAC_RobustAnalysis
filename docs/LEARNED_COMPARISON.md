# Gelernte Router im Gemma-Vergleich

Stand 24.10.2026, laut uebermittelten Ramses-Ausgaben. Historischer AnyMAC-Seed 0
mit Kostenfaktor 0,95 trainiert, aber Kooperations-Gate nicht bestanden.
G-Designer-Seed 0 und neues AnyMAC-normal Seed 0 bestehen das Dev-Gate.
AnyMAC-efficient Seed 0 ist fertig trainiert, besteht aber das Kooperations-Gate
nicht. Normal-/Efficient-Seed 1 wurden als Jobs 6984386/6984387 gestartet.
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
Die bestehende Hauptsuite bleibt fuer nicht bestandene Checkpoints gesperrt.
Eine etwaige separate Berichterstattung solcher Betriebsmodi benoetigt einen
expliziten Plan, nicht die stille Abschwaechung des Gates.

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

Danach: fuenf Bedingungen x 12 Dev-Aufgaben x ehrlich/`bare` x zwei Wiederholungen =
**240 Pilotepisoden**, alle mit gemeinsamer Aufrufhuelle 32. Das ersetzt nicht
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

Mit beiden AnyMAC-Profilen und den Standardrollen sind das
**193 Einzellaeufe und 12.240 Episoden** (ohne Trainingsinterventionen).
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

Die 60 Aufgaben bleiben 60 unabhaengige Aufgabeneinheiten, nicht 12.240.
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
