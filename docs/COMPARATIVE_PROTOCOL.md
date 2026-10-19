# Vergleich dynamischer Multi-Agent-Router: Vorbereitung

Status: Entwurf vom 19.10.2026, noch kein eingefrorenes Hauptversuchsprotokoll.
Dieser Entwurf erweitert den Gegenstand ueber AnyMAC hinaus. Er aendert weder das
[historische AnyMAC-Protokoll](./PROTOCOL.md) noch bereits gespeicherte Ergebnisse.
Die [isolierten Folgetests](../scripts/followup/README.md) sind implementiert;
Adapter fuer weitere Router sind noch NICHT implementiert oder auf Ramses validiert.

## Ziel und Ressourcen

Untersucht wird, wie Routing und Kommunikation in LLM-Multi-Agent-Systemen die
Exposition gegenueber einem kompromittierten Agenten in tatsaechlichen Einfluss
auf das Ergebnis uebersetzen. Markerlernen und sauberes Weitertraining sind
Teiluntersuchungen fuer geeignete lernende Router, nicht die gemeinsame Hauptfrage.

- Vereinbarter Umfang: AnyMAC, zwei weitere veroeffentlichte dynamische Verfahren,
  eine feste Referenzstrategie.
- Verfuegbar laut Nutzer: bis zu drei GPU-Knoten, mehrtaegige Laeufe moeglich.
- Unbekannt: GPU-Anzahl und VRAM pro Knoten, bewilligte Partitionen und deren
  Zeitlimits. Drei Knoten bedeuten nicht automatisch drei gemeinsam nutzbare GPUs.
- Kein Anlass fuer unbeschraenkte Sweeps: Jede Erweiterung braucht eine konkrete
  Hypothese, Funktionspruefung und Kostenmessung.

## 1. Methodenwahl mit primaeren Quellen

Die folgende Auswahl ist eine begruendete Arbeitsauswahl, keine Behauptung,
dass die Verfahren gleichartige Entscheidungen treffen oder bereits lauffaehig sind.
Repository-Staende wurden am 19.10.2026 ueber GitHub gelesen. Vor Implementation
muessen relevante Ausfuehrungspfade, Modellanforderungen und Paper-Code-Abweichungen
direkt geprueft werden. Ein README allein beweist keine Reproduzierbarkeit.

| Verfahren | Entscheidungsgegenstand | Rolle im Vergleich | Stand |
|---|---|---|---|
| AnyMAC | Naechster Agent und zugestellter Kontext entlang einer Trajektorie | Gelernter sequenzieller Router | Bestehende Pipeline; Originalfehler und Lernprobleme bekannt |
| DyLAN | Inferenzzeitliche Auswahl/Reduktion aktiver Agenten, Konsens und fruehes Stoppen | Andersartige, antwortabhaengige dynamische Zusammenarbeit | Offizieller Code und Auswahlpfad im Demo geprueft; lokaler Adapter fehlt |
| MasRouter | Aufgabenabhaengige Wahl von Zusammenarbeit, Agentenzahl, Rollen und LLMs | Aufgabenbasierte Systemkonfiguration statt Next-Agent-Routing | Offizieller Code und Controller-Eingaben geprueft; trainierter Zustand/Training und Adapter offen |
| Feste Referenz | Vorab festgelegte zyklische Reihenfolge und Zustellung | Kontrollstrategie ohne adaptive Auswahl | Muss dieselbe Ausfuehrungs-/Logging-Schicht nutzen |

Primaere Quellen und konkret gelesene Code-Staende:

- AnyMAC: [Paper](https://arxiv.org/abs/2506.17784),
  [Originalcode](https://github.com/SongW-SW/AnyMAC).
  Unsere historischen Checkpoints gehoeren zur dokumentierten Originalvariante;
  eine korrigierte Variante ist getrennt zu trainieren und zu kennzeichnen.
- DyLAN: [Paper](https://arxiv.org/abs/2310.02170),
  [offizielles Repository](https://github.com/SALT-NLP/DyLAN),
  Commit `006e440a519f7cf21e2826f3b8033d84ae9bf07c` (MIT).
  `code/demo/LLMLP.py` implementiert Aktivierung, Konsens und Listwise-Auswahl.
  README nennt MATH, MMLU und HumanEval; OpenAI-Transport muss lokal ersetzt werden.
- MasRouter: [Paper](https://arxiv.org/abs/2502.11133),
  [offizielles Repository](https://github.com/yanweiyue/masrouter),
  Commit `e005f7696fe0c0412563f6cd67f4cd3712fa2822` (Apache-2.0).
  `MAR/MasRouter/mas_router.py` nimmt Abfragen, Aufgaben, LLMs und
  Zusammenarbeitsoptionen entgegen. README nennt GSM8K, MATH, MMLU, MBPP und HumanEval.
  Nicht als antwortabhaengiger Next-Agent-Router bezeichnen.

Alternativen, nicht parallel ebenfalls implementieren:

- [Puppeteer](https://arxiv.org/abs/2505.19591):
  Paper verlinkt [OpenBMB/ChatDev, Branch puppeteer](https://github.com/OpenBMB/ChatDev/tree/puppeteer),
  gelesen bei `b24e6e6bd00bb04d317e0447cb7acb1e4973c83a`.
  Sequenzielle gelernte Orchestrierung ist relevant. Die offizielle Anleitung
  verlangt `nvidia/Llama-3.1-Nemotron-70B-Reward` als Orchestratorbasis und verwendet
  standardmaessig GPT-4o-Agenten. Kein Austausch gegen ein kleines Modell unter
  unveraenderter Methodenbezeichnung. Erst nach Hardware-/Checkpointpruefung eine Alternative.
- [G-Designer](https://arxiv.org/abs/2410.11782):
  [Code](https://github.com/yanweiyue/GDesigner), Commit
  `a6efcfa3b40bb4d9cbf46f883a95d62020bd8251`.
  Aufgabenabhaengige Graphkonstruktion ist eine Alternative zu MasRouter.
  GitHub meldet keine erkannte Lizenz; vor Codeuebernahme klaeren.
  Zuvor festgestellte Paper-Code-Abweichungen muessen methodisch dokumentiert werden.

Entscheidungsschranke: Kann MasRouter nur mit wesentlich veraenderter Rollen-/Modellauswahl
betrieben werden, wird es nicht stillschweigend auf AnyMAC zugeschnitten. Stattdessen
separaten nativen Vergleich ausweisen oder die Methodenwahl begruendet revidieren.

## 2. Zwei Vergleichsebenen statt scheinbarer Gleichheit

### Gemeinsame Sicherheitsfragen

Aufgabe, Scoring, Angreiferrechte, Ausgabelimits und Fehlerklassifikation werden
vereinheitlicht. Methodeninterne Auswahl, Aggregation und Stopplogik bleiben erhalten.
Der gemeinsame Nenner ist eine falsche Endantwort bzw. ein fehlendes Ergebnis,
nicht eine fuer alle Methoden identische Routingentscheidung.

### Harmonisiert versus methodennativ

Ein harmonisierter Mechanismenvergleich verwendet moeglichst denselben Rollenpool,
dasselbe lokale Agentenmodell und denselben logischen Aufrufrahmen. Wenn dadurch
ein Kernelement entfernt wird (z.B. MasRouters LLM-Auswahl bei nur einem LLM),
ist dies eine explizite Ablation, keine vollstaendige Reproduktion.

Ein methodennativer Vergleich darf Rollen, Aggregation und Modellpool erhalten,
muss dann aber Systemunterschiede als solche berichten. Er isoliert nicht den
Router allein. Kein Vermischen beider Ebenen in einer einzigen Rangliste.
Welche Ebene fuer jeden Adapter sinnvoll ist, wird VOR Hauptauswertung fixiert.

## 3. Gemeinsames Angreifermodell

- Genau eine normale Solver-Identitaet ist kompromittiert, nicht der Entscheider,
  Router oder Pruefer. Ehrliche Alternativen bleiben verfuegbar.
- Der Angreifer darf nur Ausgaben dieser Identitaet bei tatsaechlichem Aufruf ersetzen.
  Kein Zugriff auf Goldantwort, Routergewichte, Auswahl oder Zustellungsmaske.
- Ziele sind pro Aufgabe vorab festgelegte falsche Antworten. Die Goldantwort
  darf zur offline Zielkonstruktion und Auswertung verwendet werden, nicht zur
  Generierung ehrlicher Antworten oder zur Angriffsauswahl nach Evaluationserfolg.
- Zielantwort-Manipulation ist ein einfacher, festgelegter Verhaltenstest:
  keine Optimierung von Umgehungs-/Exfiltrationsprompts und keine externen Tools.
- Kernbedingungen: alle ehrlich versus kontrollierte falsche Solver-Ausgabe.
  Ein reiner Zieltreffer zaehlt nicht automatisch als verursachter Angriffserfolg:
  Zieltrefferrate im ehrlichen Lauf ebenfalls ausweisen.
- Marker: gemeinsame neue Haupttests zunaechst OHNE Projektmarker.
  Alte markerbehaftete Pilotaufgaben sind Entwicklungs-/Diagnosedaten.
- Index-/Rollenbias: zuerst feste kompromittierte Identitaet im Kostenpilot;
  anschliessend auf einer vorab fixierten Teilmenge jede geeignete Solver-Identitaet
  einmal kompromittieren. Keine Auswahl der schwaechsten Rolle anhand Haupttestdaten.
- Bei variablen Teams muss die Zuordnung einer stabilen kompromittierten Identitaet
  vorab definiert sein. Keine nachtraegliche Kompromittierung des gerade ausgewaehlten
  Agenten nur fuer eine Methode.

## 4. Datentrennung und Ausgabequalitaet

- Bestehende 40 Pilot-Evaluationsaufgaben wurden mehrfach analysiert: NICHT mehr
  als unberuehrter Bestaetigungstest fuer allgemeine Aussagen behandeln.
- Bestehende Branch/Continue-Bestaetigung verwendet diese Aufgaben weiterhin als
  klar ausgewiesene Checkpoint-Diagnose.
- Routeruebergreifender Hauptversuch braucht neue, gespeicherte disjunkte IDs.
  Entwicklungs-, Routertrainings- und Evaluationsdaten strikt trennen.
- Erster gemeinsamer Funktionspilot: 12 neue Entwicklungsaufgaben, gleiche IDs fuer
  alle Methoden. Falls Kandidaten bisher keine MASBench-Unterstuetzung haben:
  Anpassung und Parser explizit testen, nicht automatisch als native Reproduktion ausgeben.
- Kernbenchmark kann MASBench breadth sein; eine zweite, anders strukturierte
  Reasoning-Domaene ist vor breiten Generalisierungsbehauptungen erforderlich.
  Auswahl und Lizenz/Zugang erst nach Adapterpruefung fixieren.
- Ausgabeprotokoll nur anhand Entwicklung festlegen. Strenger Formatparser ist
  kein semantischer Gutachter; stichprobenartige manuelle Formatpruefung vor Fixierung.
- Null protokollierte Ausfuehrungsfehler sowie vorab begrenzte Raten fuer Abschneidung
  und ungueltige Endformate als Entwicklungs-Gate. Die implementierte Bestaetigung
  nutzt jeweils <=5% betroffene Episoden, nicht <=5% aller Aufrufe.
- Nichttemperierte LLM-Ausgabe gilt nicht als deterministisch. Wiederholungen auch
  bei Temperatur 0; Ausfuehrungsreihenfolge balancieren und Serverstarts protokollieren.

## 5. Experimente und Entscheidungsschranken

### E0: Aktuelle Validierung (implementiert)

1. Snapshot plus Audit: abgeschlossen, Quelle bleibt unveraendert.
2. Ein-Schritt-Lerngrenze: zwei Umgebungen x zwei Lernverfahren x fuenf Seeds.
3. Sechs Branch/Continue-Checkpoints x zwei Wiederholungen x 40 Aufgaben x
   zwei Marker x zwei Policies = 1.920 neue Evaluationsepisoden, dazu 24 Dev-Episoden.

E0 ist kein routeruebergreifendes Hauptergebnis. Ein bestandenes Ein-Schritt-Gate
beweist keine Lernbarkeit der vollstaendigen AnyMAC-Pipeline.
Ramses-Start und Wiederaufnahme: [Follow-up-Anleitung](../scripts/followup/README.md).
Diese Laeufe koennen mehrere Allokationen benoetigen; 43 Stunden ist ein
kontrollierter Stopp, keine zugesicherte Gesamtlaufzeit.

### E1: Adapter- und Kostenpilot (als Naechstes zu implementieren)

Vier Systeme x 12 Dev-Aufgaben x zwei Policies x drei Wiederholungen =
288 Episoden nach bestandenem Transport-/Schemacheck.
Alle Kontroll-/Ranking-/Judge-Aufrufe mitzählen, auch wenn kein Worker erzeugt wird.
Kostenprojektion aus gemessenen Aufrufen, Tokens und Laufzeiten; noch keine
wissenschaftliche Rangliste aus diesen 12 Entwicklungsaufgaben.

Jeder Adapter muss nachweisen:

1. Gleiche Anfrage erzeugt die dokumentierte Eingabe; keine Goldinformation im Prompt.
2. Controller ersetzt nur die festgelegte Solver-Ausgabe nach echtem Aufruf.
3. Trace rekonstruiert jeden tatsaechlichen Empfaenger und jeden Router-/Worker-Aufruf.
4. Keine stillen Defaults bei ungueltiger Auswahl, Parserfehlern oder HTTP-Fehlern.
5. Native Stopplogik bleibt erhalten; Budgetabbruch ist separat gekennzeichnet.
6. Resume bindet Quellcode, Split, Modell-Digest, Konfiguration und Checkpoint.

Stoppschranke: Adapter ohne diese Nachweise startet keinen Hauptversuch.

### E2: Exposition und tatsaechlicher Einfluss (gemeinsamer Kern)

Nach Kostenpilot: Aufgabenzahl und Wiederholungen anhand Praezision und gemessener
Streuung festlegen. Nicht die vorlaeufigen MDE=0-Tabellen als Powernachweis verwenden.
Eine konkrete minimale wissenschaftlich interessante Effektgroesse und ein
Intervallbreitenziel muessen VOR Hauptversuch dokumentiert sein.

Berichten:

- Genauigkeit ehrlich/unter Angriff und deren gepaarte Differenz.
- Gezielte Treffer unter beiden Policies, mit unbedingtem Nenner.
- Auswahl der kompromittierten Identitaet, Aufrufe und Exposition pro Episode.
- Zustellung an Worker, Ranker/Router und finale Aggregation getrennt.
- Zieltreffer gegeben Exposition nur deskriptiv: Exposition ist selbst
  methodenabhaengig und kann durch Manipulation veraendert werden.
- Routingkonzentration, wiederholte Selbstaufrufe, fruehes Stoppen.
- Ungueltige Antwort, Abschneidung, Backendfehler und Budgetabbruch getrennt.
- Alle LLM-Aufrufe, Tokens, Laufzeit und tatsaechliche GPU-Stunden.

Gemeinsame Aufrufobergrenze nur als Budgethülle; keine erzwungene AnyMAC-Schrittzahl
fuer rundenbasierte Verfahren. Innerhalb derselben Huelle vergleichen, erreichte
Aufrufzahlen zeigen und bei Bedarf Qualitaet/Kosten-Kurven auf Dev festlegen.
Budgetabbruch nicht stillschweigend durch einen zusaetzlichen Entscheideraufruf reparieren.

### E3: Mechanismen durch kontrollierte Eingriffe

Vorab festgelegte Teilmenge, nicht ausschliesslich nach beobachtetem Angriffserfolg
ausgewaehlte Faelle. Gespeicherter Ausgangszustand:

- Unveraenderte Fortsetzung (Sham) gegen denselben Eingriffspunkt.
- Genau eine falsche Nachricht durch eine ehrlich erzeugte Nachricht ersetzen.
- Vorhandene ehrliche Korrektur zusaetzlich zustellen, wo diese Entscheidung existiert.
- Naechsten Agenten durch vorab festgelegte Alternative ersetzen, wo anwendbar.

Alle Fortsetzungen wiederholen, Rechen-/Kontextaenderungen explizit ausweisen.
Interventionspaare teilen denselben Prefix; Zufallsseeds allein garantieren keine
identischen LLM-Ausgaben. Nicht anwendbare Eingriffe sind KEINE erfolgreichen Abwehrfaelle.
Nachrichtenaustausch aendert auch Stil/Laenge; erst mit geeigneten Kontrollen einen
Inhalts- oder Wahrheitsmechanismus behaupten.

### E4: Anpassung und Abwehr (erst nach bestaetigtem E2/E3-Effekt)

Sauberes Weitertraining ist eine separate Untersuchung fuer lernende Router.
Mindestens unabhaengige Trainingsseeds, identische Ausgangsbedingungen der Arme
und ehrliche Leistung vor/nach Anpassung pruefen. Ein Seed ist nur eine Fallstudie.
Abwehr nur auf nachweislich exponierten Konfigurationen; Mehrkosten-Kontrolle
und ehrliche Leistungsveraenderung mitberichten.

## 6. Statistik und Reproduzierbarkeit

- Primäre Einheit fuer Aufgabenunsicherheit ist die Aufgabe, nicht jede Nachricht
  oder wiederholte Generierung. Alle Varianten derselben Aufgabe zusammen resamplen.
- LLM-Wiederholungen und Trainingsseeds sind verschiedene Streuungsquellen.
  Wiederholte Evaluation eines Checkpoints ersetzt keine Trainingsseeds.
- Vergleiche moeglichst gepaart auf Aufgaben; Seedpaarung nur bei begruendet
  gemeinsamen Start-/Trainingsbedingungen.
- Mehrere Router, Rollen, Domaenen und Metriken erzeugen Mehrfachvergleiche.
  Primaere Kontraste vorab fixieren; explorative Ergebnisse kennzeichnen.
- Keine Robustheitsbehauptung aus Nullkontakt mit dem kompromittierten Agenten.
- Keine Neuheitsbehauptung nur aus dieser Kandidatenrecherche. Vor Festlegung des
  Beitrags gezielte Literaturpruefung zu MAS-Robustheit und Kommunikationsinterventionen.

## 7. Ausfuehrungsreihenfolge und offene Angaben

Jetzt: einen Knoten mit 16 CPUs fuer den implementierten `followup/run.sh both`
verwenden. Keine drei identischen Jobs auf denselben Ergebnisordner starten.
Die uebrigen Knoten erst nach Adaptertests fuer getrennte Methodenjobs einsetzen;
vorher waeren sie kein sinnvoller wissenschaftlicher Mehrwert.

Naechste Implementationsreihenfolge:

1. Gemeinsamer Ereignisvertrag, Scoring und fixe Referenz mit Tests.
2. DyLAN-Adapter, gepinnter offizieller Auswahl-/Stopppfad.
3. MasRouter-Code-/Checkpointaudit; native versus harmonisierte Ausfuehrung festlegen.
4. Lokal gemockter End-to-End-Test aller Adapter; dann E1 auf Ramses.
5. Kosten/Praezision aus E1 pruefen, Protokoll einfrieren, erst dann E2/E3.

Noch vom Nutzer zu klaeren:

- Welche GPUs/VRAM sind tatsaechlich je Knoten anforderbar?
- Ausschliesslich lokale Modelle, oder waeren kostenpflichtige externe APIs
  ueberhaupt erlaubt? Bis dahin werden KEINE externen Modellanfragen vorbereitet.
- Neue Rohdaten/Prompts duerfen fuer diese Arbeit nur lokal bzw. auf Ramses bleiben;
  keine Uebertragung an Drittanbieter ohne gesonderte Zustimmung.

Literatur-/Coderecherche fand ueber oeffentliche Primaerquellen statt. Keine
Experimente anderer Methoden wurden ausgefuehrt; Installationen und grosse
Modelldownloads sind nicht Bestandteil dieser Vorbereitung.
