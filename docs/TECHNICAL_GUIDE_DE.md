# Technische Vertiefung: Anomalieerkennung mit wenigen Normalbildern

## 1. Warum diese Aufgabe anders ist

Bei einer überwachten Segmentierung liegen Beispiele und Masken der gesuchten Fehler vor. In der industriellen Anlaufphase stehen häufig viele Varianten guter Oberflächen, aber nur wenige bekannte Fehler zur Verfügung. Eine Normalitätsbeschreibung kann dann Abweichungen erkennen, ohne jede Fehlerklasse im Training zu benötigen.

Das Projekt untersucht diesen Ansatz mit fünf oder fünfzehn normalen Fit-Bildern. Es vergleicht feste lokale Deskriptoren, Merkmale eines auf Normalbildern trainierten Encoders und den Rekonstruktionsfehler eines Autoencoders. Fehlerbilder werden ausschließlich zur abschließenden Auswertung genutzt.

Die Zahl fünfzehn bezeichnet die Bilder zum Aufbau des Modells. Zusätzlich gibt es 32 unabhängige Normalbilder zur Schwellwertbestimmung. Ein Gesamtbudget von nur fünfzehn Bildern für Training und Kalibrierung wurde damit nicht geprüft. Diese Unterscheidung ist zentral für eine ehrliche Beschreibung des Few-Shot-Ergebnisses.

## 2. Datenaufteilung ohne Informationsleck

Ein Pool von 40 synthetischen Normalbildern liefert die Fit-Teilmenge. Jeder Seed erzeugt eine andere Permutation; die fünf Bilder sind eine Teilmenge der fünfzehn Bilder desselben Seeds. Dadurch ist der Vergleich der Shot-Budgets besser kontrolliert, als wenn beide Mengen vollständig unabhängig gezogen würden.

Der Normal-Kalibrierungssatz hat einen anderen Generator-Seed. Der Testsatz enthält 40 normale und 40 fehlerhafte Bilder aus einem weiteren Seed. Beide Fit- und Kalibrierungsmasken werden auf positive Pixel geprüft; ein Fehler führt zum Abbruch. Bei eigenen Bildern prüft das CSV-Audit außerdem Komponenten-/Chargengruppen und identische Bildinhalte über Split-Grenzen.

Der Fit-Satz bestimmt Gewichte, Mittelwerte, Standardabweichungen und Prototypen. Der Kalibrierungssatz bestimmt die Schwellen. Der Testsatz misst anschließend Ranking, Lokalisation und Entscheidungen. Diese drei Informationswege dürfen nicht vermischt werden.

## 3. Feste Patch-Merkmale

Aus jedem Bild entstehen vier Merkmalskanäle: Intensitätsabweichung vom lokalen Mittelwert, Betrag des horizontalen Gradienten, Betrag des vertikalen Gradienten und lokale Standardabweichung. Ein 5×5-Ausschnitt über diese vier Kanäle besitzt 100 Werte. Solche Patches werden im Abstand von vier Pixeln extrahiert.

Der lokale Mittelwert reduziert langsam wechselnde Beleuchtung. Gradienten reagieren auf Kanten und Kratzer. Die Standardabweichung beschreibt lokale Textur. Diese fachlichen Annahmen passen gut zum verwendeten Generator, dessen Fehler vor allem lokale Kontrast- und Strukturänderungen sind. Das erklärt auch, weshalb diese Merkmale hier besser abschneiden können als ein kleines gelerntes Modell.

Die Merkmale werden mit Mittelwert und Standardabweichung der Fit-Patches standardisiert. Eine Mindeststandardabweichung von 0,01 verhindert extreme Verstärkung nahezu konstanter Dimensionen. Diese Normalisierung muss zusammen mit der Bank gespeichert und bei Inferenz unverändert angewendet werden. Eine neue Standardisierung auf Testbildern würde die Bedeutung der Distanz verändern und Informationen aus dem Testsatz einbeziehen.

## 4. Eine repräsentative Speicherbank aufbauen

Alle Patches aus fünfzehn 64×64-Bildern ergeben 15×16×16=3840 Kandidaten. Statt sie vollständig zu speichern, werden 64 oder 256 Prototypen ausgewählt.

Die Greedy-Auswahl beginnt bei einem Kandidaten mit kleinem Abstand zum Nullpunkt im standardisierten Raum. Danach wird immer der Kandidat aufgenommen, dessen Abstand zur bisher gewählten Bank am größten ist. Nach jedem Schritt wird der kleinste Abstand jedes Kandidaten zu irgendeinem gewählten Prototyp aktualisiert.

Dadurch wächst die Abdeckung des Normalraums. Der maximale verbleibende Abstand ist der ausgewiesene Abdeckungsradius. Bei fortgesetzter Auswahl kann dieser Radius nicht größer werden. Ein Test prüft diese Eigenschaft. Ein weiterer Test stellt sicher, dass selbst bei identischen Merkmalen keine Kandidatenindizes mehrfach gewählt werden.

Der Auswahlaufwand ist ungefähr `N×K×D` für Kandidatenzahl `N`, Budget `K` und Dimension `D`. Gespeichert werden Kandidatenmerkmale und ein Vektor aktueller Mindestabstände; eine vollständige `N×N`-Distanzmatrix wird nicht benötigt. Für viel größere reale Sammlungen wären Projektionen oder approximative Auswahlverfahren zu untersuchen.

## 5. Von der Patch-Distanz zur Anomaliekarte

Ein neuer Patch wird mit der gespeicherten Bank verglichen. Sein Score ist die kleinste euklidische Distanz zu einem Normalprototyp. Große Distanzen bedeuten, dass die Merkmalskombination durch die Fit-Normalbilder schlecht abgedeckt wird. Das ist zunächst eine Abweichung, noch keine semantisch bestätigte Fehlerklasse.

Die Abfragen werden in Teilmengen verarbeitet, um die temporäre Distanzmatrix zu begrenzen. Tests vergleichen kleine und große Abfrageblöcke. Die Implementierung erzwingt den direkten euklidischen Rechenweg: Eine automatische Umschaltung auf die Identität `||a-b||²=||a||²+||b||²-2aᵀb` kann bei fast identischen Vektoren durch Auslöschung kleine künstliche Restdistanzen erzeugen. Die gewählte Variante hält besonders exakte Prototypübereinstimmungen stabil.

Die grobe Karte wird in die Bildkoordinaten übertragen. Bei festen Deskriptoren liegen die Patch-Zentren bei 0, 4, 8 und so weiter. Explizite Abtastkoordinaten erhalten diese Zuordnung; außerhalb des letzten Zentrums wird der Randwert verwendet. Das ersetzt keine feine Segmentierung: Ein Patch beschreibt eine Umgebung, weshalb ein kleiner Defekt einen breiteren hohen Score erzeugen kann.

## 6. Gelernte Merkmale und Rekonstruktion

Der Autoencoder verkleinert das Bild über Faltungen auf einen kleinen Bottleneck und rekonstruiert es anschließend. Trainiert wird nur mit normalen Bildern und mittlerem absolutem Rekonstruktionsfehler.

Für die Rekonstruktionsvariante ist der Betrag der Differenz zwischen Eingabe und Rekonstruktion die Anomaliekarte. Für die Encoder-Speicherbank stammen die Patch-Merkmale aus einer Zwischenstufe desselben normal trainierten Encoders. So lassen sich gelernte Darstellung und Rekonstruktionsscore getrennt vergleichen.

Ein Autoencoder kann Fehler mitrekonstruieren oder normale Textur ungenau abbilden. Beides schadet der Erkennung. Außerdem bedeuten wenige Normalbilder, dass die Darstellung selbst wenig Variation kennenlernt. Hier wird kein ImageNet-Backbone verwendet. Die Ergebnisse dürfen deshalb nicht als Reproduktion eines PatchCore-Systems mit vortrainierter Repräsentation ausgegeben werden.

## 7. Bildentscheidung und pixelweise Lokalisation trennen

Der Bildscore ist der Mittelwert der höchsten ein Prozent der Pixelwerte. Ein einzelnes gestörtes Pixel beeinflusst ihn damit weniger als bei einem absoluten Maximum. Gleichzeitig bleiben größere lokale Abweichungen sichtbar. Fehler deutlich unterhalb dieser Flächengröße können aber verdünnt werden.

Die Bildschwelle bestimmt, ob ein Teil einen Alarm auslöst. Die Pixelschwelle bestimmt die eingezeichnete Region. Beide Aufgaben haben unterschiedliche Zielgrößen. Ein Bild kann korrekt als fehlerhaft erkannt werden, obwohl die eingezeichnete Maske unscharf ist. Genau deshalb stehen Bild-AUROC, Pixel-AP und schwellwertabhängiger Recall getrennt im Bericht.

## 8. Die Kalibrierungsformel

Seien `n` unabhängige normale Kalibrierungsscores aufsteigend sortiert. Für den gewünschten normalen Fehlalarmanteil `alpha` wird der Rang

$$k=\lceil(n+1)(1-\alpha)\rceil$$

verwendet. Wenn `k≤n`, ist die Schwelle der Wert mit diesem Rang. Ein Alarm entsteht strikt oberhalb der Schwelle. Ist `k>n`, setzt die Implementierung die Schwelle auf unendlich. So wird eine nicht durch die kleine Stichprobe gestützte endliche Schwelle vermieden.

Bei 32 Kalibrierungsbildern und `alpha=0,05` ist `k=32`; verwendet wird also das Maximum. Bei nur 20 Bildern und einem Ziel von einem Prozent reicht die Stichprobe für eine endliche Schwelle dieser Form nicht aus. Der Test prüft diesen Grenzfall und den Umgang mit identischen Scores.

Die bekannte marginale Fehlalarmkontrolle setzt Austauschbarkeit der normalen Kalibrierungs- und späteren Normalbilder voraus. Ein Wechsel von Kamera, Material oder Beleuchtung kann diese Grundlage zerstören. Die Aussage gilt außerdem für normale Bildalarme, nicht für Defekt-Recall, fehlerfreie Masken oder eine Garantie auf jeder einzelnen Charge.

Die Pixelschwelle ist lediglich das empirische 99,5-Prozent-Quantil der Kalibrierungspixel. Wegen der Korrelation innerhalb eines Bildes wird daraus keine pixelweise oder bildweise Conformal-Garantie abgeleitet.

## 9. Unsicherheit und gemessene Unterschiede

Die festen Merkmale erreichen im Demo eine Bild-AUROC von eins. Gleichzeitig liegt ihre Pixel-AP bei fünfzehn Bildern und 256 Prototypen nur ungefähr bei 0,460. Das zeigt eine einfache Bildtrennung bei erheblich ungenauerer Lokalisation. Ein leicht passender Generator kann auch mit wenigen Normalbildern perfekt getrennt werden; eine reale Kameraoberfläche kann ganz anders reagieren.

Die Encoder-Speicherbank erreicht unter derselben Konfiguration etwa 0,903 Bild-AUROC, aber nur etwa 0,192 Defekt-Recall an der kalibrierten Schwelle. Ein gutes Ranking garantiert also keinen brauchbaren Betriebspunkt. Der Autoencoder erreicht etwa 0,933 AUROC und etwa 0,633 Recall. Die Zahlen sind keine austauschbaren Qualitätsnoten, sondern Antworten auf unterschiedliche Fragen.

Wilson-Intervalle zeigen die Unsicherheit von Recall und normalen Bildalarmen. Bei 40 normalen Testbildern beweisen null beobachtete Alarme keinen wahren Fehlalarmanteil von null. AUROC-Intervalle entstehen durch Bootstrap vollständiger Bilder. Pixel werden nicht als unabhängig resampelt, weil das die effektive Stichprobe stark überschätzen würde.

## 10. Was ein technisches Review prüfen sollte

**Was bedeutet eine große Distanz?** Eine Abweichung von den erfassten normalen Merkmalen. Auch eine neue legitime Textur oder Beleuchtung kann sie erzeugen. Die Interpretation erfordert Fachwissen und repräsentative Normaldaten.

**Warum die zusätzliche Kalibrierung?** Die Fit-Daten legen das Modell fest; unabhängige Normalbilder liefern eine weniger verzerrte Grundlage für den Alarmbetriebspunkt. Die zusätzliche Bildanzahl muss zum Gesamtaufwand gerechnet werden.

**Welche Aussage liefert das Coreset?** Es reduziert die gespeicherten Normalprototypen und misst deren geometrische Abdeckung im gewählten Merkmalsraum. Es garantiert nicht, dass alle zukünftigen Gutteilvarianten enthalten sind.

**Was kommt vor einer industriellen Anwendung?** Reale Materialchargen, Beschmutzung, Lageänderungen und Beleuchtungswechsel getrennt prüfen; Fehlerkosten festlegen; Auflösung und ROI bestimmen; Schwellwertstabilität untersuchen; neue Normalvarianten überwachen und ein erneutes Kalibrierungsverfahren definieren.
