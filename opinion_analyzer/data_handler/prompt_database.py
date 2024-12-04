import json as js
import random
from opinion_analyzer.utils.helper import get_main_config

config = get_main_config()

random.seed(42)
example_json_arguments = """
                
                        [
                            {
                                "topic": "Textabschnitt aus dem Textauszug, der den Aspekt beschreibt, worum es indem Argument geht"
                                "argument": "Textabschnitt aus dem Quellentext rein, der das Argument darstellt"
                                "claim": "Kurz und bündige Umschreibung der These aus dem Argument" 
                                "evidence": "Kurz und bündige Umschreibung der Begründung aus dem Argument" 
                                "stance": "Erlaubt sind nur 'Pro' oder 'Contra', je nachdem ob das Argument für oder gegen den genannten Aspekt ist" 
                                "person": "Person, von der das Argument kommt"
                                "party": "Partei, der die Person, von der das Argument kommt, angehört"
                                "kanton": "Kanton, wo die Person, von der das Argument kommt, tätig ist"
                            },
                            {
                                "topic": "Textabschnitt aus dem Textauszug, der den Aspekt beschreibt, worum es indem Argument geht"
                                "argument": "Textabschnitt aus dem Quellentext rein, der das Argument darstellt"
                                "claim": "Kurz und bündige Umschreibung der These aus dem Argument" 
                                "evidence": "Kurz und bündige Umschreibung der Begründung aus dem Argument" 
                                "stance": "Erlaubt sind nur 'Pro' oder 'Contra', je nachdem ob das Argument für oder gegen den genannten Aspekt ist" 
                                "person": "Person, von der das Argument kommt"
                                "party": "Partei, der die Person, von der das Argument kommt, angehört"
                                "kanton": "Kanton, wo die Person, von der das Argument kommt, tätig ist"
                            }
                            
                        ]
                """

example_json_argument_segments = """
                                [
                                {"segment": "Text segment"},
                                {"segment: "Text segment"}
                                ]
                                """
extract_arguments_1 = """
                    Ich gebe dir ein Textauszug, der ein Thema beschreibst.
                    Anschließend gebe ich dir einen Quellentext.
                    Analyse den Quellentext, ob da Pro- oder Contra-Argumente zu einem Aspekt des vorgeschlagenen Themas aufgelistet werden.
                    Wenn du Argumente findest, extrahiere alle Pro- und Contra-Argumente aus dem Quellentext. 
                    Extrahiere nur die jeweiligen Textabschnitte und ändere die Formulierung nicht.
                    
                    #Hier ist der Textauszug mit dem Thema:
                    
                    {topic_text}
                    
                    #Hier ist der Quellentext mit den möglichen Argumenten:
                    
                    {argument_text}
                    
                    """

reformat_arguments = """
                            Ich gebe dir eine Übersicht von Pro- und Contra-Argumenten. 
                            Konvertiere diese Übersicht in eine Python-Liste mit einem Dictionary für jedes Argument,
                            sodass am Ende folgendener Output entstehen soll:
                            
                            [
                                {
                                    "argument": "Hier kommt das Argument.",
                                    "stance": "Erlaubt sind nur 'Pro' oder 'Contra', je nachdem ob das Argument Pro oder Contra ist."
                                },
                                {
                                    "argument": "Hier kommt das Argument.",
                                    "stance": "Erlaubt sind nur 'Pro' oder 'Contra', je nachdem ob das Argument Pro oder Contra ist."
                                }
                            ] 
                            
                            Gibt nur die Liste zurück. Sag sonst nichts weiter.
                            
                            Hier ist die Übersicht:
                            
                            """

segment_argument_text = """
                        Ich gebe dir einen Quellentext, in dem Pro- und Contra-Argumente rund ums Thema aufgelistet sein können.
                        Teile den Quellentext in kohärente Textabschnitte auf, die Argumente zu einem bestimmten Aspekt enthalten.
                        Strukturiere dabei eine JSON, die wie folgt aussehen soll:
                        
                        {example_json_argument_segments}
                        
                        # Hier ist der Quellentext mit den Argumenten:
                        
                        {argument_text}
                        
                    """

categorize_argument_zero_shot = """
            Du bist ein Experte in Sachen Argumentanalyse.
            Ich gebe dir nun einen Textauszug, der ein Thema beschreibt.
            Danach gebe ich dir einen Textauszug, der eventuell ein Argument zum Thema darstellt.
            Du sollst nur sagen, ob der Textauszug ein Pro- oder ein Contra-Argument darstellt oder keins von beiden.
            Du darfst also nur eines von folgenden Labels zurückgeben: Pro, Contra, Neutral.
            
            # Das Thema lautet: {topic_text}
            
            # Der Textauszug lautet: {text_sample}
            
            Das Label lautet also: 
            """

categorize_argument_zero_shot_cot = """
            Du bist ein Experte in Sachen Argumentanalyse.
            Ich gebe dir nun einen Textauszug, der ein Thema beschreibt.
            Danach gebe ich dir einen Textauszug, der eventuell ein Argument zum Thema darstellt.
            Du sollst entscheiden, ob der Textauszug ein explizites Pro- oder ein Contra-Argument darstellt oder keins von beiden (neutral).
            Du darfst also nur eines von folgenden Labels zurückgeben: Pro, Contra, Neutral.
            Bevor du das Label angibst, musst du immer vorher eine eine Begründung für deine Entscheidung liefern.
            Also erst die Begründung und darauf aufbauend das Label angeben.
            Wichtig ist, dass du deine Entscheidung nur auf explizite Angaben im Textauszug fällst und nicht zu viel interpretierst. 
            
            Hier einige Beispiele:

            # Das Thema lautet: Um die Umwelt für kommende Generationen zu bewahren, ist der Schutz natürlicher Ressourcen von zentraler Bedeutung.

            # Der Textauszug lautet: Luisa Neubauer, eine bekannte Aktivistin, erklärte kürzlich, dass nur ein radikales Umdenken beim Umweltschutz den fortschreitenden Klimawandel aufhalten kann. Strengere Gesetze sind notwendig, um das Gleichgewicht der Natur zu bewahren.

            # Begründung: Das Argument plädiert für strengere Umweltgesetze und radikale Maßnahmen zum Schutz der Natur, was mit einer 'pro'-Position im Umweltschutz übereinstimmt.

            # Anwort : Pro

            # Das Thema lautet: Die Legalisierung der Sterbehilfe gibt Menschen in schwierigen Lebenslagen die Möglichkeit, selbstbestimmt über das Ende ihres Lebens zu entscheiden.

            # Der Textauszug lautet: Ein Ethikprofessor warnte davor, dass die Legalisierung von Sterbehilfe dazu führen könnte, dass vulnerable Menschen sich unter Druck gesetzt fühlen, ihr Leben vorzeitig zu beenden, um ihren Angehörigen nicht zur Last zu fallen.

            # Begründung: Das Argument weist auf den potenziellen Druck auf schutzbedürftige Personen hin, was es zu einem 'contra'-Argument gegen die Legalisierung der Sterbehilfe macht.

            # Anwort : Contra

            # Das Thema lautet: Die Legalisierung der Sterbehilfe gibt Menschen in schwierigen Lebenslagen die Möglichkeit, selbstbestimmt über das Ende ihres Lebens zu entscheiden.

            # Der Textauszug lautet: Viele Menschen, die mit unheilbaren Krankheiten konfrontiert sind, wünschen sich Zugang zu alternativen Formen der medizinischen Betreuung, bevor sie über Sterbehilfe nachdenken.

            # Begründung: Das Argument weist auf die Komplexität von Entscheidungen am Lebensende hin, ohne eine starke Position zur Sterbehilfe zu beziehen, weshalb es 'neutral' ist.

            # Anwort : Neutral
            
            # Das Thema lautet:  In den Diskussionen über die Finanzierung des Bürgergeldes gab es Streitpunkte hinsichtlich der Dauer des staatlichen Kostenaufwands und der Zuständigkeit.

            # Der Textauszug lautet:  Es gibt Meinungsverschiedenheiten über die Dauer der Ausnahmegesetze fürs Bürgergeld.

            # Begründung: Das Argument ist kein Argument als solches, da es lediglich den Sachverhalt reproduziert ohne irgendwelche neuen Erkenntnisse oder Begründungen zu liefern.
            
            # Anwort : Neutral
            
            Jetzt bist du dran.

            # Das Thema lautet: {topic_text}

            # Der Textauszug lautet: {text_sample}

            """

categorize_argument_few_shot = """
            Aufgabe: Basierend auf dem angegebenen Thema und dem Argument klassifiziere, ob das Argument pro (unterstützt das Thema), contra (lehnt das Thema ab) oder neutral (nicht relevant zum Thema) ist.

            Zunächst gebe ich dir einige Beispiele, damit du siehst, wie man es machen muss:

            """
with open(config["paths"]["datasets"] / "fewshot_examples_chatgpt.json", "r") as f:
    few_shot_examples = js.load(f)

random.shuffle(few_shot_examples)

for example in few_shot_examples:
    few_shot_example = f"""
                # Thema: {example["thema"]}
                # Argument: {example["argument"]}
                # Label: {example["label"]}
                
                """
    categorize_argument_few_shot += few_shot_example

few_shot_example = """ Hier ist nun ein Beispiel, um selbst zu entscheiden. Füge nur das Label ein:
    
                # Thema: {topic_text}
                # Argument: {text_sample}
                # Label: 

                """
categorize_argument_few_shot += few_shot_example

is_argument = """
            Du bist ein Experte in Sachen Argumentanalyse.
            # Das Thema lautet: {topic_text}

            # Hier ist ein Auszug aus einem Text: {text_sample}

            Wenn der Textauszug die Aussage im Thema befürwortet, sag "Ja".
            Wenn der Textauszug der Aussage widerspricht, sag "Ja".
            Wenn keins von beidem zutrifft, sag "Nein".
            Sag "Ja" oder "Nein".
            Sag sonst nichts weiter.
            Das label lautet also: 
            """

is_debatable = """Du bist ein Experte in Sachen Argumentanalyse.
                Ich gebe dir nun einen Textauszug und du bist entscheiden, ob der Inhalt im Textauszug debattiertbar ist 
                oder einfach nur eine Tatsache darstellt, für die es keine Pro- oder Contra-Argumente gibt.
                Wenn der Textauszug debattierbar ist, sag "Ja".
                Wenn der Textauszug einfach eine Tatsache ist, sag "Nein".
                Sag sonst nichts weiter.
                
                # Der Textauszug lautet: {text}           
            """

make_sentence_concrete = """
                    Du bist ein Kommunikatonsexperte.
                    Ich gebe dir einen Satz und einen Kontext, in dem der Satz vorkommt.
                    Schreibe den Satz so um, dass er auch ohne den Kontext zu verstehen ist.
                    Achte dabei, dass du wirklich für alle Begriffe oder Phrasen, die man nicht durch Allgemeinwissen kennt, 
                    eine kurze Erklärung oder Umschreibung beifügst. 
                    Beispielsweise, wenn über ein Gesetz gesprochen wird, schau im Kontext nach, was genau mit dem Gesetz gemeint ist.
                    Weitere Wörter, die man definitiv erklären muss, sind: Pronomen, {ambiguous_words}.  
                    Wenn der Kontext keine Erklärung hergibt, muss du diese Begriffe nicht umschreiben.
                    Du darfst also nichts hinzudichten.
                    
                    Wichtig: Gib nur den reformulierten Satz zurück und sag sonst nichts weiter!

                    # Hier ist der Satz:
                    {sentence}

                    # Hier ist der Kontext:
                    {context}

                    # Der neue Satz lautet:

                    """

make_sentence_concrete_1 = """
                    Du bist ein Kommunikatonsexperte.
                    Ich gebe dir einen Satz und einen Kontext, in dem der Satz vorkommt.
                    Schreibe den Satz so um, dass er auch ohne den Kontext zu verstehen ist.
                    Achte dabei, dass du wirklich für alle Begriffe oder Phrasen, die man nicht durch Allgemeinwissen kennt, 
                    eine kurze Erklärung oder Umschreibung beifügst. 
                    Beispielsweise, wenn irgendwo von 'Gesetz', 'Gesetzesänderung', 'Bestimmung', 'Versammlung' etc. gesprochen wird, 
                    schau im Kontext nach, was genau damit gemeint ist und füge zusätzliche Beschreibungen hinzu,
                    damit der Leser auch versteht, was genau damit gemeint ist.
                    Weitere Wörter, die man definitiv disambiguieren muss, sind: Pronomen, {ambiguous_words}.  
                    Wenn der Kontext keine Erklärung hergibt, muss du diese Begriffe nicht umschreiben.
                    Du darfst also nichts hinzudichten.
                    Deine Umschreibung darf aber ruhig mehrere Sätze umfassen.
                    
                    Wichtig: Gib nur den reformulierten Satz zurück und sag sonst nichts weiter!

                    # Hier ist der Satz:
                    {sentence}

                    # Hier ist der Kontext:
                    {context}

                    # Der neue Satz lautet:

                    """

make_sentence_concrete_2 = """
                    Du bist ein Kommunikatonsexperte.
                    Ich gebe dir einen Satz. Der Satz hat folgende Wörter, die ambig sind: {ambiguous_words}.
                    Anschließend gebe ich dir einen Textauszug, der den Kontext darstellt, in dem der Satz vorkommt.
                    Setze anstelle der ambigen Wörter Umschreibungen basierend auf den Kontext ein, damit man den Satz in Isolation besser versteht.
                    Wichtig: Gib nur den reformulierten Satz zurück und sag sonst nichts weiter!

                    # Hier ist der Satz:
                    {sentence}

                    # Hier ist der Kontext:
                    {context}

                    # Der neue Satz lautet:

                    """
find_main_points = """
                    Du bist ein Experte in Sachen Textanalyse.
                    Ich gebe dir einen Textauszug.
                    Liste die wesentliche Themenschwerpunkte in dem Textauszug in Bulletpoints auf.
                    
                    # Hier ist der Textauszug:
                    {text}
                    """

find_reasoning_1 = """
                Es geht um folgende Grundaussage:
                
                {topic}
                
                Ein Argumentanalyse hat ergeben, dass folgender Auszug ein {stance}-Haltung zu dieser Grundaussage darstellt:
                
                {claim}
                
                Ich gebe dir nun einen breiten Kontext für den genannten Auszug.
                Extrahiere aus dem Kontext die Begründung, welche die {stance}-Haltung zur Grundaussage untermauert.
                Sofern du eine Begründung finden kannst, extrahiere sie einfach nur und verändere nicht den Wortlaut.
                Wichtig: Extrahiere nur eine Begründung, sofern der Kontext auch wirklich eine Begründung für die {stance}-Haltung liefert.
                Wenn du keine Begründung finden kannst, sag einfach: Keine Begründung.
                
                Hier ist der Kontext:
                
                {context}
                """

find_reasoning_2 = """
                Ich gebe dir folgenden Auszug:

                {claim}

                Nachfolgend gebe ich dir den bereiten Kontext, in dem der Auszug vorkommt:
                
                {context}
                
                
                Falls der Kontext einen Abschnitt mit einer Begründung für die Aussage im Auszug enthält, extrahiere den
                Abschnitt und formuliere die Begründung kurz und prägnant in eigenen Worten. Strukturiere deine Antwort als JSON:
                
                {{
                    "reasoning_segment":"Extrahierter Abschnitt mit der Begründung.", 
                    "reasoning":"Begründung in eigenen Worten."
                }} 
                
                Wenn du keinerlei Begründugn findest, gib einfach leere Strings zurück, wie folgt:
                
                {{
                    "reasoning_segment":"", 
                    "reasoning":""
                }}
                
                """

find_reasoning_3 = """
                Du bist ein Experte in Sachen Argumentanalyse.
                
                Ich gebe dir folgende Grundaussage:

                {topic}
                
                Ein Argumentanalyse hat ergeben, dass folgender Auszug ein {stance}-Haltung zu dieser Grundaussage darstellt:
                
                {claim}

                Nachfolgend gebe ich dir den bereiten Kontext, in dem der Auszug vorkommt:

                {context}
                
                
                Extrahiere aus dem Kontext die Begründung, welche die {stance}-Haltung zur Grundaussage untermauert 
                und formuliere die Begründung prägnant in eigenen Worten. Strukturiere deine Antwort als JSON:

                {{
                    "reasoning_segment":"Extrahierter Abschnitt mit der Begründung.", 
                    "reasoning":"Begründung in eigenen Worten."
                }}
                
                Wichtig: Extrahiere nur eine Begründung, sofern der Kontext auch wirklich eine Begründung für die {stance}-Haltung liefert.

                Wenn du keinerlei Begründugn findest, gib einfach leere Strings zurück, wie folgt:

                {{
                    "reasoning_segment":"", 
                    "reasoning":""
                }}

                """

find_reasoning_4 = """
                Du bist ein Experte in Sachen Argumentanalyse.

                Ich gebe dir folgende Grundaussage:

                {topic}

                Nachfolgend gebe ich dir einen bereiten Kontext:

                {context}

                Suche und extrahiere einen Abschnitt im Kontext, welcher ein sehr gute Begründung für eine 
                {stance}-Haltung zur vorhin genannten Grundaussage liefert 
                und formuliere diese Begründung prägnant in eigenen Worten. 
                Strukturiere deine Antwort als JSON:

                {{
                    "reasoning_segment":"Extrahierter Abschnitt mit der Begründung.", 
                    "reasoning":"Begründung in eigenen Worten."
                }}

                Wichtig: Extrahiere nur eine Begründung, sofern der Kontext auch wirklich eine gute und explizite Begründung für eine {stance}-Haltung gegenüber der Grundaussage liefert.

                Wenn du keinerlei Begründugn findest, gib einfach leere Strings zurück, wie folgt:

                {{
                    "reasoning_segment":"", 
                    "reasoning":""
                }}

                """

extract_person = """
                Ich gebe dir einen Satz sowie den Kontext, in dem der Satz vorkommt.
                Basierend auf den Satz und dem Kontext fülle folgende JSON aus:
                
                {{
                    "person": "Person, von der der Satz kommt.",
                    "party": "Partei, die die Person angehört.",
                    "canton": "Partei, die die Person angehört."
                }}
                
                Solltest du zu den Feldern in der JSON keine Informationen finden, 
                trage einfach leere Werte ein.
                
                Wichtig: Gibt nur die JSON zurück und sag nichts weiter!
                
                # Satz:
                {sentence}
                
                # Kontext:
                {context}
                """

prompt_dict = {
    "extract_arguments_1": extract_arguments_1,
    "example_json_arguments": example_json_arguments,
    "segment_argument_text": segment_argument_text,
    "example_json_argument_segments": example_json_argument_segments,
    "reformat_arguments": reformat_arguments,
    "categorize_argument_zero_shot": categorize_argument_zero_shot,
    "categorize_argument_few_shot": categorize_argument_few_shot,
    "categorize_argument_zero_shot_cot": categorize_argument_zero_shot_cot,
    "is_argument": is_argument,
    "make_sentence_concrete": make_sentence_concrete,
    "make_sentence_concrete_1": make_sentence_concrete_1,
    "make_sentence_concrete_2": make_sentence_concrete_2,
    "find_main_points": find_main_points,
    "is_debatable": is_debatable,
    "find_reasoning_1": find_reasoning_1,
    "find_reasoning_2": find_reasoning_2,
    "find_reasoning_3": find_reasoning_3,
    "find_reasoning_4": find_reasoning_4,
    "extract_person": extract_person,
}
if __name__ == "__main__":
    print(prompt_dict["categorize_argument_few_shot"])
