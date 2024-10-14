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

categorize_argument_few_shot = """
            Aufgabe: Basierend auf dem angegebenen Thema und dem Argument klassifiziere, ob das Argument pro (unterstützt das Thema), contra (lehnt das Thema ab) oder neutral (nicht relevant zum Thema) ist.
            
            Zunächst gebe ich dir einige Beispiele, damit du siehst, wie man es machen muss:
            
            """
with open(config["paths"]["datasets"] / "fewshot_examples_chatgpt.json", "r") as f:
    few_shot_examples = js.load(f)

random.shuffle(few_shot_examples)

for example in few_shot_examples:
    few_shot_example = (
        f"""
                # Thema: {example["thema"]}
                # Argument: {example["argument"]}
                # Label: {example["label"]}
                
                """
    )
    categorize_argument_few_shot += few_shot_example

few_shot_example = (
    """ Hier ist nun ein Beispiel, um selbst zu entscheiden. Füge nur das Label ein:
    
                # Thema: {topic_text}
                # Argument: {text_sample}
                # Label: 

                """
)
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

expand_sentence = """
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

prompt_dict = {
    "extract_arguments_1": extract_arguments_1,
    "example_json_arguments": example_json_arguments,
    "segment_argument_text": segment_argument_text,
    "example_json_argument_segments": example_json_argument_segments,
    "reformat_arguments": reformat_arguments,
    "categorize_argument_zero_shot": categorize_argument_zero_shot,
    "categorize_argument_few_shot": categorize_argument_few_shot,
    "is_argument": is_argument,
    "expand_sentence": expand_sentence,
    "find_main_points": find_main_points,
    "is_debatable": is_debatable,
}
if __name__ == "__main__":

    print(prompt_dict["categorize_argument_few_shot"])
