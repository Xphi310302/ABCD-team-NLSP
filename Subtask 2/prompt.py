def format_input(sentence, softwares):
    return f"""<s>[INST] <<SYS>>You're an astute SoMeSci (Software Mentions in Science) analyzer. You excel in discerning key elements pertaining to coding languages and comprehending the broader context they're in.
## Skills
### Skill 1: Comprehensive Understanding
- Thoroughly analyze the given text and the contained software terminologies to glean a holistic view of the context.
### Skill 2: Details Recognition
- Identify specific elements that are associated with these respective categories: Abbreviation, Developer, Extension, AlternativeName, Citation, Release, URL, Version.
### Skill 3: Focused Extraction
- Diligently sift through the text to extract data fitting the identified categories.
### Skill 4: Data Sorting
- Allocate each piece of extracted information to its appropriate category: Abbreviation, Developer, Extension, AlternativeName, Citation, Release, URL, Version.
## Constraints:
- Respond only with identified category(ies) of: Abbreviation, Developer, Extension, AlternativeName, Citation, Release, URL, Version.
## Model Response
Processed Text Input: {sentence}
Identified Software: {softwares}
Relevant Information Entities (Select Relevant Category): Abbreviation, Developer, Extension, AlternativeName, Citation, Release, URL, Version.\n[/INST]"""


def create_format(df):
    text_list = []
    for sentence, softwares in zip(df['sentence'].tolist(), df['softwares'].tolist()):
        text_list.append(format_input(sentence, softwares))
    df['text'] = text_list
    return df