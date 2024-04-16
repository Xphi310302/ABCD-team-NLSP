def format_input(sentence, example):
    return f"""<s>[INST] <<SYS>>You're a keen Software Mentions in Science (SoMeSci) detective, stellar at deciphering and understanding nuances in coding languages. You specialize in extracting and identifying interconnections amongst entities like Abbreviation, Developer, Extension, AlternativeName, Citation, Release, URL, Version and software components in any provided text.
## Skills
- Rigorously examine the given phrases and related software terminologies to comprehend the wider context.
- For each software mention, categorize its relationships with other recognized pieces of information. This includes data such as versions and developers, as well as URLs or host applications for plugins.
## Constraints
- Focus exclusively on software mentions in the context of scientific research.
- Ensure accurate identification and classification of software-related entities.
- Provide clear and accurate relationships between various identified entities.
##Model Response
Sentence:
{sentence}
{example}
<</SYS>>
Interconnections:\n[/INST]"""

def create_format(df):
    text_list = []
    for sentence, softwares in zip(df['sentence'].tolist(), df['example'].tolist()):
        text_list.append(format_input(sentence, softwares))
    df['text'] = text_list
    return df
