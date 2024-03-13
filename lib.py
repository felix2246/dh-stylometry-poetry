import string

import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')


def clean_text(text: str) -> str:
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha() and word not in set(stopwords.words('english'))]
    text = ' '.join(tokens)
    text = text.lower()
    return text


def verb_ratio(text: str) -> float:
    tokens = word_tokenize(text)
    if len(tokens) == 0:
        return 0
    tagged_tokens = pos_tag(tokens)
    verb_count = sum(1 for word, tag in tagged_tokens if tag.startswith('VB'))
    return verb_count / len(tokens)


def noun_ratio(text: str) -> float:
    tokens = word_tokenize(text)
    if len(tokens) == 0:
        return 0
    tagged_tokens = pos_tag(tokens)
    noun_count = sum(1 for word, tag in tagged_tokens if tag.startswith('NN'))
    return noun_count / len(tokens)


def adjective_ratio(text: str) -> float:
    tokens = word_tokenize(text)
    if len(tokens) == 0:
        return 0
    tagged_tokens = pos_tag(tokens)
    adjective_count = sum(1 for word, tag in tagged_tokens if tag.startswith('JJ'))
    return adjective_count / len(tokens)


def average_verse_length(text: str) -> float:
    if verse_count(text) == 0:
        return 0
    return sum([len(verse) for verse in text.split("\n")]) / verse_count(text)


def verse_count(text: str) -> int:
    return len(text.split("\n"))


def average_word_length(text: str) -> float:
    if len(word_tokenize(text)) == 0:
        return 0
    return sum([len(word) for word in word_tokenize(text)]) / len(word_tokenize(text))


def alliterations_ratio(text: str) -> float:
    verses = text.split("\n")
    alliteration_count = 0

    if len(word_tokenize(text)) == 0:
        return 0

    for verse in verses:
        words = word_tokenize(verse)
        words = [word.lower() for word in words if word.lower() not in set(stopwords.words('english'))]
        previous_sound = None
        for word in words:
            if word[0] == previous_sound:
                alliteration_count += 1
            previous_sound = word[0]

    return alliteration_count / len(word_tokenize(text))


def punctuation_ratio(text: str) -> float:
    if len(word_tokenize(text)) == 0:
        return 0
    return sum([1 for char in text if char in set(".,;:!?")]) / len(word_tokenize(text))
