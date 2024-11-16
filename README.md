
# Text Processing and Understanding

This repository demonstrates various techniques for processing and understanding text data. The goal is to provide a comprehensive text normalization pipeline that includes:

- **Removing HTML tags**
- **Tokenization**
- **Removing unnecessary tokens**
- **Stopword handling**
- **Handling contractions**
- **Correcting spelling errors**
- **Stemming and Lemmatization**
- **Part-of-Speech tagging, Chunking, and Parsing**
- **Building a Text Normalizer**
- **Understanding Text Syntax and Structure**
- **Building Dependency Parsing**

This project includes a step-by-step guide with code implementation and a sample dataset.

## Features

- Text cleaning and preprocessing pipeline
- Advanced NLP techniques including dependency parsing
- Easy-to-use Python implementation with comments and explanations
- Example dataset to test and demonstrate the techniques

## Prerequisites

Before running the code, make sure to install the necessary libraries:

```bash
pip install nltk beautifulsoup4 spacy pyspellchecker
```

You will also need to download some NLTK datasets:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('universal_tagset')
```

## Dataset

For demonstration purposes, this repository uses a text dataset containing a variety of content. Feel free to replace it with your own dataset as needed.

```plaintext
Dataset file: `text_data.csv` containing a column 'text' with raw textual content.
```

## Example Code Implementation

### Step 1: Removing HTML Tags

To clean up HTML tags from raw text:

```python
from bs4 import BeautifulSoup

def remove_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

raw_text = "<html><body><h1>Hello World!</h1></body></html>"
clean_text = remove_html_tags(raw_text)
print(clean_text)  # Output: Hello World!
```

### Step 2: Tokenization

Tokenizing the text into individual words:

```python
import nltk
nltk.download('punkt')

from nltk.tokenize import word_tokenize

def tokenize_text(text):
    return word_tokenize(text)

text = "This is an example sentence."
tokens = tokenize_text(text)
print(tokens)  # Output: ['This', 'is', 'an', 'example', 'sentence', '.']
```

### Step 3: Removing Unnecessary Tokens

Remove tokens like punctuation and short words:

```python
import string

def remove_unnecessary_tokens(tokens):
    return [word for word in tokens if word not in string.punctuation and len(word) > 2]

clean_tokens = remove_unnecessary_tokens(tokens)
print(clean_tokens)  # Output: ['This', 'example', 'sentence']
```

### Step 4: Stopword Handling

Removing common stopwords like "the", "is", etc.:

```python
from nltk.corpus import stopwords

nltk.download('stopwords')

def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word.lower() not in stop_words]

tokens_no_stopwords = remove_stopwords(clean_tokens)
print(tokens_no_stopwords)  # Output: ['example', 'sentence']
```

### Step 5: Handling Contractions

Expanding common contractions (e.g., "isn't" to "is not"):

```python
import contractions

def expand_contractions(text):
    return contractions.fix(text)

expanded_text = expand_contractions("I can't do this.")
print(expanded_text)  # Output: I cannot do this.
```

### Step 6: Correcting Spelling Errors

Using a spell checker to correct typos:

```python
from spellchecker import SpellChecker

def correct_spelling(text):
    spell = SpellChecker()
    corrected = [spell.correction(word) for word in text.split()]
    return ' '.join(corrected)

text_with_typos = "I have a speling mistake."
corrected_text = correct_spelling(text_with_typos)
print(corrected_text)  # Output: I have a spelling mistake.
```

### Step 7: Stemming and Lemmatization

Reducing words to their base form:

```python
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('wordnet')

# Stemming
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in tokens_no_stopwords]
print(stemmed_words)  # Output: ['exampl', 'sentenc']

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in tokens_no_stopwords]
print(lemmatized_words)  # Output: ['example', 'sentence']
```

### Step 8: POS Tagging, Chunking, and Parsing

Extracting grammatical structures:

```python
from nltk import pos_tag, ne_chunk

def pos_tagging(text):
    tokens = word_tokenize(text)
    return pos_tag(tokens)

tagged_text = pos_tagging("The quick brown fox jumped over the lazy dog.")
print(tagged_text)  # Output: [('The', 'DT'), ('quick', 'JJ'), ('brown', 'JJ'), ...]

def chunking(tagged_tokens):
    chunked = ne_chunk(tagged_tokens)
    return chunked

chunked_text = chunking(tagged_text)
print(chunked_text)  # Output: (S (NP The/DT quick/JJ brown/JJ fox/NN) jumped/VBD ...)
```

### Step 9: Building a Text Normalizer

Bringing it all together:

```python
def normalize_text(text):
    text = remove_html_tags(text)
    tokens = tokenize_text(text)
    tokens = remove_unnecessary_tokens(tokens)
    tokens = remove_stopwords(tokens)
    text = expand_contractions(' '.join(tokens))
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

raw_text = "<html>This is an example sentence, I can't do this!</html>"
normalized_text = normalize_text(raw_text)
print(normalized_text)  # Output: example sentence I cannot do
```

### Step 10: Dependency Parsing

Parsing the syntactic structure of sentences:

```python
import spacy

# Load the spacy model
nlp = spacy.load("en_core_web_sm")

def dependency_parsing(text):
    doc = nlp(text)
    for token in doc:
        print(token.text, token.dep_, token.head.text)

text = "The quick brown fox jumps over the lazy dog."
dependency_parsing(text)
```

## Conclusion

This repository showcases a full NLP pipeline with all the preprocessing steps necessary for building a robust text normalization and understanding system. You can use these techniques as building blocks for more advanced text analysis, sentiment analysis, and other NLP applications.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

