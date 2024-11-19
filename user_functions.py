# Import necessary libraries
import re  # For regular expression operations
from collections import Counter  # For counting occurrences of items
import spacy  # For NLP tasks (spaCy)
from nltk import pos_tag  # For part-of-speech tagging
from nltk.tokenize import word_tokenize  # For tokenizing sentences into words
from nltk.corpus import stopwords  # To filter out common words (stopwords)

# Load the English spaCy model for word vectors (word2vec)
word2vec = spacy.load('en')

# Set of stopwords (common words like "the", "a", etc. that are usually not useful in NLP)
stop_words = set(stopwords.words("english"))

# Function to preprocess the input sentence by normalizing, tokenizing, and removing stopwords
def preprocess(input_sentence):
    """
    Preprocess the input sentence:
    - Convert to lowercase
    - Remove punctuation
    - Tokenize the sentence into words
    - Remove stopwords
    """
    # Convert the input sentence to lowercase
    input_sentence = input_sentence.lower()
    
    # Remove punctuation using regular expressions (keeps only alphanumeric characters and spaces)
    input_sentence = re.sub(r'[^\w\s]', '', input_sentence)
    
    # Tokenize the sentence into individual words
    tokens = word_tokenize(input_sentence)
    
    # Remove stopwords from the tokenized words
    input_sentence = [i for i in tokens if not i in stop_words]
    
    # Return the list of filtered tokens
    return input_sentence

# Function to compare the overlap between user message tokens and possible response tokens
def compare_overlap(user_message, possible_response):
    """
    Compare the overlap of words between the user's message and a possible response.
    Returns the number of common words (similar words) between the two.
    """
    similar_words = 0
    
    # Iterate through each token in the user's message
    for token in user_message:
        # Check if the token is present in the possible response
        if token in possible_response:
            similar_words += 1  # Increase count if a similar word is found
    
    # Return the total count of similar words
    return similar_words

# Function to extract nouns from a part-of-speech tagged message
def extract_nouns(tagged_message):
    """
    Extracts all nouns from a part-of-speech tagged message.
    Returns a list of noun words (tokens with a POS tag starting with 'N').
    """
    message_nouns = list()
    
    # Iterate through the tagged message (list of (word, POS_tag) tuples)
    for token in tagged_message:
        # If the token's POS tag starts with 'N' (indicating it's a noun)
        if token[1].startswith("N"):
            message_nouns.append(token[0])  # Add the noun to the list
    
    # Return the list of extracted nouns
    return message_nouns

# Function to compute similarity between tokens and a given category using word vectors
def compute_similarity(tokens, category):
    """
    Compute the similarity between each token and a category (represented by a word vector).
    Returns a list of tuples containing the token, category, and their similarity score.
    """
    output_list = list()
    
    # Iterate through each token in the list of tokens
    for token in tokens:
        # Append the token, category, and their similarity score to the output list
        output_list.append([token.text, category.text, token.similarity(category)])
    
    # Return the list of similarity results
    return output_list
