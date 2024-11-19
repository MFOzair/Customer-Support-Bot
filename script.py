# Import necessary libraries
from collections import Counter
from responses import responses, blank_spot, menu  # Import predefined responses, blank spot, and menu
from user_functions import preprocess, compare_overlap, pos_tag, extract_nouns, compute_similarity
import spacy  # For NLP tasks

# Load the English language model for word2vec using spaCy
word2vec = spacy.load('en')

# Define a tuple containing exit commands
exit_commands = ("quit", "goodbye", "exit", "no")

# Define the ChatBot class
class ChatBot:
  
  # Method to check if the user message contains an exit command
  def make_exit(self, user_message):
    # Iterate through the list of exit commands
    for exit_command in exit_commands:
      # If the exit command is found in the user's message, print 'Goodbye' and return True
      if exit_command in user_message:
        print('Goodbye')
        return True
    # If no exit command is found, return False (continue chatting)
    return False

  # Main chat method to start the conversation with the user
  def chat(self):
    # Display the menu and wait for the user's message
    user_message = input(menu)
    
    # Continue the conversation until an exit command is detected
    while not self.make_exit(user_message):
      # Get a response based on the user's message and respond
      user_message = self.respond(user_message)

  # Method to find the most relevant response based on the user's message
  def find_intent_match(self, responses, user_message):
    # Preprocess the user's message and convert it to a bag-of-words (BoW) representation
    bow_user_message = Counter(preprocess(user_message))
    
    # Create BoW representations for each predefined response
    bow_responses = [Counter(preprocess(response)) for response in responses]
    
    # Compute the similarity between the user's message and each response
    similarity_list = [compare_overlap(bre, user_message) for bre in bow_responses]
    
    # Find the index of the response with the highest similarity
    response_index = similarity_list.index(max(similarity_list))
    
    # Return the response with the highest similarity
    return responses[response_index]
    
  # Method to extract entities (keywords) from the user's message
  def find_entities(self, user_message):
    # Perform part-of-speech tagging on the preprocessed user message
    tagged_user_message = pos_tag(preprocess(user_message))
    
    # Extract the nouns (potential entities) from the tagged message
    message_noun = extract_nouns(tagged_user_message)
    
    # Use word2vec (spaCy model) to get the vector representation of the extracted nouns
    tokens = word2vec(" ".join(message_noun))
    
    # Get the vector representation of the 'blank_spot' category
    category = word2vec(blank_spot)
    
    # Compute the similarity between the user's message tokens and the blank spot category
    word2vec_result = compute_similarity(tokens, category)
    
    # Sort the similarity results by score (similarity) in ascending order
    word2vec_result.sort(key=lambda x: x[2])
    
    # If no relevant entity is found, return 'blank_spot'
    if len(word2vec_result) < 1:
      return blank_spot
    else:
      # Return the entity with the highest similarity score
      return word2vec_result[-1][0]

  # Method to generate a response based on the user's message
  def respond(self, user_message):
    # Find the most relevant response based on intent
    best_response = self.find_intent_match(responses, user_message)
    
    # Extract entities (keywords) from the user's message
    entity = self.find_entities(user_message)
    
    # Format the response with the extracted entity and prompt the user for the next input
    input_message = input(best_response.format(entity))
    
    # Return the next user input (to continue the conversation)
    return input_message

# Initialize the ChatBot instance
restaurannt_bot = ChatBot()

# Start the chat interaction by calling the .chat() method
restaurannt_bot.chat()
