"""
Given an input text, this pipeline will:
1. Create a constituency parse tree
2. Generate a graph embedding of the parse tree
3. Concatenate the graph embedding with the input text embedding
4. Pass the concatenated embedding into a decoder to generate a response
5. Return the response
6. Store the embedding of the response and the graph embedding for the reply
7. Repeat steps 1-6 for the reply
"""

from con_parser.model import 
