import os

BASE_PATH = os.path.dirname(__name__)
lines_path = os.path.join(BASE_PATH, "data/movie_lines.txt")
conv_path = os.path.join(BASE_PATH, "data/movie_conversations.txt")

dictionary_length = 10000
max_length = 20
min_line_length = 3

epoch = 5
step = 2048
batch_size = 128

clear_model = False
padding_type = 'post'
truncating_type = 'post'