import sys
sys.path.append("..")
from dataset import ptb

corpus, word_to_id, id_to_word = ptb.load_data("train")

print(word_to_id["lexus"])
