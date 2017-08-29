# load ascii text and covert to lowercase
filename = "alice-in-wonderland.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()

# create mapping of unique chars to integers
chars = sorted(list(set(raw_text))) # all chars
char_to_int = dict((c, i) for i, c in enumerate(chars)) # map keys => chars, values => ints

n_chars = len(raw_text)
n_vocab = len(chars)
print "Total Characters: ", n_chars
print "Total Vocab: ", n_vocab

# prepare the dataset of input to output pairs encoded as integers
# each pattern is comprised of 100 characters and output is the next character
# input = 100 chars, output = 1 char
seq_length = 100
dataX = []
dataY = []

for i in range(0, n_chars - seq_length, 1):
  seq_in = raw_text[i:i + seq_length]
  seq_out = raw_text[i + seq_length]
  dataX.append([char_to_int[char] for char in seq_in])
  dataY.append(char_to_int[seq_out])

n_patterns = len(dataX)
print "Total Patterns: ", n_patterns
