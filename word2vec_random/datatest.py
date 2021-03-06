from gensim.models import word2vec

model = word2vec.Word2Vec()

model = word2vec.Word2Vec.load("classifier/Word2VectforNLPTraining2")

print(type(model.syn0))

# number of words, number of features
print(model.syn0.shape)

# access individual word vector
# returns a 1 * # of features numpy array
print(model["man"])

# doesnt_match function tries to deduce which word in a set is most
# dissimilar from the others
print(model.doesnt_match("man woman child kitchen".split()) + '\n')
print(model.doesnt_match("paris berlin london austria".split()) + '\n')

# most_similar(): returns the score of the most similar words based on the criteria
# Find the top-N most similar words. Positive words contribute positively towards the
# similarity, negative words negatively.

print("most similar:")
print(model.most_similar(positive=['woman', 'boy'], negative=['man'], topn=10))
print()
print(model.most_similar("garbage"))
