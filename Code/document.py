import math


class Document:
    def __init__(self, name, words_vector, is_spam):
        self.name = name
        self.words_vector = words_vector
        self.is_spam = is_spam
        self.length_of_vector = self.calculate_length_of_vector()

    def calculate_length_of_vector(self):
        sum_of_squares = 0
        for word in self.words_vector:
            count = self.words_vector[word]
            sum_of_squares += (count * count)
        return math.sqrt(sum_of_squares)

    def dot_product(self, document):
        ans = 0
        for word in self.words_vector:
            ans += (self.words_vector[word] * document.words_vector[word])
        return ans

    def cosine_similarity(self, document):
        if self.length_of_vector == 0 or document.length_of_vector == 0:
            return -2
        return self.dot_product(document) / (self.length_of_vector * document.length_of_vector)
