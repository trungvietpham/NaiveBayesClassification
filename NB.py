import numpy as np
class GaussianNaiveBayes():
  def __init__(self, X, y):
    self.num_example, self.num_feature = X.shape
    self.num_class = len(np.unique(y))
    self.offset = 1e-6


  def fit(self, X, y):
    self.classes_mean = {}
    self.classes_variance = {}
    self.classes_prior = {}

    for c in range(self.num_class):
      X_c = X[y==c]
      self.classes_mean[str(c)] = np.mean(X_c, axis=0)
      self.classes_variance[str(c)] = np.var(X_c, axis=0)
      self.classes_prior[str(c)] = X_c.shape[0]/self.num_example


  def predict(self, X):
    prob = np.zeros((X.shape[0], self.num_class))

    for c in range(self.num_class):
      prior = self.classes_prior[str(c)]
      prob_c = self.density_function(X, self.classes_mean[str(c)], self.classes_variance[str(c)])
      prob[:, c] = prob_c + np.log(prior)

    return np.argmax(prob, axis=1)

  def density_function(self, x, mean, sigma):
    const = -self.num_feature/2 * np.log(2*np.pi)- np.sum(np.log(sigma+self.offset))
    probs = - np.sum(np.power(x-mean, 2)/(2*pow(sigma+self.offset, 2)), 1)
    return const + probs

class MultinomialNaiveBayes():
  def __init__(self, X,y):
    self.num_example = X.shape[0]
    self.num_classes = len(np.unique(y))

  def fit(self, X, y):
    self.classes_num_of_occur = {}
    self.classes_prior = {}

    #lấy ra mảng các example có label là c
    for c in range(self.num_classes):
      X_c = X[y==c]
      self.classes_num_of_occur[str(c)] = np.sum(X_c, axis=0)
      self.classes_num_of_occur[str(c)] = self.classes_num_of_occur[str(c)] + 1
      self.classes_prior[str(c)] = (X_c.shape[0])/self.num_example

  def predict(self, X):
    #P(ci|x) ~ P(x|ci)*P(ci) 
    #Lấy log: 
    #log(P(ci|x)) ~ log(P(ci)) + E[feature[j]*log(P(xj|ci))]
    prob = np.zeros((X.shape[0], self.num_classes))

    for c in range(self.num_classes):
      prior = self.classes_prior[str(c)]
      frequence = np.array(self.classes_num_of_occur[str(c)])
      omega = np.sum(frequence)
      for index in range(X.shape[0]):
        prob_i = 0
        prob_i = np.sum(X[index]*np.log(frequence/omega))
        
        prob[index][c] = prob_i + np.log(prior)
    return np.argmax(prob, axis=1)

