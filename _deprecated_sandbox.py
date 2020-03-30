# # def computeMarginal(X_train, X_test, variable_set = {'x2'}, conditioning_set = {'x3'}):
# #   numerator = getJoint(X_train, variable_set)
# #   denominator = getJoint(X_train, conditioning_set)

# #   numerator(1) / denominator(1)

# #   return lambda X_test: numerator(X_test) / denominator(X_test)

# def getJoint(X, variable_set = {'x2'}):
#   return KernelDensity(kernel = 'gaussian', bandwidth = .25).fit(X[list(variable_set)])
#   # TODO: use cross-validation:

#   # from sklearn.grid_search import GridSearchCV
#   # from sklearn.cross_validation import LeaveOneOut
#   # bandwidths = 10 ** np.linspace(-1, 1, 100)
#   # grid = GridSearchCV(KernelDensity(kernel='gaussian'),
#   #                     {'bandwidth': bandwidths},
#   #                     cv=LeaveOneOut(len(x)))
#   # grid.fit(x[:, None]);
#   # grid.best_params_


# def computeExpectation(X, variable_set = {'x1'}, conditioning_set = {'x2'}, intervening_set = {'x3'}):

#   marg_x1 = getJoint(X, {'x1'})
#   marg_x2 = getJoint(X, {'x2'})
#   marg_x3 = getJoint(X, {'x3'})
#   marg_x1_x2 = getJoint(X, {'x1', 'x2'})
#   marg_x1_x3 = getJoint(X, {'x1', 'x3'})
#   marg_x2_x3 = getJoint(X, {'x2', 'x3'})
#   marg_x1_x2_x3 = getJoint(X, {'x1', 'x2', 'x3'})

#   cond_x2_on_x1 = lambda x1, x2: \
#     marg_x1_x2.score_samples(np.array((x1, x2)).reshape(1,-1)) / \
#     marg_x1.score_samples(np.array((x1)).reshape(1,-1))

#   cond_x3_on_x1_x2 = lambda x1, x2, x3: \
#     marg_x1_x2_x3.score_samples(np.array((x1, x2, x3)).reshape(1,-1)) / \
#     marg_x1_x2.score_samples(np.array((x1, x2)).reshape(1,-1))


#   ipsh()


# def plotMarginals(X, marg_x1, marg_x2, marg_x3):
#   samples_x1 = np.unique(X['x1'])
#   samples_x1 = np.clip(samples_x1, -25, 25)
#   samples_x1 = samples_x1.reshape((len(samples_x1), 1))
#   probabs_x1 = np.exp(marg_x1.score_samples(samples_x1))
#   # print(f'E[x1] = {np.sum(np.multiply(samples_x1.T, probabs_x1), axis = 1) / probabs_x1.shape[0]}')
#   print(f'E[x1] = {np.sum(np.multiply(samples_x1.T, probabs_x1), axis = 1) / np.sum(probabs_x1)}')

#   samples_x2 = np.unique(X['x2'])
#   samples_x2 = np.clip(samples_x2, -25, 25)
#   samples_x2 = samples_x2.reshape((len(samples_x2), 1))
#   probabs_x2 = np.exp(marg_x2.score_samples(samples_x2))
#   # print(f'E[x2] = {np.sum(np.multiply(samples_x2.T, probabs_x2), axis = 1) / probabs_x2.shape[0]}')
#   print(f'E[x2] = {np.sum(np.multiply(samples_x2.T, probabs_x2), axis = 1) / np.sum(probabs_x2)}')

#   samples_x3 = np.unique(X['x3'])
#   samples_x3 = np.clip(samples_x3, -25, 25)
#   samples_x3 = samples_x3.reshape((len(samples_x3), 1))
#   probabs_x3 = np.exp(marg_x3.score_samples(samples_x3))
#   # print(f'E[x3] = {np.sum(np.multiply(samples_x3.T, probabs_x3), axis = 1) / probabs_x3.shape[0]}')
#   print(f'E[x3] = {np.sum(np.multiply(samples_x3.T, probabs_x3), axis = 1) / np.sum(probabs_x3)}')

#   pyplot.plot(samples_x1, probabs_x1)
#   pyplot.plot(samples_x2, probabs_x2)
#   pyplot.plot(samples_x3, probabs_x3)
#   pyplot.show()


