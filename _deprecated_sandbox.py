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




# x = torch.randn(3,2)
# x.requires_grad_(True)
# t = [tmp + 1 for tmp in x]
# t[0].sum().backward()
# print(x.grad)



# a = torch.tensor(2.)
# a.requires_grad_(True)
# b = a + 10
# c = a * 10
# d = torch.stack((b,c))


# a.grad = None
# b.backward()
# print(a.grad) # expect 1

# a.grad = None
# c.backward()
# print(a.grad) # expect 10

# a.grad = None
# torch.mean(d).backward() # expect 5.5
# print(a.grad)







# def getCounterfactualTemplate(dataset_obj, classifier_obj, causal_model_obj, factual_instance, action_set):

#   add if not bool action_set like below

#   counterfactual_template = dict.fromkeys(
#     dataset_obj.getInputAttributeNames(),
#     np.NaN,
#   )

#   # get intervention and conditioning sets
#   intervention_set = set(action_set.keys())

#   # intersection_of_non_descendents_of_intervened_upon_variables
#   conditioning_set = set.intersection(*[
#     causal_model_obj.getNonDescendentsForNode(node)
#     for node in intervention_set
#   ])

#   # assert there is no intersection
#   assert set.intersection(intervention_set, conditioning_set) == set()

#   # set values in intervention and conditioning sets
#   for node in conditioning_set:
#     counterfactual_template[node] = factual_instance[node]

#   for node in intervention_set:
#     counterfactual_template[node] = action_set[node]

#   return counterfactual_template

# def getSamplesDataFrame(dataset_obj, classifier_obj, causal_model_obj, factual_instance, action_set, num_samples):
#   if not bool(action_set): # if action_set is empty, CFE = F
#     return pd.DataFrame(dict(zip(
#       dataset_obj.getInputAttributeNames(),
#       [num_samples * [factual_instance[node]] for node in dataset_obj.getInputAttributeNames()],
#     )))

#   counterfactual_template = getCounterfactualTemplate(dataset_obj, classifier_obj, causal_model_obj, factual_instance, action_set)

#   # this dataframe has populated columns set to intervention or conditioning values
#   # and has NaN columns that will be set accordingly.
#   samples_df = pd.DataFrame(dict(zip(
#     dataset_obj.getInputAttributeNames(),
#     [num_samples * [counterfactual_template[node]] for node in dataset_obj.getInputAttributeNames()],
#   )))

#   return samples_df



# getStructuralEquation
# getConditionalDensity










# tmp_df = {}
#   tmp_df['recourse_type'] = []
#   tmp_df['value_x1'] = []
#   tmp_df['sample_x2'] = []
#   for k1,v1 in per_value_x1_results.items():
#     for k2,v2 in v1.items():
#       for elem in v2:
#         tmp_df['recourse_type'].append(k2)
#         tmp_df['value_x1'].append(k1)
#         tmp_df['sample_x2'].append(elem)
#   tmp_df = pd.DataFrame.from_dict(tmp_df)

#   # likely_value_x1 = [
#   #   elem
#   #   for elem in np.unique(tmp_df['value_x1'])
#   #   if scm_obj.noises_distributions['u1'].pdf(elem) > 0.05
#   # ]
#   # tmp_df = tmp_df[tmp_df['value_x1'].isin(likely_value_x1)] # filter to rows with highly likely value_x1

#   # ipsh()
#   # for elem in likely_value_x1:
#   #   samples_x2_for_likely_x1 = tmp_df[tmp_df['value_x1'] == elem]
#   #   print(
#   #     f'value_x1: {elem}\n',
#   #     samples_x2_for_likely_x1.groupby('recourse_type')['sample_x2'].mean(),
#   #     # samples_x2_for_likely_x1.groupby('recourse_type')['sample_x2'].std()
#   #   )

#   ax = sns.boxplot(x="value_x1", y="sample_x2", hue="recourse_type", data=tmp_df, palette="Set3", showmeans=True)
#   # TODO: average over high dens pdf, and show a separate plot/table for the average over things...
#   # ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
#   # pyplot.show()
#   pyplot.savefig(f'{experiment_folder_name}/_sanity_3.pdf')
