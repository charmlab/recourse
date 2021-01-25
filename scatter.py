import numpy as np
from matplotlib import pyplot as plt

def scatterFactual(args, objs, factual_instance, ax):
  assert len(objs.dataset_obj.getInputAttributeNames()) == 3
  ax.scatter(
    factual_instance['x1'],
    factual_instance['x2'],
    factual_instance['x3'],
    marker='P',
    color='black',
    s=70
  )


def scatterRecourse(args, objs, factual_instance, action_set, recourse_type, marker_type, legend_label, ax):

  assert len(objs.dataset_obj.getInputAttributeNames()) == 3

  if recourse_type in ACCEPTABLE_POINT_RECOURSE:
    # point recourse

    point = computeCounterfactualInstance(args, objs, factual_instance, action_set, recourse_type)
    color_string = 'green' if didFlip(args, objs, factual_instance, point) else 'red'
    ax.scatter(point['x1'], point['x2'], point['x3'], marker=marker_type, color=color_string, s=70, label=legend_label)

  elif recourse_type in ACCEPTABLE_DISTR_RECOURSE:
    # distr recourse

    samples_df = getRecourseDistributionSample(args, objs, factual_instance, action_set, recourse_type, args.num_display_samples)
    x1s = samples_df.iloc[:,0]
    x2s = samples_df.iloc[:,1]
    x3s = samples_df.iloc[:,2]
    color_strings = ['green' if didFlip(args, objs, factual_instance, sample.to_dict()) else 'red' for _, sample in samples_df.iterrows()]
    ax.scatter(x1s, x2s, x3s, marker=marker_type, color=color_strings, alpha=0.1, s=30, label=legend_label)

    # mean_distr_samples = {
    #   'x1': np.mean(samples_df['x1']),
    #   'x2': np.mean(samples_df['x2']),
    #   'x3': np.mean(samples_df['x3']),
    # }
    # color_string = 'green' if didFlip(args, objs, factual_instance, mean_distr_samples) else 'red'
    # ax.scatter(mean_distr_samples['x1'], mean_distr_samples['x2'], mean_distr_samples['x3'], marker=marker_type, color=color_string, alpha=0.5, s=70, label=legend_label)

  else:

    raise Exception(f'{recourse_type} not recognized.')


def scatterFit(args, objs, experiment_folder_name, experimental_setups, node, parents, total_df):

  fig, axes = plt.subplots(1, len(parents))
  if len(parents) == 1:
    axes = [axes]
  for idx, parent in enumerate(parents):
    for recourse_type in np.setdiff1d(np.unique(total_df['recourse_type']), 'true data'):
      tmp = total_df[total_df['recourse_type'] == recourse_type]
      marker = [elem[1] for elem in experimental_setups if elem[0] == recourse_type][0]
      axes[idx].scatter(tmp[parent], tmp[node], marker=marker, alpha=0.3, s=8, label=recourse_type)
      axes[idx].set_xlabel(parent, fontsize='xx-small')
      axes[idx].set_ylabel(node, fontsize='xx-small')
      axes[idx].set_title(f'{node} on {parent}', fontsize='xx-small')
      axes[idx].grid(True)
      axes[idx].legend(fontsize='xx-small')
      axes[idx].tick_params(axis='both', which='major', labelsize='xx-small')
      axes[idx].tick_params(axis='both', which='minor', labelsize='xx-small')
  plt.savefig(f'{experiment_folder_name}/_sanity_fit_{getConditionalString(node, parents)}.pdf')
  plt.close()


def scatterDataset(dataset_obj, ax):
  assert len(dataset_obj.getInputAttributeNames()) <= 3
  X_train, X_test, y_train, y_test = dataset_obj.getTrainTestSplit()
  X_train_numpy = X_train.to_numpy()
  X_test_numpy = X_test.to_numpy()
  y_train = y_train.to_numpy()
  y_test = y_test.to_numpy()
  number_of_samples_to_plot = min(200, X_train_numpy.shape[0], X_test_numpy.shape[0])
  for idx in range(number_of_samples_to_plot):
    color_train = 'black' if y_train[idx] == 1 else 'magenta'
    color_test = 'black' if y_test[idx] == 1 else 'magenta'
    if X_train.shape[1] == 2:
      ax.scatter(X_train_numpy[idx, 0], X_train_numpy[idx, 1], marker='s', color=color_train, alpha=0.2, s=10)
      ax.scatter(X_test_numpy[idx, 0], X_test_numpy[idx, 1], marker='o', color=color_test, alpha=0.2, s=15)
    elif X_train.shape[1] == 3:
      ax.scatter(X_train_numpy[idx, 0], X_train_numpy[idx, 1], X_train_numpy[idx, 2], marker='s', color=color_train, alpha=0.2, s=10)
      ax.scatter(X_test_numpy[idx, 0], X_test_numpy[idx, 1], X_test_numpy[idx, 2], marker='o', color=color_test, alpha=0.2, s=15)


def scatterDecisionBoundary(dataset_obj, classifier_obj, ax):

  if len(dataset_obj.getInputAttributeNames()) == 2:

    x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    X = np.linspace(ax.get_xlim()[0] - x_range / 10, ax.get_xlim()[1] + x_range / 10, 1000)
    Y = np.linspace(ax.get_ylim()[0] - y_range / 10, ax.get_ylim()[1] + y_range / 10, 1000)
    X, Y = np.meshgrid(X, Y)
    Xp = X.ravel()
    Yp = Y.ravel()

    # if normalized_fixed_model is False:
    #   labels = classifier_obj.predict(np.c_[Xp, Yp])
    # else:
    #   Xp = (Xp - dataset_obj.attributes_kurz['x0'].lower_bound) / \
    #        (dataset_obj.attributes_kurz['x0'].upper_bound - dataset_obj.attributes_kurz['x0'].lower_bound)
    #   Yp = (Yp - dataset_obj.attributes_kurz['x1'].lower_bound) / \
    #        (dataset_obj.attributes_kurz['x1'].upper_bound - dataset_obj.attributes_kurz['x1'].lower_bound)
    #   labels = classifier_obj.predict(np.c_[Xp, Yp])
    labels = classifier_obj.predict(np.c_[Xp, Yp])
    Z = labels.reshape(X.shape)

    cmap = plt.get_cmap('Paired')
    ax.contourf(X, Y, Z, cmap=cmap, alpha=0.5)

  elif len(dataset_obj.getInputAttributeNames()) == 3:

    fixed_model_w = classifier_obj.coef_
    fixed_model_b = classifier_obj.intercept_

    x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    X = np.linspace(ax.get_xlim()[0] - x_range / 10, ax.get_xlim()[1] + x_range / 10, 10)
    Y = np.linspace(ax.get_ylim()[0] - y_range / 10, ax.get_ylim()[1] + y_range / 10, 10)
    X, Y = np.meshgrid(X, Y)
    Z = - (fixed_model_w[0][0] * X + fixed_model_w[0][1] * Y + fixed_model_b) / fixed_model_w[0][2]

    surf = ax.plot_wireframe(X, Y, Z, alpha=0.3)


def visualizeDatasetAndFixedModel(dataset_obj, classifier_obj, experiment_folder_name):

  if not len(dataset_obj.getInputAttributeNames()) <= 3:
    print(f'[INFO] Cannot visualize dataset and model in {len(dataset_obj.getInputAttributeNames())} dimensions. Skipping.')
    return

  fig = plt.figure()
  if len(dataset_obj.getInputAttributeNames()) == 2:
    ax = plt.subplot()
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.grid()
  elif len(dataset_obj.getInputAttributeNames()) == 3:
    ax = plt.subplot(1, 1, 1, projection = '3d')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')
    ax.view_init(elev=10, azim=-20)

  scatterDataset(dataset_obj, ax)
  scatterDecisionBoundary(dataset_obj, classifier_obj, ax)

  ax.set_title(f'{dataset_obj.dataset_name}')
  ax.grid(True)

  # plt.show()
  plt.savefig(f'{experiment_folder_name}/_dataset_and_model.pdf')
  plt.close()


