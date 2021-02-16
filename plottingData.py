import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set()


def show_histograms(Y, X, title=''):
    ax = sns.distplot(X[Y==-1], hist=False, kde_kws = {'shade': True, 'linewidth': 2}, label='$Y=-1$')
    sns.distplot(X[Y==1], hist=False, kde_kws = {'shade': True, 'linewidth': 2}, label='$Y=+1$')
    ax.set_xlabel("$X_2$")
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend()
    plt.show()
    return ax


n_samples = 10000
scm_class = 'fair-IMF-LIN'
# scm_class = 'fair-CAU-LIN'
# scm_class = 'fair-CAU-ANM'

# classifier_type = 'linear'
classifier_type = 'radial'

U_A = np.random.uniform(0, 1, n_samples)
A = 2 * (U_A > 0.5) - 1  # needs to be +1 or -1

U_X_1 = np.random.normal(0, 1, n_samples)
X_1 = 0.5 * A + U_X_1

U_X_2 = np.random.normal(0, 1, n_samples)
X_2 = 0 + U_X_2

U_X_3 = np.random.normal(0, 1, n_samples)
if scm_class == 'fair-IMF-LIN':
    X_3 = 0.5 * A + U_X_3
elif scm_class == 'fair-CAU-LIN':
    X_3 = 0.5 * (A + X_1 - X_2) + U_X_3
elif scm_class == 'fair-CAU-ANM':
    X_3 = 0.5 * A + 0.1 * (X_1**3 - X_2**3) + U_X_3
else:
    print('SCM type not recognised')

if classifier_type == 'linear':
    h = (1 + np.exp(-2*(X_1 + X_2 - X_3))) ** (-1)
elif classifier_type == 'radial':
    h = (1 + np.exp(4-(X_1 + 2 * X_2 + X_3)**2)) ** (-1)
else:
    print('Classifier type not recognised')

noise = np.random.uniform(0, 1, n_samples)
Y = 2 * (noise < h) - 1  # needs to be +1 or -1

print('SCM:', scm_class)
print('Classifier:', classifier_type)

print("The following 4 numbers should be roughly equal for a balanced dataset")
print("Class +1:", sum((Y == 1)))
print("Class -1:", sum((Y == -1)))
print("Attribute +1:", sum((A == 1)))
print("Attribute -1:", sum((A == -1)))

title_string = scm_class+' with '+classifier_type+' classifier '
label_dist_x2 = show_histograms(Y, X_2, title_string)
fig = label_dist_x2.get_figure()
fig.savefig(title_string+'.pdf')
# synth_data = np.stack([Y, A, X_1, X_2, X_3], axis=1)
# synth_data_frame = pd.DataFrame(synth_data)
# sns.pairplot(synth_data_frame)
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.set_xlabel("X_1")
# ax.set_ylabel("X_2")
# ax.set_zlabel("X_3")
# ax.scatter(X_1[(Y == 1) & (A == 1)], X_2[(Y == 1) & (A == 1)], X_3[(Y == 1) & (A == 1)], color='green', marker='o')
# ax.scatter(X_1[(Y == -1) & (A == 1)], X_2[(Y == -1) & (A == 1)], X_3[(Y == -1) & (A == 1)], color='red', marker='o')
# ax.scatter(X_1[(Y == 1) & (A == -1)], X_2[(Y == 1) & (A == -1)], X_3[(Y == 1) & (A == -1)], color='blue', marker='x')
# ax.scatter(X_1[(Y == -1) & (A == -1)], X_2[(Y == -1) & (A == -1)], X_3[(Y == -1) & (A == -1)], color='orange', marker='x')
# plt.show()
