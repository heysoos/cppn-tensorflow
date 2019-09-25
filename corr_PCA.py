import scipy.stats
from sklearn.decomposition import PCA
from calculate_all_correlations_parallel import theta_vec dist_vec

thetas = theta_vec((1023,1023))
distances = dist_vec((1023, 1023))

mask = distances < 511

for i in range(len(corrs_means)):
    corrs_means[i] = corrs_means[i][mask]

N = len(corrs_means)

ret = scipy.stats.binned_statistic_2d(
            distances[mask], thetas[mask], corrs_means, statistic='mean', bins=[512, 16])

c = ret.statistic
c = c.reshape(N * 32, 512)

pca = PCA(n_components=64)
pca.fit(c)

############## TSNE ##############

from sklearn import manifold

tsne = manifold.TSNE(n_components=2, init='pca',
                     random_state=0, perplexity=30, n_iter=5000)

cc = np.concatenate((corrs_means, aesthetic_corr, np.stack(nat_corr)))
c_tsne = tsne.fit_transform(cc)


cc_label = [
    np.stack([None for i in range(len(corrs_means))]),
    np.stack([None for i in range(len(aesthetic_corr))]),
    np.stack(img_folders)
]

cc_label = [item for sublist in cc_label for item in sublist]

cc_c = [
    np.stack(['k' for i in range(len(corrs_means))]),
    np.stack(['b' for i in range(len(aesthetic_corr))]),
    np.stack(['g' for i in range(len(aesthetic_corr))])
]

cc_c = [item for sublist in cc_c for item in sublist]


for i in range(len(cc)):
    plt.scatter(cc[i, 0], cc[i, 1], c=cc_c[i], s=1)


