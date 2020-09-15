import math
from os import path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from git import Repo
from scipy.spatial import distance
from sklearn.cluster import DBSCAN


def clone_repo(url, directory):
    if path.exists(directory):
        return Repo(directory)
    return Repo.clone_from(url, directory)


def create_contributions_matrix(limit):
    import time
    seconds = time.time()
    print("Cloning repo...")
    repo = clone_repo("https://github.com/facebook/react.git", "react")
    print("done in %d seconds" % (time.time() - seconds))

    files_index = {}
    contributions = {}

    def mark_commit_to_file(name, file_name, file_changes):
        if name not in contributions:
            contributions[name] = {}
        name_contributions = contributions[name]
        if file_name not in files_index:
            files_index[file_name] = len(files_index)
        file_index = files_index[file_name]
        if file_index not in name_contributions:
            name_contributions[file_index] = 0
        name_contributions[file_index] += file_changes

    print("Processing commits..")
    seconds = time.time()
    for commit in repo.iter_commits():
        for file, stats in commit.stats.files.items():
            mark_commit_to_file(commit.author.name, file, stats["lines"])
            if commit.author.name != commit.committer.name and commit.committer.name != "GitHub":
                mark_commit_to_file(commit.committer.name, file, stats["lines"])
    print("done in %d seconds" % (time.time() - seconds))

    popular_contributors = np.array(list(sorted([(name, sum(files.values())) for name, files in contributions.items()],
                                                key=lambda x: -x[1])[:limit]))[:, 0]

    matrix = np.zeros((len(popular_contributors), len(files_index)))
    for contributor_index in range(len(popular_contributors)):
        for file_index, changes in contributions[popular_contributors[contributor_index]].items():
            matrix[contributor_index][file_index] = changes
    return popular_contributors, matrix


def get_contribution_matrix():
    limit = 50
    file_name = "contributions.npy"
    if path.exists(file_name):
        with open(file_name, 'rb') as f:
            matrix = np.load(f)
            popular_contributors = np.load(f)
        print("Commits are recovered from cache")
    else:
        popular_contributors, matrix = create_contributions_matrix(limit)
        with open(file_name, 'wb') as f:
            np.save(f, matrix)
            np.save(f, popular_contributors)
    return popular_contributors, matrix


def get_contribution_sum(matrix):
    n = matrix.shape[0]
    contribution_sum = np.zeros(n)
    for i in range(n):
        contribution_sum[i] = sum(matrix[i])
        matrix[i] /= contribution_sum[i]
    return contribution_sum


def main():
    popular_contributors, matrix = get_contribution_matrix()

    contribution_sum = get_contribution_sum(matrix)
    edges_limit = 50
    edges = sorted([
        (popular_contributors[i], popular_contributors[j], distance.cosine(matrix[i], matrix[j]))
        for j in range(len(popular_contributors))
        for i in range(j + 1, len(popular_contributors))
    ], key=lambda x: x[2])
    g = nx.Graph()
    g.add_nodes_from(popular_contributors)
    g.add_weighted_edges_from(edges[:edges_limit])

    contribution_sum = contribution_sum / sum(contribution_sum) * 10000 + 100
    nodes = list(g.nodes)
    plt.figure(figsize=(20, 20))
    positions = nx.spring_layout(g, k=2 / math.sqrt(len(nodes)), iterations=50)
    nx.draw_networkx_nodes(g, positions, node_size=contribution_sum, alpha=0.3)
    nx.draw_networkx_edges(g, positions)
    for i in range(len(nodes)):
        label = nodes[i]
        pos = positions[label]
        plt.text(pos[0], pos[1], label, fontsize=max(10, contribution_sum[i] / 100))
    plt.show()


def do_cluster():
    popular_contributors, matrix = get_contribution_matrix()
    contribution_sum = get_contribution_sum(matrix)

    clustering = DBSCAN(eps=0.5, min_samples=2, metric='cosine').fit(matrix)
    g = nx.Graph()
    g.add_nodes_from(popular_contributors)
    clusters = {}
    for label, contributor in zip(clustering.labels_, popular_contributors):
        if label not in clusters:
            clusters[label] = set()
        if label != -1:
            for u in clusters[label]:
                g.add_edge(contributor, u)
        clusters[label].add(contributor)
    contribution_sum = contribution_sum / sum(contribution_sum) * 10000 + 100
    nodes = list(g.nodes)
    colors = ['#FFF700', '#E52B50', '#FFBF00', '#9966CC', '#7FFFD4', '#007FFF', '#7FFF00', '#DC143C']
    plt.figure(figsize=(10, 10))
    positions = nx.spring_layout(g, k=0.9)
    color = 0
    for label, cluster in clusters.items():
        if label == -1:
            c = "#000000"
            for v in cluster:
                positions[v] *= 2
        else:
            c = colors[color]
            color += 1
        nx.draw_networkx_nodes(g, positions, nodelist=list(cluster), alpha=0.7, node_color=c)
    nx.draw_networkx_edges(g, positions, alpha=0)
    for i in range(len(nodes)):
        label = nodes[i]
        pos = positions[label]
        plt.text(pos[0], pos[1], label, fontsize=max(10, contribution_sum[i] / 100))
    plt.show()


if __name__ == '__main__':
    main()
    do_cluster()
