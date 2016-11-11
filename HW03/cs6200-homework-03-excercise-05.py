import numpy as np
from scipy.io import arff
import pprint


def z_normalize(features):
    return (features - np.mean(features, axis=0)) / np.std(features, axis=0)


class Kmeans:
    def __init__(self, k, random_centroids, trail, data, max_iterations):
        self.k = k
        self.X = data
        self.max_iterations = max_iterations
        self.trail = trail
        self.random_c = np.array(random_centroids)
        self.centroids = self.init_centroids()
        self.clusters = None
        self.converge()

    def init_centroids(self):
        start = (self.trail - 1) * self.k
        end = self.trail * self.k
        centroid_indices = self.random_c[start:end]
        centroids = []
        for ci in centroid_indices:
            centroids.append(self.X[ci].flatten().tolist())
        return centroids

    def generate_clusters(self):
        clusters = [[] for c in self.centroids]
        for data_point in self.X:
            distance = float("inf")
            closest_centriod_index = None
            for cen_index, centriod in enumerate(self.centroids):
                temp_distance = np.linalg.norm(data_point - centriod)
                if temp_distance < distance:
                    distance = temp_distance
                    closest_centriod_index = cen_index
            clusters[closest_centriod_index].append(data_point)
        self.clusters = clusters

    def sum_cluster(self, cluster):
        sum = cluster[0]
        for data_point in cluster[1:]:
            sum += data_point
        return sum

    def meam_cluster(self, cluster):
        sum = self.sum_cluster(cluster)
        return sum / len(cluster)

    def move_centriods(self):
        new_centriods = []
        for cluster in self.clusters:
            new_centriods.append(np.mean(cluster, axis=0).tolist())
        return new_centriods

    def converge(self):
        # pprint.pprint(self.centroids)
        iteration = 0
        while iteration <= self.max_iterations:
            iteration += 1
            # print "iteration {}".format(iteration)
            self.generate_clusters()
            old_centroids = self.centroids
            self.centroids = self.move_centriods()
            # print "sse {}".format(self.generate_sse())
            if self.centroids == old_centroids:
                print "{}-means converged after {} iterations".format(self.k, iteration)
                break

    def generate_sse(self):
        distance = [[] for c in self.clusters]
        for index, cluster in enumerate(self.clusters):
            for data_point in cluster:
                distance[index].append(np.linalg.norm(data_point - self.centroids[index]))
        return np.sum(np.sum(distance))


def load_data():
    data, meta = arff.loadarff('data/segment.arff')
    X = data[meta.names()[:-1]]  # everything but the last column
    y = data[meta.names()[-1]]
    X = X.view(np.float).reshape(
        data.shape + (-1,))  # converts the record array to a normal numpy array
    return X, y


def get_random_centriod():
    return [773, 1010, 240, 126, 319, 1666, 1215, 551, 668, 528, 1060, 168, 402, 80, 115, 221,
            242, 1951, 1725, 754, 1469, 135, 877, 1287, 645, 272, 1203, 1258, 1716, 1158, 586,
            1112, 1214, 153, 23, 510, 05, 1254, 156, 936, 1184, 1656, 244, 811, 1937, 1318, 27,
            185, 1424, 190, 663, 1208, 170, 1507, 1912, 1176, 1616, 109, 274, 1, 1371, 258, 1332,
            541, 662, 1483, 66, 12, 410, 1179, 1281, 145, 1410, 664, 155, 166, 1900, 1134, 1462,
            954, 1818, 1679, 832, 1627, 1760, 1330, 913, 234, 1635, 1078, 640, 833, 392, 1425,
            610, 1353, 1772, 908, 1964, 1260, 784, 520, 1363, 544, 426, 1146, 987, 612, 1685, 1121,
            1740, 287, 1383, 1923, 1665, 19, 1239, 251, 309, 245, 384, 1306, 786, 1814, 7, 1203,
            1068, 1493, 859, 233, 1846, 1119, 469, 1869, 609, 385, 1182, 1949, 1622, 719, 643,
            1692, 1389, 120, 1034, 805, 266, 339, 826, 530, 1173, 802, 1495, 504, 1241, 427, 1555,
            1597, 692, 178, 774, 1623, 1641, 661, 1242, 1757, 553, 1377, 1419, 306, 1838, 211, 356,
            541, 1455, 741, 583, 1464, 209, 1615, 475, 1903, 555, 1046, 379, 1938, 417, 1747, 342,
            1148, 1697, 1785, 298, 185, 1145, 197, 1207, 1857, 158, 130, 1721, 1587, 1455, 190,
            177, 1345, 166, 1377, 1958, 1727, 1134, 1953, 1602, 114, 37, 164, 1548, 199, 1112, 128,
            167, 102, 87, 25, 249, 1240, 1524, 198, 111, 1337, 1220, 1513, 1727, 159, 121, 1130,
            1954, 1561, 1260, 150, 1613, 1152, 140, 1473, 1734, 137, 1156, 108, 110, 1829, 1491,
            1799, 174, 847, 177, 1468, 97, 1611, 1706, 1123, 79, 171, 130, 100, 143, 1641, 181,
            135, 1280, 1442, 1188, 133, 99, 186, 1854, 27, 160, 130, 1495, 101, 1411, 814, 109, 95,
            111, 1582, 1816, 170, 1663, 1737, 1710, 543, 1143, 1844, 159, 48, 375, 1315, 1311, 1422]


def part_a():
    X, y = load_data()
    # X= z_normalize(X)
    sse_matrix = []
    for k in range(10, 12):
        trail_sse = []
        for i in range(1, 4):
            kmeans = Kmeans(data=X, k=k, random_centroids=get_random_centriod(), trail=i, max_iterations=50)
            trail_sse.append(kmeans.generate_sse())
        sse_matrix.append(trail_sse)
    pprint.pprint(sse_matrix, width=25)
    means = map(lambda x: np.mean(x), sse_matrix)
    stds = map(lambda x: np.std(x), sse_matrix)
    print means
    print stds


if __name__ == "__main__":
    part_a()
    # print kmeans.centroids
    # print len(kmeans.clusters)
