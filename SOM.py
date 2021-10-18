
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import tensorflow as tf
class SOM:
    def __init__(this, width, height, input_dimension):
        this.width = width
        this.height = height
        this.input_dimension = input_dimension
        this.weight = tf.Variable(tf.random.normal([ this.width * this.height, this.input_dimension ]))
        this.input = tf.placeholder(tf.float32, [this.input_dimension])
        this.location = this._generate_location(this.width, this.height) #tf.to_float[[y,x] for y in range(this.height) for x in range(this.width)]

        this.bmu = this._get_bmu()
        this.update_weight = this._update_neighbor()

    def _generate_location(this, width, height):
        loc = []
        for y in range(height):
            for x in range(width):
                loc.append([y, x])
        return tf.to_float(loc)

    def _get_bmu(this):
        square_diff = tf.square(this.input - this.weight)
        distance = tf.sqrt(tf.reduce_mean(square_diff, axis = 1))
        bmu_index = tf.argmin(distance)
        bmu_location = tf.to_float([tf.div(bmu_index, this.width), tf.mod(bmu_index, this.height)])
        return bmu_location

    def _update_neighbor(this):
        lr = .1
        sigma = tf.to_float(tf.maximum(this.width, this.height) / 2)
        square_diff = tf.square(this.bmu - this.location)
        distance = tf.sqrt(tf.reduce_mean(square_diff, axis = 1))
        neighbor_strength = tf.exp(tf.div(tf.negative(tf.square(distance)), 2 * tf.square(sigma)))
        rate = neighbor_strength * lr
        rate_stack = tf.stack([tf.tile(tf.slice(rate,[i], [1]), [this.input_dimension]) for i in range(this.width * this.height)])
        input_weight_difference = this.input - this.weight
        weight_diff = input_weight_difference * rate_stack
        new_weight = this.weight + weight_diff
        return tf.assign(this.weight, new_weight)

    def _train(this, dataset, epoch):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(epoch):
                for data in dataset:
                    sess.run(this.update_weight, feed_dict = {this.input: data})
            location = sess.run(this.location)
            if epoch % 100 == 0:
                print("epoch ke",epoch," dengan location=",location)
            weight = sess.run(this.weight)
            
            clusters = [[] for i in range(this.height)]
            for i, loc in enumerate(location):
                clusters[int(loc[0])].append(weight[i])
            this.clusters = clusters

def main():

    def load_data():
        dataset = pd.read_csv("credit_card_general_clustering.csv")
        feature = dataset[
                ['BALANCE','BALANCE_FREQUENCY','PURCHASES','ONEOFF_PURCHASES',
                'PURCHASES_FREQUENCY','ONEOFF_PURCHASES_FREQUENCY','PURCHASES_TRX'
                ,'PAYMENTS','MINIMUM_PAYMENTS','PRC_FULL_PAYMENT','TENURE']]
        return feature

    dataset = load_data()
    #Cleaning data
    print(dataset.isnull().sum().sort_values(ascending=False).head())
    #Input missing value with mean
    mean = dataset['MINIMUM_PAYMENTS'].mean()
    dataset['MINIMUM_PAYMENTS'].fillna(mean,inplace=True)
    print(dataset.isnull().sum().sort_values(ascending=False).head())

    #Normalize data

    def normalize(dataset):
        normalized_dataset = scaler.transform(dataset)
        return normalized_dataset

    scaler = MinMaxScaler().fit(dataset)
    normalized_dataset = normalize(dataset)
    print(normalized_dataset)
    #PCA 
    # pca= PCA()
    # pca.fit(normalized_dataset)
    # import matplotlib.pyplot as plt
    # print(pca.explained_variance_ratio_)
    # # plt.plot(pca.explained_variance_ratio_.cumsum())
    # # plt.show()

    pca= PCA(n_components=3)
    component = pca.fit_transform(normalized_dataset)
    print('\n\n')
    print(component)
    width =3
    height = 3
    input_dimension = 3
    epoch = 1
    
    som = SOM(width, height, input_dimension)
    som._train(component, epoch)
    plt.imshow(som.clusters)
    plt.show()
main()