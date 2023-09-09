from math import dist
import numpy as np
import pandas as pd
import pickle
import random
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

def get_pickle(path, name):
    with open(path + name, "rb") as handle:
        data = pickle.load(handle)
    return data



def CR_VSM(data, users, vectores, distance_type):
    dictionary_dataframe = []

    tqdm.pandas()

    print("Calculating centroid for the users ...")
    centroids, distances = centroid_users(data, vectores, distance_type)

    print("Getting negatives from different restaurants...")
    
    for _ in range(10):
        neg_samples = (
            data.groupby("id_user", group_keys=False)
            .progress_apply(
                lambda x: getSamplesDifferentRestaurantCentroid(
                    x, data, centroids, distances, vectores, distance_type
                )
            )
            .reset_index(drop=True)
        )
        dictionary_dataframe.extend(neg_samples.to_dict(orient="records"))
        dictionary_dataframe.extend(data.to_dict(orient="records"))

    print("Getting negatives from same restaurants...")

    # Filter out the restaurants that have images from only one user (note that using the 10-repeats
    # has no effect on the result of this operation)
    
    data = (
        data.groupby("id_restaurant")
        .filter(lambda x: x["id_user"].nunique() > 1)
        .reset_index(drop=True)
    )

    # Apply the algorithm to each restaurant
    for _ in range(10):
        same_res_bpr_samples = (
            data.groupby("id_restaurant", group_keys=False)
            .progress_apply(
                lambda x: getSamplesSameRestaurantCentroid(
                    x, centroids, distances, vectores, distance_type
                )
            )
            .reset_index(drop=True)
        )
        dictionary_dataframe.extend(same_res_bpr_samples.to_dict(orient="records"))
        dictionary_dataframe.extend(data.to_dict(orient="records"))


    print("Creating dataframe...")
    dataframe = pd.DataFrame.from_dict(dictionary_dataframe)

   
    print("Positive samples: ", dataframe[dataframe["take"] == 1].shape[0])
    print("Negative samples: ", dataframe[dataframe["take"] == 0].shape[0])

    # assert that all users have the same number of take=0 and take=1 samples
    positive_counts_per_user = (
        dataframe[dataframe["take"] == 1].groupby("id_user").size()
    )
    negative_counts_per_user = (
        dataframe[dataframe["take"] == 0].groupby("id_user").size()
    )
    assert np.all(positive_counts_per_user == negative_counts_per_user)

    # assert all users have at least 10 positive samples
    assert np.all(positive_counts_per_user >= 10)

    return dataframe


def centroid_users(data, vectores, distance_type):
    vectores = np.array(
        get_pickle(
            "../data/"
            + city
            + "/tripadimgrest_elvis_"
            + city
            + "/IMGMODEL/data_10+10/",
            "IMG_VEC",
        )
    )

    user_ids = data["id_user"]
    user_centroids = []
    user_distances = []
    img_ids = data["id_img"].to_numpy()[:, None]

    histogramas = [] #HISTOGRAMA


    for user_id in user_ids.unique():

        user_indices = np.where(data["id_user"] == user_id)[0]
        vect = vectores[img_ids[user_indices]]

        user_centroid = np.mean(vect, axis=0)
        user_centroids.append(user_centroid)
        if distance_type == 1:
            distances = np.linalg.norm(vect - user_centroid, axis=1)
        elif distance_type == 2:
            vect = np.squeeze(vect, axis=1)

            # Vect is a matrix of shape (n_samples, n_features)
            # User_centroid is a vector of shape (1, n_features)

            # The norm is a vector of shape (n_samples, 1)
            norm = np.linalg.norm(vect, axis=1) * np.linalg.norm(user_centroid)

            # Distances is a vector of shape (n_samples, 1)
            distances = np.dot(vect, np.squeeze(user_centroid)) / norm

        else:
            raise ValueError("Tipo de distancia no existente")

        # If we do euclidean distance we choose percentile 90, if we do cosine similarity we choose percentile 10
        if distance_type == 1:
            p90 = np.percentile(distances, 90)
        elif distance_type == 2:
            p90 = np.percentile(distances, 10)
        else:
            raise ValueError("Tipo de distancia no existente")

        user_distances.append(p90)
        
    return np.array(user_centroids), np.array(user_distances)


def getSamplesDifferentRestaurantCentroid(
    data_user, data, centroid_ids, p90, vectores, distance_type
):

	
    # Keep at each sample repeated only 10 times
   
    user_ids = data["id_user"].to_numpy()[:, None]
    img_ids = data["id_img"].to_numpy()[:, None]
    rest_ids = data["id_restaurant"].to_numpy()[:, None]

    id_user = data_user["id_user"].values[0]
    rest_user = data_user["id_restaurant"].to_numpy()[:, None]

    # We select random "sample ids" (indexes on the data, not img ids)
    new_negatives = np.random.randint(data.shape[0], size=data_user.shape[0])

    # We count how many of those random sample ids are from the same user or restaurant as the original samples
    # in the same position as the original samples
    

    
    if distance_type == 1:
        num_invalid_samples = np.sum(
            (
                (user_ids[new_negatives] == id_user)
                | (rest_ids[new_negatives] == rest_user)
                | (
                    np.linalg.norm(
                        vectores[img_ids[new_negatives]] - centroid_ids[id_user]
                    )
                    < p90[id_user]
                )
            )
        )

        # Resample again the sample ids ONLY for those samples, iteratively until all are valid
        while num_invalid_samples > 0:
            new_negatives[
                np.where(
                    (
                        (user_ids[new_negatives] == id_user)
                        | (rest_ids[new_negatives] == rest_user)
                        | (
                            np.linalg.norm(
                                vectores[img_ids[new_negatives]] - centroid_ids[id_user]
                            )
                            < p90[id_user]
                        )
                    )
                )[0]
            ] = np.random.randint(data.shape[0], size=num_invalid_samples)

            num_invalid_samples = np.sum(
                (
                    (user_ids[new_negatives] == id_user)
                    | (rest_ids[new_negatives] == rest_user)
                    | (
                        np.linalg.norm(
                            vectores[img_ids[new_negatives]] - centroid_ids[id_user]
                        )
                        < p90[id_user]
                    )
                )
            )
    elif distance_type == 2:
   
        centr = centroid_ids[id_user]

        num_invalid_samples = np.sum(
            (
                (user_ids[new_negatives] == id_user)
                | (rest_ids[new_negatives] == rest_user)
                | (
                    np.sum(np.squeeze(vectores[img_ids[new_negatives]], axis=1) * centr, axis=1)
                    / (
                        np.linalg.norm(np.squeeze(vectores[img_ids[new_negatives]], axis=1))
                        * np.linalg.norm(centr)
                    )
                    > p90[id_user]
                )
            )
        )

        # Resample again the sample ids ONLY for those samples, iteratively until all are valid
        while num_invalid_samples > 0:
   
            centr = centroid_ids[id_user]

            new_negatives[
                np.where(
                    (
                        (user_ids[new_negatives] == id_user)
                        | (rest_ids[new_negatives] == rest_user)
                        | (
                            np.sum(np.squeeze(vectores[img_ids[new_negatives]], axis=1) * centr, axis=1)
                            / (
                                np.linalg.norm(np.squeeze(vectores[img_ids[new_negatives]], axis=1))
                                * np.linalg.norm(centr)
                            )
                            > p90[id_user]
                        )
                    )
                )[0]
            ] = np.random.randint(data.shape[0], size=num_invalid_samples)

            num_invalid_samples = np.sum(
                (
                    (user_ids[new_negatives] == id_user)
                    | (rest_ids[new_negatives] == rest_user)
                    | (
                        np.sum(np.squeeze(vectores[img_ids[new_negatives]], axis=1) * centr, axis=1)
                        / (
                            np.linalg.norm(np.squeeze(vectores[img_ids[new_negatives]], axis=1))
                            * np.linalg.norm(centr)
                        )
                        > p90[id_user]
                    )
                )
            )
    else:
        raise ValueError("Tipo de distancia no existente")

    data_user["id_img"] = img_ids[new_negatives]
    data_user["take"] = 0
    data_user["id_restaurant"] = rest_ids[new_negatives]

    return data_user

   
def getSamplesSameRestaurantCentroid(rest, centroid_ids, p90, vectores, distance_type):
    # Works the same way as getSamplesDifferentRestaurant, but only restaurant-wise
    user_ids = rest["id_user"].to_numpy()[:, None]
    img_ids = rest["id_img"].to_numpy()[:, None]

   
    rest_chunks = np.array_split(rest, len(rest) // 500 + 1)
    all_negatives = []

    for chunk in rest_chunks:
        user_ids_chunks = chunk["id_user"].to_numpy()[:, None]

        new_negatives = np.random.randint(len(rest), size=len(chunk))

        if distance_type == 1:
            num_invalid_samples = np.sum(
                (
                    (user_ids[new_negatives] == user_ids_chunks)
                    | (
                        np.linalg.norm(
                            vectores[img_ids[new_negatives]]
                            - centroid_ids[user_ids_chunks]
                        )
                        < p90[user_ids_chunks]
                    )
                )
            )
   
            while num_invalid_samples > 0:
                new_negatives[
                    np.where(user_ids[new_negatives] == user_ids_chunks)[0]
                ] = np.random.randint(len(rest), size=num_invalid_samples)

                num_invalid_samples = np.sum(
                    (
                        (user_ids[new_negatives] == user_ids_chunks)
                        | (
                            np.linalg.norm(
                                vectores[img_ids[new_negatives]]
                                - centroid_ids[user_ids_chunks]
                            )
                            < p90[user_ids_chunks]
                        )
                    )
                )

        elif distance_type == 2:
   
            centr = np.squeeze(centroid_ids[user_ids_chunks], axis=(1,2))
            

            num_invalid_samples = np.sum(
                (
                    # Invalid if same user
                    (user_ids[new_negatives] == user_ids_chunks)
                    # Invalid if the cosine similarity is higher than the p90 (p10 in this case)
                    | (
                        np.sum(np.squeeze(vectores[img_ids[new_negatives]], axis=1) * centr, axis=1)
                        / (
                            np.linalg.norm(np.squeeze(vectores[img_ids[new_negatives]], axis=1))
                            * np.linalg.norm(centr)
                        )
                        > p90[user_ids_chunks]
                    )
                    
                )
            )

            while num_invalid_samples > 0:
                new_negatives[
                    np.where(user_ids[new_negatives] == user_ids_chunks
                    | (
                        np.sum(np.squeeze(vectores[img_ids[new_negatives]], axis=1) * centr, axis=1)
                        / (
                            np.linalg.norm(np.squeeze(vectores[img_ids[new_negatives]], axis=1))
                            * np.linalg.norm(centr)
                        )
                        > p90[user_ids_chunks]
                    ))[0]
                ] = np.random.randint(len(rest), size=num_invalid_samples)

                centr = np.squeeze(centroid_ids[user_ids_chunks], axis=1)

                num_invalid_samples = np.sum(
                    (
                        # Invalid if same user
                        user_ids[new_negatives] == user_ids_chunks
                        # Invalid if the cosine similarity is higher than the p90 (p10 in this case)
                        | (
                        np.sum(np.squeeze(vectores[img_ids[new_negatives]], axis=1) * centr, axis=1)
                        / (
                            np.linalg.norm(np.squeeze(vectores[img_ids[new_negatives]], axis=1))
                            * np.linalg.norm(centr)
                        )
                        > p90[user_ids_chunks]
                        )
                    )
                )
        else:
            raise ValueError("Tipo de distancia no existente")

        all_negatives.append(new_negatives)

    all_negatives = np.concatenate(all_negatives)

    rest["id_img"] = img_ids[all_negatives]
    rest["take"] = 0

    return rest




# Main

# 1º Ciudad con la que queremos trabajar
cities = np.array(['barcelona', 'madrid', 'gijon', 'london', 'newyork', 'paris'])

# 1º Ciudad con la que queremos trabajar
for city in cities:
	
	print("Ciudad: " + city)
	# 2º Obtenemos el dataframe
	datos = get_pickle(
	    "../data/"
	    + city
	    + "/tripadimgrest_virgindata_"
	    + city
	    + "/",
	    city + ".pkl",
	)
	datos_DEV = get_pickle("../data/" + city + "/tripadimgrest_virgindata_" + city + "/",
	                        city + '_DEV.pkl')

	vectores = np.array(
	    get_pickle(
		"../data/" + city + "/tripadimgrest_elvis_" + city + "/IMGMODEL/data_10+10/",
		"IMG_VEC",
	    )
	)


	df = pd.DataFrame(datos)
	df_DEV = pd.DataFrame(datos_DEV)
	
	# 3º Obtenemos los usuarios únicos
	users = df["id_user"].unique()
	users_DEV = df_DEV["id_user"].unique()

	# 4º Ejecutamos la función
	
	"""CENTROIDES (distancia euclídea)"""
	df = df.drop_duplicates(subset=['id_img'], keep= 'first')
	new_dataframe = CR_VSM(df, users, vectores, 1)

	df_DEV = df_DEV.drop_duplicates(subset=['id_img'], keep='first')
	new_dataframe_DEV = CR_VSM(df_DEV, users_DEV, vectores, 1)

	
	# 5º Reseteamos los índices de la serie
	new_dataframe = new_dataframe.reset_index(drop=True)
	new_dataframe_DEV = new_dataframe_DEV.reset_index(drop=True)

	# 5º Lo guardamos en formato .pkl
	new_dataframe.to_pickle(
	    "../data/"
	    + city
	    + "/tripadimgrest_virgindata_"
	    + city
	    + "/TRAIN_IMG"
	)

	new_dataframe_DEV.to_pickle(
	      "../data/" + city + "/tripadimgrest_virgindata_" + city + "/TRAIN_DEV_IMG")
