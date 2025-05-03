import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt

from cv2 import cvtColor,resize
from PIL import Image, ImageDraw, ImageFont

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def reshape_image(input_img):

    image_path = input_img
    image = cv.imread(image_path)

    # Convert from BGR to RGB
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image_resized=cv.resize(image_rgb,(200,200  )) #resize image for faster processing
    #we reshape xy pixels of the image into 1D vector of length nx * ny for point classification
    reshaped_image=image_resized.reshape(-1,3)
    #initialise color lists

    return reshaped_image

def reshaped_image_3d_plot(input_img):
    reshaped_image= reshape_image(input_img)
    red=[]
    green=[]
    blue=[]

    for index,element in enumerate(reshaped_image) :
        red.append(reshaped_image[index][0])
        green.append(reshaped_image[index][1])
        blue.append(reshaped_image[index][2])

    #convert lists into array for subsequent RGB color plotting

    red=np.array(red)
    green=np.array(green)
    blue=np.array(blue)

    #3d plot



    fig = plt.figure(figsize=(13,13))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('R',color="r", fontsize=12)
    ax.set_xlim([0,255])
    ax.set_ylabel('G',color="g", fontsize=12)
    ax.set_ylim([0,255])
    ax.set_zlabel('B',color="b", fontsize=12)
    ax.set_zlim([0,255])
    ax.scatter(red, green, blue,c=np.stack((red/255, green/255, blue/255), axis=-1), marker='o')

    plt.title("RGB Decomposition of initial image",fontsize=20)

    return fig

def find_RAL(centroid_colors):
    RAL_df = pd.read_csv("ral_classic.csv") #create dataframe from RAL table
    # Apply the conversion to the RGB column
    RAL_df['RGB_converted'] = RAL_df['RGB'].apply(convert_to_list).copy()


    """KMeans model to find closest (n=1) label"""
    X = np.array(RAL_df['RGB_converted'].tolist())  # Convert the lists to a 2D array (n_samples x 3 features)
    Y = RAL_df['RAL'].copy()  # Target variable (1D)


    classifier = KNeighborsClassifier(n_neighbors=1)
    classifier.fit(X=X,y=Y)

    predicted_RAL_code=classifier.predict(np.array([centroid_colors])) #this is the chosen ral
    predicted_RAL_text=RAL_df.loc[RAL_df['RAL'] == predicted_RAL_code[0], 'English'].iloc[0]

    return predicted_RAL_code,predicted_RAL_text

def KMeans_models (reshaped_image):
    scaler=MinMaxScaler()
    X=scaler.fit_transform(reshaped_image)
    k_range = range(2,4) # range of K means that can be tuned

    k_index=[]
    k_inertia=[]
    silhouette_k=[]
    centroids_k=[]
    labels_k=[]

    for k in k_range:
        print (f"Processing cycle with {k} centroids, out of a max of {max(k_range)}")
        k_index.append(k)
        model = KMeans (n_clusters=k,random_state=42,init="random",max_iter=500, n_init=10)
        model.fit(X)
        centroids_k.append(model.cluster_centers_)  # Get cluster centers
        labels_k.append(model.labels_)  # Get labels for each point
        k_inertia.append(model.inertia_)
        silhouette_k.append(silhouette_score(X, model.labels_, metric='euclidean'))
        # Apply the elbow method to find the optimal number of clusters.
        #lookup https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html#sklearn.metrics.silhouette_score
        #for automatic inflection point evaluation in elbow method

    return k_index,centroids_k,labels_k,k_inertia,silhouette_k

def colors_k_centroids_all (centroids_k):
    """COLOR DISTRUBUTION OF CENTROIDS"""
    #centroids_k = KMeans_models (reshaped_image)[1]

    max_centroids = centroids_k[-1]
    red_centroid=[]
    green_centroid=[]
    blue_centroid=[]

    for k in range(len(max_centroids)):
        red_centroid.append(max_centroids[k][0])
        green_centroid.append(max_centroids[k][1])
        blue_centroid.append(max_centroids[k][2])

    red_array=np.array(red_centroid)
    green_array=np.array(green_centroid)
    blue_array=np.array(blue_centroid)

    return red_array,green_array,blue_array

def three_main_centroids_colors(labels_k,red_array,green_array,blue_array):
    # labels_k = KMeans_models (reshaped_image)[2]
    # red_array,green_array,blue_array = colors_k_centroids_all (centroids_k)[:2]


    unique, counts = np.unique(labels_k, return_counts=True) #find centroid label and corresponding count
    #find 3 most populated centroids counts /indexes
    counting=sorted(zip(unique, counts), reverse=False)[:3]
    total=counts.sum()

    counting_unzip=list(zip(*counting)) #create one list with indexes and the other with counts
    centroid_k_main_index=list(counting_unzip[0])

    red_array_k_main=[]
    green_array_k_main=[]
    blue_array_k_main=[]

    for k in centroid_k_main_index:
        red_array_k_main.append(red_array[k])
        green_array_k_main.append(green_array[k])
        blue_array_k_main.append(blue_array[k])

    percentage_counts_k= np.round(counting_unzip[1]/total*100, 1)


    k_main_color_list=[[(int(np.round(red_array_k_main[i]*255,0)), int(np.round(green_array_k_main[i]*255,0)), int(np.round(blue_array_k_main[i]*255,0)))] for i in centroid_k_main_index]

    return centroid_k_main_index,percentage_counts_k,k_main_color_list

def create_color_rectangle_centroid(k_main_color_list,percentage_counts_k):

    color_square_list=k_main_color_list

    # Create a new RGB image (width x height)
    width, height = 200, 100
    for i in range(len(color_square_list)):

        image = Image.new("RGB", (width, height), color=(color_square_list[i][0]))  # Dodger blue

        # Create a drawing context
        draw = ImageDraw.Draw(image)

        # Define your text
        text = f"{percentage_counts_k[i]} %"

        # Optional: Load a font (or use default)
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 45)
        except IOError:
            print("⚠️ Falling back to default font!")
            font = ImageFont.load_default()

        bbox = draw.textbbox((0, 0), text, font=font, stroke_width=1)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Calculate centered position
        x = (width - text_width) / 2
        y = (height - text_height) / 2

        # Draw text
        draw.text((x, y), text, fill=(0, 0, 0), font=font, stroke_width=1, stroke_fill=(255, 255, 255))

        # Save image
        image.save(f"./images/rectangle_image{i}.png")
    return "images created"

def convert_to_list(color_string):
    return np.array(list(map(int, color_string.split('-'))))

def classify_RAL (rgb_color):

    color_df = pd.read_csv("ral_classic.csv")
    # Apply the conversion to the column
    color_df['RGB_converted'] = color_df['RGB'].apply(convert_to_list).copy()
    X = np.array(color_df['RGB_converted'].tolist())  # Convert the lists to a 2D array (n_samples x 3 features)
    Y = color_df['RAL'].copy()  # Target variable (1D)

    classifier = KNeighborsClassifier(n_neighbors=1)
    classifier.fit(X=X,y=Y)
    predicted_RAL_code=classifier.predict(np.array([rgb_color])) #this is the chosen ral
    predicted_RAL_text=color_df.loc[color_df['RAL'] == predicted_RAL_code[0], 'English'].iloc[0]

    return(predicted_RAL_code,predicted_RAL_text)


def return_product_URL (predicted_RAL_code, predicted_RAL_text):
    # Set the preset search query
    search_query = str(predicted_RAL_text) + str(predicted_RAL_code) + " table spray"

    # Construct the search URL for Google
    url = f"https://www.google.com/search?q={search_query}"

    # Open the URL in the default web browser
    return url
