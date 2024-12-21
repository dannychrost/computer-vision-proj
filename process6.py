# 50.59% accuracy
import pandas as pd
import os
import numpy as np
import cv2
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import dlib
import joblib
from tqdm import tqdm

# Config
n_actors = 10
images_per_person = 10
x_y = 100
seed = 112524
n_components_optimal = 80

# Loading dlib models
predictor_path = "shape_predictor_5_face_landmarks.dat"
cnn_detector_path = "mmod_human_face_detector.dat"
shape_predictor = dlib.shape_predictor(predictor_path)
face_detector = dlib.cnn_face_detection_model_v1(cnn_detector_path)


# Data augmentation function
def augment_images(images, labels):
    augmented_images = []
    augmented_labels = []
    for image, label in zip(images, labels):
        augmented_images.append(image)
        augmented_labels.append(label)
        flipped = cv2.flip(image, 1)
        augmented_images.append(flipped)
        augmented_labels.append(label)
        for angle in [-10, 10]:
            M = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), angle, 1)
            rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
            augmented_images.append(rotated)
            augmented_labels.append(label)
    return augmented_images, augmented_labels

# 1. Load and preprocess data
csv_file = 'IMDb-Face.csv'
df = pd.read_csv(csv_file)
df = df.head(4000)
df.drop(columns=['url'], inplace=True) 
df['location'] = df.apply(lambda row: os.path.join('downloads2', str(row['index']), row['image']), axis=1)
df = df[df['location'].apply(os.path.exists)]
df = df.reset_index(drop=True)
actor_counts = df['name'].value_counts()
actors_to_keep = actor_counts[actor_counts >= images_per_person].index
df = df[df['name'].isin(actors_to_keep)]

# 2. Align faces through dlib
processed_images = []
labels = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Aligning Faces"):
    image_path = row['location']
    if not os.path.exists(image_path):
        print(f"File does not exist: {image_path}")
        continue
    
    bgr_image = cv2.imread(image_path)
    if bgr_image is None:
        continue
    
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    
    faces = face_detector(rgb_image, 0)
    if len(faces) == 0:
        continue

    face_rect = faces[0].rect
    shape = shape_predictor(rgb_image, face_rect)
    aligned_face = dlib.get_face_chip(rgb_image, shape, size=x_y)

    gray_image = cv2.cvtColor(aligned_face, cv2.COLOR_RGB2GRAY)
    
    processed_images.append(gray_image)
    labels.append(row['name'])

# Augment the dataset
# #augmented_images, augmented_labels = augment_images(processed_images, labels)

print("Original dataset size:", len(processed_images))
print("Number of unique actors:", len(set(labels)))

# 3. Data train and eval prep
X = np.array([img.flatten() for img in processed_images])
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
pca = PCA(n_components=n_components_optimal, whiten=True, random_state=seed)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print(f"Shape of PCA-transformed data: {X_train_pca.shape}")
print(f"Explained variance by PCA components: {sum(pca.explained_variance_ratio_):.2f}")

classifier = SVC(kernel='linear', C=1, random_state=seed)
classifier.fit(X_train_pca, y_train)

y_pred = classifier.predict(X_test_pca)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

eigenfaces = pca.components_.reshape((n_components_optimal, x_y, x_y))
fig, axes = plt.subplots(2, 5, figsize=(15, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(eigenfaces[i], cmap='gray')
    ax.set_title(f"Eigenface {i+1}")
    ax.axis('off')
plt.show()

# Test on a new image (with alignment)
def recognize_image(image_path, pca, classifier, label_encoder):
    bgr_image = cv2.imread(image_path)
    if bgr_image is None:
        return None
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    faces = face_detector(rgb_image, 0)
    if len(faces) == 0:
        return None
    face_rect = faces[0].rect
    shape = shape_predictor(rgb_image, face_rect)
    aligned_face = dlib.get_face_chip(rgb_image, shape, size=x_y)
    gray_image = cv2.cvtColor(aligned_face, cv2.COLOR_RGB2GRAY)
    flattened_image = gray_image.flatten()
    pca_projection = pca.transform([flattened_image])
    prediction = classifier.predict(pca_projection)
    return label_encoder.inverse_transform(prediction)[0]

test_image_path = df.iloc[0]['location']
predicted_actor = recognize_image(test_image_path, pca, classifier, label_encoder)
print(f"Predicted actor: {predicted_actor}")

# Save models
joblib.dump(pca, 'eigenface_pca_model.pkl')
joblib.dump(classifier, 'eigenface_classifier.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
