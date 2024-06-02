import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def load_images_from_folder(folder, size=(512, 512)):
    images = []
    labels = []
    class_names = []

    for dirname, _, filenames in os.walk(folder):
        for filename in filenames:
            filepath = os.path.join(dirname, filename)
            img = cv2.imread(filepath)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, size)
                label = os.path.basename(dirname)
                images.append(img)
                labels.append(label)
                if label not in class_names:
                    class_names.append(label)

    return images, labels, class_names

folder_path = 'C:\\Users\\natha\\Downloads\\Compressed\\DIP-20240531T090906Z-001\\DIP\\Training'
images, labels, class_names = load_images_from_folder(folder_path)
print(class_names)

# Fungsi untuk normalisasi gambar
def normalize_images(images):
    normalized_images = []
    for img in images:
        norm_img = img.astype(np.float32) / 255.0
        normalized_images.append(norm_img)
    return normalized_images

normalized_images = normalize_images(images)

# Fungsi untuk menerapkan Gaussian Blur
def apply_gaussian_blur(images, kernel_size=(5, 5), sigma=0):
    blurred_images = []
    for img in images:
        blurred = cv2.GaussianBlur(img, kernel_size, sigma)
        blurred_images.append(blurred)
    return blurred_images

blurred_images = apply_gaussian_blur(normalized_images)

# Fungsi untuk segmentasi gambar menggunakan KMeans
def segment_images(images, n_clusters=2):
    segmented_images = []
    for img in images:
        img_flat = img.reshape((-1, 3)) if img.ndim == 3 else img.reshape((-1, 1))
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(img_flat)
        segmented_image = kmeans.labels_.reshape(img.shape[:2])
        segmented_images.append(segmented_image)
    return segmented_images

segmented_images = segment_images(blurred_images)

# Menampilkan beberapa gambar hasil preprocessing
plt.figure(figsize=(17, 17))
for i, (norm_image, label) in enumerate(zip(segmented_images, labels)):
    plt.subplot(4, 4, i + 1)
    plt.imshow(norm_image, cmap='gray', vmin=0, vmax=1)
    plt.title(label)
    plt.axis("on")
plt.show()

label_encoder = LabelEncoder()
numeric_labels = label_encoder.fit_transform(labels)

data_features = np.array([img.flatten() for img in blurred_images])

X_train, X_test, y_train, y_test = train_test_split(data_features, numeric_labels, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

knn_predictions = knn.predict(X_test)

knn_accuracy = accuracy_score(y_test, knn_predictions)
print(f"KNN Accuracy: {knn_accuracy * 100:.2f}%")

svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

svm_predictions = svm.predict(X_test)

svm_accuracy = accuracy_score(y_test, svm_predictions)
print(f"SVM Accuracy: {svm_accuracy * 100:.2f}%")

def process_image(image_array):
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (512, 512))
        img = img.astype(np.float32) / 255.0
        return img.flatten()
    else:
        return None

image_path = folder_path = 'C:\\Users\\natha\\Downloads\\Compressed\\DIP-20240531T090906Z-001\\DIP\\Testing\\image(1).jpg'
uploaded = cv2.imread(image_path)
pov = input("Inputkan POV image (tampak_atas/tampak_samping/tampak_bawah) : ")

# Menyaring labels sesuai dengan POV yang dimasukkan
filtered_class_names = [name for name in class_names if pov in name]
filtered_labels = [label for label in labels if any(pov in cls for cls in filtered_class_names)]
filtered_data_features = [data_features[i] for i, label in enumerate(labels) if any(pov in cls for cls in filtered_class_names)]

for file_name in uploaded.keys():
    print(f'User uploaded file "{file_name}" with length {len(uploaded[file_name])} bytes')

    image_array = np.frombuffer(uploaded[file_name], np.uint8)

    processed_image = process_image(image_array)

    if processed_image is not None:
        knn_prediction = knn.predict([processed_image])
        svm_prediction = svm.predict([processed_image])

        knn_label = label_encoder.inverse_transform(knn_prediction)
        svm_label = label_encoder.inverse_transform(svm_prediction)

        print(f"KNN Prediction: {knn_label[0]}")
        print(f"SVM Prediction: {svm_label[0]}")
    else:
        print("Failed to load or process the image.")