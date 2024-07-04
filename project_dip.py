
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
from PIL import Image as PILImage
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from google.colab import files
import pandas as pd
import io
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

from google.colab import drive
drive.mount('/content/drive/')

#preprocessing
def normalize_image(image):
    return image / 255.0

def sharpen_image(image, alpha=1.5, beta=-0.5, gamma=0):
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(image, alpha, blurred, beta, gamma)
    return sharpened, blurred

def apply_threshold(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary

def find_contours(binary_image):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def automatic_crop(image, contours):
    x, y, w, h = cv2.boundingRect(contours[0])
    cropped_image = image[y:y+h, x:x+w]
    return cropped_image

# Apply K-Means segmentation
def kmeans_segmentation(image, k=4):
    image = (image * 255).astype(np.uint8)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    X = img_gray.flatten().reshape(-1, 1)

    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X)

    segmented_img = kmeans.cluster_centers_[kmeans.labels_].reshape(64, 64)
    segmented_img = normalize_image(segmented_img)

    return segmented_img.flatten()

def show_image(image, title):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

dataset_dir = '/content/drive/MyDrive/DIP/'

sample_image_path = os.path.join(dataset_dir, 'Training', os.listdir(os.path.join(dataset_dir, 'Training'))[0], os.listdir(os.path.join(dataset_dir, 'Training', os.listdir(os.path.join(dataset_dir, 'Training'))[0]))[0])
sample_image = cv2.imread(sample_image_path)
sample_image_resized = cv2.resize(sample_image, (64, 64), interpolation=cv2.INTER_AREA)

sharpened_image, blurred_image = sharpen_image(sample_image_resized)
normalized_image = normalize_image(sharpened_image)
normalized_image_8u = (normalized_image * 255).astype(np.uint8)
binary_image = apply_threshold(sharpened_image)
contours = find_contours(binary_image)
cropped_image = automatic_crop(sharpened_image, contours)

show_image(cv2.cvtColor(sample_image_resized, cv2.COLOR_BGR2GRAY), 'Normal Image')
show_image(cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2GRAY), 'GaussianBlur Image')
show_image(normalized_image_8u, 'Normalized Image')
show_image(binary_image, 'Threshold Image')
contour_image = cv2.drawContours(cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR), contours, -1, (0, 255, 0), 1)
show_image(cv2.cvtColor(contour_image, cv2.COLOR_BGR2GRAY), 'Contour Image')
show_image(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY), 'Cropped Image')

train_data = []
train_labels = []

train_dir = os.path.join(dataset_dir, 'Training')
for class_name in os.listdir(train_dir):
    class_dir = os.path.join(train_dir, class_name)
    for image_name in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image_name)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
        image, _ = sharpen_image(image)
        binary_image = apply_threshold(image)
        contours = find_contours(binary_image)
        cropped_image = automatic_crop(image, contours)
        cropped_image_resized = cv2.resize(cropped_image, (64, 64), interpolation=cv2.INTER_AREA)
        cropped_image_normalized = normalize_image(cropped_image_resized)
        segmented_image = kmeans_segmentation(cropped_image_normalized)
        train_data.append(segmented_image)
        train_labels.append(class_name)

train_data = np.array(train_data)
train_labels = np.array(train_labels)

print('Finished processing training images.')
print(f'Total images processed: {len(train_data)}')

test_data = []
test_labels = []

test_dir = os.path.join(dataset_dir, 'Testing')
for class_name in os.listdir(test_dir):
    class_dir = os.path.join(test_dir, class_name)
    for image_name in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image_name)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
        image, _ = sharpen_image(image)
        binary_image = apply_threshold(image)
        contours = find_contours(binary_image)
        cropped_image = automatic_crop(image, contours)
        cropped_image_resized = cv2.resize(cropped_image, (64, 64), interpolation=cv2.INTER_AREA)  # Resize to 64x64
        cropped_image_normalized = normalize_image(cropped_image_resized)
        segmented_image = kmeans_segmentation(cropped_image_normalized)
        test_data.append(segmented_image)
        test_labels.append(class_name)

test_data = np.array(test_data)
test_labels = np.array(test_labels)

print('Finished processing test images.')
print(f'Total images processed: {len(test_data)}')

def show_examples(data, labels, classes, num_examples=4):
    fig, axes = plt.subplots(len(classes), num_examples, figsize=(num_examples*2, len(classes)*2))
    for i, class_name in enumerate(classes):
        class_indices = [index for index, label in enumerate(labels) if label == class_name]
        for j in range(num_examples):
            image_index = class_indices[j]
            image = data[image_index].reshape(64, 64)
            axes[i, j].imshow(image, cmap='gray')
            axes[i, j].axis('on')
            if j == 0:
                axes[i, j].set_title(class_name)
    plt.tight_layout()
    plt.show()

classes = list(set(train_labels))
show_examples(train_data, train_labels, classes)

classifier = svm.SVC() #kernel = rbf
classifier.fit(train_data, train_labels)
y_pred_svm = classifier.predict(test_data)
accuracy_svm = accuracy_score(test_labels, y_pred_svm)
print("Accuracy of SVM:", accuracy_svm)

#encode label menjadi angka
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
test_labels_encoded = label_encoder.transform(test_labels)

#reduksi data
pca = PCA(n_components=2)
train_data_2d = pca.fit_transform(train_data)
test_data_2d = pca.transform(test_data)

#pelatihan model
classifier_2d = svm.SVC()
classifier_2d.fit(train_data_2d, train_labels_encoded)

#plotting dengan meshgrid
x_min, x_max = train_data_2d[:, 0].min() - 1, train_data_2d[:, 0].max() + 1
y_min, y_max = train_data_2d[:, 1].min() - 1, train_data_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

#plot decision boundary
Z = classifier_2d.predict(np.c_[xx.ravel(), yy.ravel()]) #mengubah 2d menjadi 1d
Z = Z.reshape(xx.shape).astype(float) #kembali ke 2d

num_classes = len(np.unique(train_labels_encoded))
colors = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', '#FFFFAA'][:num_classes])
plt.contourf(xx, yy, Z, alpha=0.3, cmap=colors)
plt.scatter(train_data_2d[:, 0], train_data_2d[:, 1], c=train_labels_encoded, cmap=ListedColormap(['#FF0000', '#00FF00', '#0000FF', '#FF00FF'][:num_classes]), edgecolors='k', marker='o')
plt.title('SVM Decision Boundary')
plt.xlabel('Fitur x (PCA Component 1)')
plt.ylabel('Fitur y (PCA Component 2)')
plt.show()

#Tumor Pituitari (merah), Tidak Ada Tumor (hijau), Tumor Glioma (biru), dan Tumor Meningioma (ungu).

cm_svm = confusion_matrix(test_labels, y_pred_svm, labels=classes)
cm_svm_df = pd.DataFrame(cm_svm, index=classes, columns=classes)

print("\nSVM Confusion Matrix:")
print(cm_svm_df)

def calculate_metrics(cm):
    TN = cm.sum(axis=0) - cm.sum(axis=1)
    FP = cm.sum(axis=1) - np.diag(cm)
    FN = cm.sum(axis=0) - np.diag(cm)
    TP = np.diag(cm)t
    specificity = TN / (TN + FP)
    precision = TP / (TP + FP)
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    error = (FP + FN) / (TP + FP + TN + FN)
    return specificity, precision, accuracy, error

specificity, precision, accuracy, error = calculate_metrics(cm_svm)

print("\nSVM Specificity per class:", specificity)
print("SVM Precision per class:", precision)
print("SVM Accuracy per class:", accuracy)
print("SVM Error per class:", error)

plt.figure(figsize=(10, 7))
sns.heatmap(cm_svm_df, annot=True, fmt='d', cmap='Blues')
plt.title('SVM Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

kf = KFold(n_splits=5, shuffle=True, random_state=42)

svm_cv_scores = cross_val_score(classifier, train_data, train_labels, cv=kf, scoring='accuracy')
print("K-Fold Cross Validation scores for SVM:", svm_cv_scores)
print("Mean accuracy of SVM with K-Fold Cross Validation:", svm_cv_scores.mean())

uploaded = files.upload()
for image_name in uploaded.keys():
    image_path = image_name
    image = cv2.imread(image_path)
    image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
    image, _ = sharpen_image(image)
    binary_image = apply_threshold(image)
    contours = find_contours(binary_image)
    cropped_image = automatic_crop(image, contours)
    cropped_image_resized = cv2.resize(cropped_image, (64, 64), interpolation=cv2.INTER_AREA)
    cropped_image_normalized = normalize_image(cropped_image_resized)
    segmented_image = kmeans_segmentation(cropped_image_normalized)
    image_flat = segmented_image.flatten()

    predicted_class_knn = knn.predict([image_flat])
    print("Predicted class by KNN: ", predicted_class_knn[0])

    predicted_class_svm = classifier.predict([image_flat])
    print("Predicted class by SVM: ", predicted_class_svm[0])

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

def normalize_image(image):
    return image / 255.0

def sharpen_image(image, alpha=1.5, beta=-0.5, gamma=0):
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(image, alpha, blurred, beta, gamma)
    return sharpened, blurred

def apply_threshold(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary

def find_contours(binary_image):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def automatic_crop(image, contours):
    x, y, w, h = cv2.boundingRect(contours[0])
    cropped_image = image[y:y+h, x:x+w]
    return cropped_image

def kmeans_segmentation(image, k=4):
    image = (image * 255).astype(np.uint8)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    X = img_gray.flatten().reshape(-1, 1)

    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X)

    segmented_img = kmeans.cluster_centers_[kmeans.labels_].reshape(img_gray.shape)
    segmented_img = normalize_image(segmented_img)

    return segmented_img

def display_image_with_title(image, title):
    img_bytes = cv2.imencode('.png', image)[1].tobytes()
    img_widget = widgets.Image(value=img_bytes, format='png', width=256, height=256)
    return widgets.VBox([widgets.Label(title), img_widget])

def classify_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
    sharpened_image, blurred_image = sharpen_image(image)
    binary_image = apply_threshold(sharpened_image)
    contours = find_contours(binary_image)
    cropped_image = automatic_crop(sharpened_image, contours)
    cropped_image_resized = cv2.resize(cropped_image, (64, 64), interpolation=cv2.INTER_AREA)
    cropped_image_normalized = normalize_image(cropped_image_resized)
    segmented_image = kmeans_segmentation(cropped_image_normalized)
    image_flat = segmented_image.flatten()

    predicted_class_knn = knn.predict([image_flat])[0]
    predicted_class_svm = classifier.predict([image_flat])[0]

    return predicted_class_knn, predicted_class_svm, blurred_image, sharpened_image, binary_image, contours, cropped_image

def on_submit_button_clicked(b):
    clear_output(wait=True)
    display(title)
    for name, file_info in upload_widget.value.items():
        image_path = '/content/' + name
        with open(image_path, 'wb') as f:
            f.write(file_info['content'])

        predicted_class_knn, predicted_class_svm, blurred_image, sharpened_image, binary_image, contours, cropped_image = classify_image(image_path)

        display(widgets.HTML(f"<h2>Hasil Klasifikasi:</h2>"))
        display(widgets.HTML(f"<h3>KNN: {predicted_class_knn}</h3>"))
        display(widgets.HTML(f"<h3>SVM: {predicted_class_svm}</h3>"))

        img = PILImage.open(io.BytesIO(file_info['content']))
        img.thumbnail((256, 256))
        display(img)

        gaussian_img = display_image_with_title(cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2GRAY), 'GaussianBlur Image')
        threshold_img = display_image_with_title(binary_image, 'Threshold Image')
        contour_image = cv2.drawContours(cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR), contours, -1, (0, 255, 0), 1)
        contour_img = display_image_with_title(cv2.cvtColor(contour_image, cv2.COLOR_BGR2GRAY), 'Contour Image')
        cropped_img = display_image_with_title(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY), 'Cropped Image')

        display(widgets.HBox([gaussian_img, threshold_img, contour_img, cropped_img]))

    display(widgets.HBox([upload_widget, submit_button]))

title = widgets.HTML(value="<h1 style='color: #4285F4; font-family: Arial, sans-serif;'>NeuroScan: Aplikasi Klasifikasi Tumor Otak</h1>")

upload_widget = widgets.FileUpload(accept='image/*', multiple=False, description="Pilih Gambar", style={'button_color': '#4285F4'})

submit_button = widgets.Button(description="Submit", button_style='info', style={'button_color': '#4285F4'})
submit_button.on_click(on_submit_button_clicked)

buttons_box = widgets.VBox([upload_widget, submit_button])
display(HTML("""
<style>
    .widget-upload {flex-direction: column;}
    .widget-button {margin-top: 10px;}
    .widget-box {background-color: #f9f9f9; padding: 20px; border-radius: 10px; border: 1px solid #ccc;}
    .image-container {display: flex; flex-wrap: wrap; gap: 20px; justify-content: center;}
    .image-container > div {margin: 10px;}
    .widget-upload input[type="button"] {
        background-color: #4285F4;
        color: white;
        border-radius: 5px;
        padding: 5px 10px;
    }
    .widget-upload input[type="button"]:hover {
        background-color: #357ae8;
    }
</style>
"""))
display(title)
display(buttons_box)
