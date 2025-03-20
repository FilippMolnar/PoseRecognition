import mediapipe as mp
import cv2
from ultralytics import YOLO
import os
import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import shutil
import yaml

class Classifier:
    data = None
    
    def __init__(self, model_path=None):
        self.yolo_model = YOLO("yolov8n.pt")
        self.mp_pose = mp.solutions.pose
        self.pose_3d = self.mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        
    def get_data_paths_labels(self, folder_path):
        data = []
        for filename in os.listdir(folder_path):
            if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                image_path = os.path.join(folder_path, filename)
                label = os.path.basename(folder_path)
                data.append((image_path, label))
        return data
    
    def get_all_data(self, parent_folder):
        all_data = []
        for folder_name in os.listdir(parent_folder):
            folder_path = os.path.join(parent_folder, folder_name)
            if os.path.isdir(folder_path):
                data = self.get_data_paths_labels(folder_path)
                all_data.extend(data)
        self.data = all_data
    
    def count_people(self, image_path):
        results = self.yolo_model(image_path)
        
        people_count = sum(1 for box in results[0].boxes if int(box.cls) == 0)
        return people_count
    
    def resize_image(self, image, new_width=400):
        original_height, original_width = image.shape[:2]
        new_width = 400
        new_height = int((new_width / original_width) * original_height)
        resized_image = cv2.resize(image, (new_width, new_height))
        return resized_image
    
    def filter_data(self):
        if not self.data:
            return
        filtered_data = []
        skipped = []
        for image_path, label in self.data:
            image = cv2.imread(image_path)
            if image is None:
                continue  # Skip unreadable files

            resized_image = self.resize_image(image, 400)
            rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

            results = self.pose_3d.process(rgb_image)
            
            # Process the image
            # Save and display output
            # Store keypoints if detected
            data_kps = []
            if results.pose_landmarks:
                n_vis_landmarks = 0
                for i, landmark in enumerate(results.pose_landmarks.landmark):
                    if i not in range(1, 7):  # Skip eye landmarks
                        n_vis_landmarks += 1 if landmark.visibility > 0.7 else 0
                
                n_landmarks = len(results.pose_landmarks.landmark)-6
                print(f'{n_vis_landmarks}/{n_landmarks} landmarks visible, percentage: {n_vis_landmarks/n_landmarks*100:.2f}%')   

                if n_vis_landmarks/n_landmarks > 0.5:
                    filtered_data.append((image_path, label))
                else:
                    skipped.append(image_path)
                    self.mp_drawing.draw_landmarks(resized_image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                    name = image_path.split("\\")[-1]
                    print(f'skipped/{label}_{name}')
                    cv2.imwrite(f"skipped/{label}_{name}", resized_image)
        self.filtered_data = filtered_data
        
    def generate_3d_keypoints_csv(self, output_csv):
        header = ["label", "filename"]
        for i in range(33):  # BlazePose has 33 keypoints
            if i not in range(1, 7):  # Exclude eye landmarks (1-6)
                header.extend([f"x_{i}", f"y_{i}", f"z_{i}", f"visibility_{i}"])
                
        data = []
        for image_path, label in self.filtered_data:
            image = cv2.imread(image_path)
            if image is None:
                continue  # Skip unreadable files

            resized_image = self.resize_image(image, 400)
            rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

            results = self.pose_3d.process(rgb_image)
            
            if results.pose_landmarks:
                keypoints = [label, image_path]  # Store label and filename

                for i, landmark in enumerate(results.pose_landmarks.landmark):
                    if i not in range(1, 7):  # Skip eye landmarks
                        keypoints.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])

                data.append(keypoints)
                
                
        with open(output_csv, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            writer.writerows(data)

        print(f"Keypoints saved in {output_csv}")



    def train_random_forest(self, k=5):
        data = pd.read_csv("3d_keypoints.csv")

        # Extract features and labels
        labels = data["label"]
        filenames = data["filename"]
        features = data.drop(columns=["label", "filename"])
        
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        scores = []
        
        for train_index, test_index in skf.split(features, labels):
            X_train, X_test = features.iloc[train_index], features.iloc[test_index]
            y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]
            
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_train, y_train)
            
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            scores.append(acc)
            
        print(f"Cross-validation accuracy scores: {scores}")
        print(f"Mean accuracy: {sum(scores) / k:.4f}")
        
        # Train final model on full dataset
        final_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        final_clf.fit(features, labels)
        
        with open("random_forest_model.pkl", "wb") as f:
            pickle.dump(final_clf, f)

        print("Final model trained and saved.")

    def prepare_yolo_dataset(self, image_paths, labels, dataset_dir="yolo_dataset", test_size=0.2):
        """
        Prepares the dataset in the YOLO classification format.
        
        :param image_paths: List of image file paths.
        :param labels: List of corresponding labels.
        :param dataset_dir: Directory to store the YOLO-formatted dataset.
        :param test_size: Proportion of the dataset to use for testing.
        """
        class_names = sorted(set(labels))  # Unique class names
        class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
        
        train_dir = os.path.join(dataset_dir, "train")
        val_dir = os.path.join(dataset_dir, "val")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        
        # Split dataset
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        for img_path, label in zip(train_paths, train_labels):
            class_folder = os.path.join(train_dir, label)
            os.makedirs(class_folder, exist_ok=True)
            shutil.copy(img_path, os.path.join(class_folder, os.path.basename(img_path)))
        
        for img_path, label in zip(val_paths, val_labels):
            class_folder = os.path.join(val_dir, label)
            os.makedirs(class_folder, exist_ok=True)
            shutil.copy(img_path, os.path.join(class_folder, os.path.basename(img_path)))
        
        # Create dataset.yaml
        yaml_data = {
            "path": dataset_dir,
            "train": "train",
            "val": "val",
            "names": {idx: name for name, idx in class_to_idx.items()}
        }
        
        with open(os.path.join(dataset_dir, "dataset.yaml"), "w") as f:
            yaml.dump(yaml_data, f, default_flow_style=False)
        
        print(f"Dataset prepared at {dataset_dir}")
        return os.path.join(dataset_dir, "dataset.yaml")

    def train_yolo_classifier(self, dataset_yaml, model_name="yolov8n-cls.pt", epochs=10):
        """
        Trains a YOLOv8 classifier using the prepared dataset.
        
        :param dataset_yaml: Path to the dataset.yaml file.
        :param model_name: YOLOv8 classification model variant.
        :param epochs: Number of training epochs.
        """
        model = YOLO(model_name)
        model.train(data=dataset_yaml, epochs=epochs, device="cpu", batch=1, imgsz=640)
        
        print("Training completed.")
        return model

    def save_tuples_to_csv(self, data, filename):
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["path", "label"])
            writer.writerows(data)

    def load_tuples_from_csv(self, filename):
        paths, labels = [], []
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            for row in reader:
                paths.append(row[0])
                labels.append(row[1])
        return paths, labels

if __name__ == '__main__':
    classifier = Classifier()
    # classifier.get_all_data("data")
    # classifier.filter_data()
    # classifier.save_tuples_to_csv(classifier.filtered_data, "filtered_data.csv")
    # image_paths, labels = classifier.load_tuples_from_csv("filtered_data.csv")
    # classifier.generate_3d_keypoints_csv("3d_keypoints.csv")
    # classifier.train_random_forest(k=10)
    
    # dataset_yaml = classifier.prepare_yolo_dataset(image_paths, labels)
    dataset_yaml = r'C:\\Users\\filip\\Desktop\\TUDelft\\ComputerVision\\PoseRecognition\\yolo_dataset'
    trained_model = classifier.train_yolo_classifier(dataset_yaml, epochs=20)