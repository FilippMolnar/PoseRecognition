from ultralytics import YOLO
import cv2

data = ['data\\bench_press\\1.jpg', 'data\\bench_press\\12.jpg', 'data\\bench_press\\14.png', 'data\\bench_press\\16.jpg', 'data\\bench_press\\19.jpg', 'data\\bench_press\\25.jpg', 'data\\bench_press\\27.png', 'data\\bench_press\\31.jpg', 'data\\bench_press\\34.jpg', 'data\\bench_press\\36.jpg', 'data\\bench_press\\37.jpg', 'data\\bench_press\\39.jpg', 'data\\bench_press\\42.jpg', 'data\\bench_press\\47.jpg', 'data\\bench_press\\49.jpg', 'data\\bench_press\\50.jpg', 'data\\bench_press\\52.jpg', 'data\\bench_press\\53.jpg', 'data\\bench_press\\60.png', 'data\\bench_press\\61.jpg', 'data\\bench_press\\64.png', 'data\\bench_press\\73.jpg', 'data\\bench_press\\75.jpeg', 'data\\bench_press\\83.jpg', 'data\\bench_press\\85.jpg', 'data\\bench_press\\89.jpg', 'data\\bench_press\\9.jpg', 'data\\deadlift\\31.jpg', 'data\\deadlift\\32.jpg', 'data\\deadlift\\35.jpg', 'data\\deadlift\\44.jpg', 'data\\deadlift\\47.jpg', 'data\\deadlift\\48.jpg', 'data\\deadlift\\50.jpeg', 'data\\deadlift\\55.jpg', 'data\\deadlift\\6.jpg', 'data\\deadlift\\67.jpg', 'data\\deadlift\\69.jpg', 'data\\deadlift\\70.jpeg', 'data\\deadlift\\71.jpg', 'data\\deadlift\\72.jpg', 'data\\deadlift\\8.jpg', 'data\\deadlift\\87.jpg', 'data\\deadlift\\88.jpg', 'data\\deadlift\\89.jpg', 'data\\deadlift\\9.jpg', 'data\\deadlift\\93.jpg', 'data\\deadlift\\95.jpg', 'data\\deadlift\\96.jpeg', 'data\\plank\\23.jpg', 'data\\plank\\31.jpg', 'data\\plank\\58.jpg', 'data\\plank\\61.jpg', 'data\\plank\\66.jpg', 'data\\plank\\71.jpg', 'data\\plank\\75.jpg', 'data\\plank\\79.jpg', 'data\\plank\\85.jpg', 'data\\plank\\89.jpg', 'data\\plank\\9.jpg', 'data\\plank\\98.jpg', 'data\\split_leap\\13.jpg', 'data\\split_leap\\18.jpg', 'data\\split_leap\\29.jpg', 'data\\split_leap\\31.jpg', 'data\\split_leap\\32.jpg', 'data\\split_leap\\39.jpg', 'data\\split_leap\\40.jpg', 'data\\split_leap\\42.jpg', 'data\\split_leap\\45.jpg', 'data\\split_leap\\47.jpg', 'data\\split_leap\\5.jpg', 'data\\split_leap\\50.jpg', 'data\\split_leap\\51.jpg', 'data\\split_leap\\52.jpg', 'data\\split_leap\\66.jpg', 'data\\split_leap\\67.jpg', 'data\\split_leap\\70.jpg', 'data\\split_leap\\72.jpg', 'data\\split_leap\\74.jpg',
        'data\\split_leap\\75.jpg', 'data\\split_leap\\77.jpg', 'data\\split_leap\\8.png', 'data\\split_leap\\80.jpg', 'data\\split_leap\\87.jpg', 'data\\split_leap\\88.jpg', 'data\\plank\\89.jpg', 'data\\plank\\9.jpg', 'data\\plank\\98.jpg', 'data\\split_leap\\13.jpg', 'data\\split_leap\\18.jpg', 'data\\split_leap\\29.jpg', 'data\\split_leap\\31.jpg', 'data\\split_leap\\32.jpg', 'data\\split_leap\\39.jpg', 'data\\split_leap\\40.jpg', 'data\\split_leap\\42.jpg', 'data\\split_leap\\45.jpg', 'data\\split_leap\\47.jpg', 'data\\split_leap\\5.jpg', 'data\\split_leap\\50.jpg', 'data\\split_leap\\51.jpg', 'data\\split_leap\\52.jpg', 'data\\split_leap\\66.jpg', 'data\\split_leap\\67.jpg', 'data\\split_leap\\70.jpg', 'data\\split_leap\\72.jpg', 'data\\split_leap\\74.jpg', 'data\\split_leap\\75.jpg', 'data\\plank\\89.jpg', 'data\\plank\\9.jpg', 'data\\plank\\98.jpg', 'data\\split_leap\\13.jpg', 'data\\split_leap\\18.jpg', 'data\\split_leap\\29.jpg', 'data\\split_leap\\31.jpg', 'data\\split_leap\\32.jpg', 'data\\split_leap\\39.jpg', 'data\\split_leap\\40.jpg', 'data\\split_leap\\42.jpg', 'data\\plank\\89.jpg', 'data\\plank\\9.jpg', 'data\\plank\\98.jpg', 'data\\split_leap\\13.jpg', 'data\\split_leap\\18.jpg', 'data\\split_leap\\29.jpg', 'data\\plank\\89.jpg', 'data\\plank\\9.jpg', 'data\\plank\\98.jpg', 'data\\split_leap\\13.jpg', 'data\\split_leap\\18.jpg', 'data\\split_leap\\29.jpg', 'data\\split_leap\\31.jpg', 'data\\split_leap\\32.jpg', 'data\\split_leap\\39.jpg', 'data\\split_leap\\40.jpg', 'data\\split_leap\\42.jpg', 'data\\split_leap\\45.jpg', 'data\\split_leap\\47.jpg', 'data\\split_leap\\5.jpg', 'data\\split_leap\\50.jpg', 'data\\split_leap\\51.jpg', 'data\\split_leap\\52.jpg', 'data\\split_leap\\66.jpg', 'data\\split_leap\\67.jpg', 'data\\split_leap\\70.jpg', 'data\\split_leap\\72.jpg', 'data\\split_leap\\74.jpg', 'data\\split_leap\\75.jpg', 'data\\split_leap\\77.jpg', 'data\\split_leap\\8.png', 'data\\split_leap\\80.jpg', 'data\\split_leap\\87.jpg', 'data\\split_leap\\88.jpg', 'data\\split_leap\\96.jpg', 'data\\split_leap\\97.jpg', 'data\\split_leap\\98.jpg', 'data\\split_leap\\99.jpg', 'data\\squat\\16.jpg',
        'data\\squat\\19.jpg', 'data\\squat\\22.jpg', 'data\\squat\\24.jpg', 'data\\squat\\39.jpg', 'data\\squat\\42.png', 'data\\squat\\43.jpg', 'data\\squat\\46.jpg', 'data\\squat\\47.jpg', 'data\\squat\\49.jpg', 'data\\squat\\54.jpg', 'data\\squat\\66.jpg', 'data\\squat\\70.jpeg', 'data\\squat\\84.jpg', 'data\\stag_leap\\10.jpg', 'data\\stag_leap\\12.jpg', 'data\\stag_leap\\13.jpg', 'data\\stag_leap\\14.jpg', 'data\\stag_leap\\20.png', 'data\\stag_leap\\21.png', 'data\\stag_leap\\27.png', 'data\\stag_leap\\28.png', 'data\\stag_leap\\29.png', 'data\\stag_leap\\3.jpg', 'data\\stag_leap\\30.png', 'data\\stag_leap\\31.png', 'data\\stag_leap\\36.png', 'data\\stag_leap\\37.png', 'data\\stag_leap\\40.png', 'data\\stag_leap\\41.png', 'data\\stag_leap\\42.png', 'data\\stag_leap\\47.png', 'data\\stag_leap\\50.png', 'data\\stag_leap\\53.png', 'data\\stag_leap\\58.jpg', 'data\\stag_leap\\59.jpg', 'data\\stag_leap\\60.jpg', 'data\\stag_leap\\62.jpg', 'data\\stag_leap\\63.jpg', 'data\\stag_leap\\64.jpg', 'data\\stag_leap\\65.jpg', 'data\\stag_leap\\66.jpg', 'data\\stag_leap\\68.jpg', 'data\\stag_leap\\7.jpg', 'data\\stag_leap\\72.jpg', 'data\\stag_leap\\73.jpg', 'data\\stag_leap\\76.jpg', 'data\\stag_leap\\77.jpg', 'data\\stag_leap\\78.jpg', 'data\\stag_leap\\79.jpg', 'data\\stag_leap\\8.png', 'data\\stag_leap\\9.jpg']

# Load a pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")  # Using the smallest YOLOv8 model

def detect_people(image_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        return  # Skip unreadable files

    # Resize while maintaining aspect ratio
    original_height, original_width = image.shape[:2]
    new_width = 400
    new_height = int((new_width / original_width) * original_height)
    resized_image = cv2.resize(image, (new_width, new_height))
    
    # Run YOLOv8 inference on the image
    results = model(resized_image)
    people_count = sum(1 for box in results[0].boxes if int(box.cls) == 0)
    print(people_count)
    if people_count == 1:
        return    
    # Draw bounding boxes for detected people (class 0 is 'person' in COCO dataset)
    for result in results:
        for box in result.boxes:
            if int(box.cls) == 0:  # 'person' class
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(resized_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    
    # Display the image
    
    cv2.imshow("YOLO People Detection", resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

for image_path in data:
    detect_people(image_path)