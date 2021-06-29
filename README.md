# object-identifier-and-camera-to-object-and-object-to-object-distance-estimator
Identifies different object classes(coco dataset), calculates camera to object distance and object to object distance.

Part 1: detects objectes defined in https://cocodataset.org/#home along with confidence score of each object.
SSD MobileNet architecture of dnn used.

Part 2: calculates distance between camera to object.
Haar-cascade-classifier(for frontal face detection) used.
for this part, instead of lena.png, use a captured image at a distance of 30 Inches(can be changed) from camera and measure the face width at that point.
similar to the below description:

![image](https://user-images.githubusercontent.com/75139237/123779777-28037d00-d8f0-11eb-94fc-f344e204eb8f.png)

The output for Part 1 and Part 2 is shown below:

https://user-images.githubusercontent.com/75139237/123778696-fd64f480-d8ee-11eb-8e1d-39fba3dbdef5.mp4

Part 3: calculates the midpoint of each bounding box and euclidean distance is calculated between two object midpoints:

![image](https://user-images.githubusercontent.com/75139237/123779239-985dce80-d8ef-11eb-9523-df63967d8958.png)

For single object identified: 

![image](https://user-images.githubusercontent.com/75139237/123779352-b4fa0680-d8ef-11eb-84ca-e051562bb19e.png)

Final output:

https://user-images.githubusercontent.com/75139237/123779034-5e8cc800-d8ef-11eb-9d60-da0dc9882615.mp4

![image](https://user-images.githubusercontent.com/75139237/123779525-e2df4b00-d8ef-11eb-9364-f041327a39ec.png)
