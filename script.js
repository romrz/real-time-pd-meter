import vision from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.4";

const { FaceLandmarker, FilesetResolver, DrawingUtils } = vision;

const VISION_PATH = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.4/wasm"
const MODEL_PATH = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
const IRIS_WIDTH_IN_MM = 12;

let videoWidth = 1920;
let videoHeight;

const videoElement = document.getElementsByClassName('input_video')[0];
const canvasElement = document.getElementsByClassName('output_canvas')[0];
const canvasCtx = canvasElement.getContext('2d');
const drawingUtils = new DrawingUtils(canvasCtx);

let tryonActive = false;
let faceMeshActive = true;
let faceLandmarker;
let runningMode = "VIDEO";
let lastVideoTime = -1;
let results = undefined;
let frameAspectRatio = 0;
let frameImage = new Image();

async function createFaceLandmarker() {
  const filesetResolver = await FilesetResolver.forVisionTasks(VISION_PATH);
  faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
    baseOptions: {
      modelAssetPath: MODEL_PATH,
      delegate: "GPU",
    },
    outputFaceBlendshapes: true,
    runningMode,
    numFaces: 1,
  });
}

function enableCamera(event) {
  if (! faceLandmarker) {
    console.log("Wait! faceLandmarker not loaded yet.");
    return;
  }

  navigator
  .mediaDevices
  .getUserMedia({video: {width: {ideal: videoWidth}}})
  .then((stream) => {
    videoElement.srcObject = stream;
    videoElement.addEventListener("loadeddata", predictWebcam);

    document.getElementById('loader').style.display = 'none';
    document.getElementsByClassName('ui')[0].style.display = 'flex';
  });
}

async function predictWebcam() {
  const radio = videoElement.videoHeight / videoElement.videoWidth;
  videoWidth = videoElement.videoWidth
  videoHeight = videoWidth * radio;
  canvasElement.width = videoElement.videoWidth;
  canvasElement.height = videoElement.videoHeight;
  
  let startTimeMs = performance.now();
  if (lastVideoTime !== videoElement.currentTime) {
    lastVideoTime = videoElement.currentTime;
    results = faceLandmarker.detectForVideo(videoElement, startTimeMs);
  }
  
  canvasCtx.reset();
  canvasCtx.restore();
  onResults(results)
  
  window.requestAnimationFrame(predictWebcam);
}

function getDistance(p1, p2) {
  return Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2) + Math.pow(p1.z - p2.z, 2));
}

function getDistanceInScreenCoordinates(p1, p2) {
  return Math.sqrt(Math.pow(p1.x * videoWidth - p2.x * videoWidth, 2) + Math.pow(p1.y * videoHeight - p2.y * videoHeight, 2));
}

function onResults(results) {
  if (results.faceLandmarks && results.faceLandmarks[0]) {
    let pupils = getPupils(results.faceLandmarks[0]);
    let nose = getNose(results.faceLandmarks[0]);
    
    let pupilsDistance = getDistanceInScreenCoordinates(pupils.left, pupils.right);

    let pdLeft = (IRIS_WIDTH_IN_MM / pupils.left.widthPx) * pupilsDistance
    let pdRight = (IRIS_WIDTH_IN_MM / pupils.right.widthPx) * pupilsDistance

    updatePD((pdLeft + pdRight) / 2);

    if (faceMeshActive) {
      drawFaceMesh(results)
      drawPoints(pupils, nose, results.faceLandmarks[0]); 
    }

    if (tryonActive) {
      drawTryonFrame(pupils, nose)
    }
  }
    
  canvasCtx.restore();
}

function getPupils(landmarks) {
  return {
    left: {
      x: (landmarks[FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS[0].start].x + landmarks[FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS[2].start].x) / 2.0,
      y: (landmarks[FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS[0].start].y + landmarks[FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS[2].start].y) / 2.0,
      z: (landmarks[FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS[0].start].z + landmarks[FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS[2].start].z) / 2.0,
      width: getDistance(landmarks[FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS[0].start], landmarks[FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS[2].start]),
      widthPx: getDistanceInScreenCoordinates(landmarks[FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS[0].start], landmarks[FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS[2].start]),
    },
    right: {
      x: (landmarks[FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS[0].start].x + landmarks[FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS[2].start].x) / 2.0,
      y: (landmarks[FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS[0].start].y + landmarks[FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS[2].start].y) / 2.0,
      z: (landmarks[FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS[0].start].z + landmarks[FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS[2].start].z) / 2.0,
      width: getDistance(landmarks[FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS[0].start], landmarks[FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS[2].start]),
      widthPx: getDistanceInScreenCoordinates(landmarks[FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS[0].start], landmarks[FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS[2].start]),
    }
  }
}

function getNose(landmarks) {
  return {
    x: (landmarks[197].x),
    y: (landmarks[197].y),
    z: (landmarks[197].z),
  }
}

function updatePD(pd) {
  document.getElementById('pd-score').innerHTML = pd.toFixed(0);
}

function drawPoints(pupils, nose, landmarks) {
  // Nose
  canvasCtx.fillStyle = 'orange';
  drawPoint(nose.x, nose.y);
 
  // Pupils
  canvasCtx.fillStyle = 'chartreuse';
  drawPoint(pupils.left.x, pupils.left.y);
  drawPoint(pupils.right.x, pupils.right.y);

  // Left Iris
  drawPoint(landmarks[FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS[2].start].x, landmarks[FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS[2].start].y);
  drawPoint(landmarks[FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS[0].start].x, landmarks[FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS[0].start].y);
  drawPoint(landmarks[FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS[1].start].x, landmarks[FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS[1].start].y);
  drawPoint(landmarks[FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS[3].start].x, landmarks[FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS[3].start].y);

  // Right Iris
  drawPoint(landmarks[FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS[2].start].x, landmarks[FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS[2].start].y);
  drawPoint(landmarks[FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS[0].start].x, landmarks[FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS[0].start].y);
  drawPoint(landmarks[FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS[1].start].x, landmarks[FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS[1].start].y);
  drawPoint(landmarks[FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS[3].start].x, landmarks[FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS[3].start].y);
}

function drawPoint(x, y) {
  let dotSize = 1

  canvasCtx.fillRect(x * videoWidth - dotSize, y * videoHeight - dotSize, dotSize * 2, dotSize * 2);
}

function drawTryonFrame(pupils, nose) { 
  let frameWidthPx = (pupils.left.widthPx / IRIS_WIDTH_IN_MM) * 153.0;

  let frameRotation = Math.atan2(
    (pupils.left.y - pupils.right.y) * videoHeight,
    (pupils.left.x - pupils.right.x ) * videoWidth
  );

  let frameHeightPx = frameWidthPx * frameAspectRatio;

  let framePositionPx = {
    x: nose.x * videoWidth - frameWidthPx / 2.0,
    y: nose.y * videoHeight - frameHeightPx / 2.0,
  }

  canvasCtx.translate(nose.x * videoWidth, nose.y * videoHeight);
  canvasCtx.rotate(frameRotation);
  canvasCtx.translate(- (nose.x * videoWidth), - (nose.y * videoHeight));
  canvasCtx.drawImage(
    frameImage,
    framePositionPx.x,
    framePositionPx.y,
    frameWidthPx,
    frameHeightPx,
  );

  canvasCtx.setTransform(1, 0, 0, 1, 0, 0);
}

function drawFaceMesh(results) {
  if (results.faceLandmarks) {
    for (const landmarks of results.faceLandmarks) {
      drawingUtils.drawConnectors(
        landmarks,
        FaceLandmarker.FACE_LANDMARKS_TESSELATION,
        { color: "#C0C0C030", lineWidth: 1 }
      );
      drawingUtils.drawConnectors(
        landmarks,
        FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE,
        { color: "#E0E0E0", lineWidth: 1 }
      );
      drawingUtils.drawConnectors(
        landmarks,
        FaceLandmarker.FACE_LANDMARKS_LEFT_EYE,
        { color: "#E0E0E0", lineWidth: 1 }
      );
      drawingUtils.drawConnectors(
        landmarks,
        FaceLandmarker.FACE_LANDMARKS_FACE_OVAL,
        { color: "#E0E0E0", lineWidth: 1 }
      );
      drawingUtils.drawConnectors(
        landmarks,
        FaceLandmarker.FACE_LANDMARKS_LIPS,
        { color: "#E0E0E0", lineWidth: 1 }
      );
      drawingUtils.drawConnectors(
        landmarks,
        FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS,
        { color: "#30FF30", lineWidth: 1 }
      );
      drawingUtils.drawConnectors(
        landmarks,
        FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS,
        { color: "#30FF30", lineWidth: 1 }
      );
    }
  }
}

function loadTryonFrameImage() {
  frameImage.onload = function () {
    frameAspectRatio = frameImage.height / frameImage.width;
  }

  frameImage.src = 'elwood-frame.png';
}

await createFaceLandmarker();
enableCamera();
loadTryonFrameImage();

// Event listeners
document.getElementById('toggle-tryon').addEventListener('click', function () {
  tryonActive = !tryonActive;
})
document.getElementById('toggle-face-mesh').addEventListener('click', function () {
  faceMeshActive = !faceMeshActive;
})