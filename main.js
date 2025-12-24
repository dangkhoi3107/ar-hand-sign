const videoEl = document.getElementById("webcam");
const canvasEl = document.getElementById("draw-canvas");
const canvasCtx = canvasEl.getContext("2d");

// UI Elements
const bigLabelEl = document.getElementById("big-prediction-label");
const confBarFillEl = document.getElementById("conf-bar-fill");
const fpsLabelEl = document.getElementById("fps-label");
const errorOverlayEl = document.getElementById("error-overlay");
const chkShowLandmarks = document.getElementById("chk-show-landmarks");

// =========================
// Application state
// =========================
const appState = {
  lastFrameTime: performance.now(),
  smoothedPoint: null,
};

// MediaPipe Hands state
let hands = null;
let handsResults = null;
let handsProcessing = false;

// ONNX Model state
let session = null;
let labelMapping = null;
let modelLoaded = false;

// =========================
// SEQUENCE CONFIG
// =========================
let SEQ_LEN = 30;
let FEAT_DIM = 63;

const PRED_STRIDE = 2;
// TĂNG VOTE_WIN LÊN ĐỂ GIẢM NHẢY LOẠN XẠ (Smoothing)
// Càng cao càng đầm (nhưng delay 1 chút), càng thấp càng nhạy (nhưng hay bị flicker)
const VOTE_WIN = 15;
const CONF_THRESH = 0.60;

let seqBuffer = [];
let frameCounter = 0;
let predHistory = [];

// =========================
// Hand Connections (for drawing skeleton)
// =========================
const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],         // Thumb
  [0,5],[5,6],[6,7],[7,8],         // Index
  [0,9],[9,10],[10,11],[11,12],    // Middle
  [0,13],[13,14],[14,15],[15,16],  // Ring
  [0,17],[17,18],[18,19],[19,20],  // Pinky
  [5,9],[9,13],[13,17],[17,5]      // Palm
];

// =========================
// Init
// =========================
async function initWebcam() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: 1280, height: 720 },
      audio: false,
    });

    videoEl.srcObject = stream;
    await new Promise((resolve) => {
      videoEl.onloadedmetadata = () => {
        videoEl.play();
        resolve();
      };
    });

    resizeCanvasToVideo();
    window.addEventListener("resize", resizeCanvasToVideo);

    initHands();
  } catch (err) {
    console.error(err);
    showError("Camera error. Please allow permissions.");
  }
}

function resizeCanvasToVideo() {
  if (videoEl.videoWidth && videoEl.videoHeight) {
    canvasEl.width = videoEl.videoWidth;
    canvasEl.height = videoEl.videoHeight;
  }
}

function updateFps(now) {
  const dt = now - appState.lastFrameTime;
  appState.lastFrameTime = now;
  const fps = 1000 / Math.max(dt, 1);
  fpsLabelEl.textContent = `FPS: ${fps.toFixed(0)}`;
}

function showError(msg) {
  errorOverlayEl.textContent = msg;
  errorOverlayEl.hidden = false;
}

// =========================
// MediaPipe
// =========================
function initHands() {
  if (typeof Hands === "undefined") {
    showError("MediaPipe Hands not loaded.");
    return;
  }

  hands = new Hands({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
  });

  hands.setOptions({
    maxNumHands: 1,
    modelComplexity: 1,
    minDetectionConfidence: 0.6, // Tăng nhẹ để đỡ bắt nhầm rác
    minTrackingConfidence: 0.5,
  });

  hands.onResults((results) => {
    handsResults = results;
  });

  loadGestureModel();
  requestAnimationFrame(frameLoop);
}

// =========================
// ONNX Model Loading
// =========================
async function loadGestureModel() {
  try {
    const metaPath = "./model_meta.json";
    const modelPath = "./sign_model.onnx";

    const metaResp = await fetch(metaPath);
    const meta = await metaResp.json();

    if (meta.labels) {
      labelMapping = { idxToLabel: (idx) => meta.labels[idx] };
    }
    if (meta.seq_len) SEQ_LEN = meta.seq_len;
    if (meta.feat_dim) FEAT_DIM = meta.feat_dim;

    session = await ort.InferenceSession.create(modelPath, {
      executionProviders: ['wasm']
    });

    modelLoaded = true;
    console.log("✅ Model loaded. Labels:", meta.labels);
    seqBuffer = [];
    predHistory = [];

  } catch (err) {
    console.error(err);
    showError("Model load failed. Check console.");
    modelLoaded = false;
  }
}

// =========================
// Logic: Feature Extraction
// =========================
function extractFeat63(landmarks) {
  const features = [];
  const wrist = landmarks[0];

  const dx = landmarks[9].x - wrist.x;
  const dy = landmarks[9].y - wrist.y;
  const dz = (landmarks[9].z || 0) - (wrist.z || 0);
  const scale = Math.sqrt(dx*dx + dy*dy + dz*dz) || 1.0;

  for (const lm of landmarks) {
    features.push((lm.x - wrist.x) / scale);
    features.push((lm.y - wrist.y) / scale);
    features.push(((lm.z || 0) - (wrist.z || 0)) / scale);
  }
  return features;
}

function zeroFeat63() {
  return new Array(63).fill(0);
}

function pushFrameToBuffer(feat63) {
  seqBuffer.push(feat63);
  if (seqBuffer.length > SEQ_LEN) seqBuffer.shift();
}

function majorityVote(arr) {
  const cnt = new Map();
  for (const x of arr) cnt.set(x, (cnt.get(x) || 0) + 1);
  let best = null, bestC = -1;
  for (const [k, v] of cnt.entries()) {
    if (v > bestC) {
      bestC = v;
      best = k;
    }
  }
  return best;
}

// =========================
// Logic: Inference
// =========================
async function runSequenceModelOnce() {
  if (!modelLoaded || !session || seqBuffer.length < SEQ_LEN) return null;

  frameCounter++;
  if (frameCounter % PRED_STRIDE !== 0) return null;

  try {
    const flatData = new Float32Array(SEQ_LEN * FEAT_DIM);
    for (let i = 0; i < SEQ_LEN; i++) {
      flatData.set(seqBuffer[i], i * FEAT_DIM);
    }

    const tensor = new ort.Tensor('float32', flatData, [1, SEQ_LEN, FEAT_DIM]);
    const feeds = { input: tensor };
    const results = await session.run(feeds);
    const outputData = results.output.data;

    let bestIdx = 0, bestVal = outputData[0];
    for (let i = 1; i < outputData.length; i++) {
      if (outputData[i] > bestVal) {
        bestVal = outputData[i];
        bestIdx = i;
      }
    }
    return { bestIdx, conf: bestVal };

  } catch (e) {
    console.error(e);
    return null;
  }
}

async function classifyBySequence(landmarksOrNull) {
  if (landmarksOrNull) pushFrameToBuffer(extractFeat63(landmarksOrNull));
  else pushFrameToBuffer(zeroFeat63());

  const pred = await runSequenceModelOnce();
  if (!pred) return null;

  // Smoothing logic
  predHistory.push(pred.bestIdx);
  if (predHistory.length > VOTE_WIN) predHistory.shift();

  const stableIdx = Number(majorityVote(predHistory));
  const modelLabel = labelMapping.idxToLabel(stableIdx);

  // Confidence bar update
  // Normalize confidence (sigmoid approximation roughly for display)
  // Or just use the raw score if it's already softmaxed. Assuming raw logits here:
  // Simple clamping for display visualization
  let displayConf = Math.min(Math.max(pred.conf + 5, 0) / 10, 1) * 100; // Fake normalization for UI
  // Note: if your model outputs Softmax, use pred.conf * 100 directly.

  if (modelLabel) {
    return { label: modelLabel, conf: pred.conf, uiConf: displayConf };
  }
  return null;
}

// =========================
// Drawing Skeleton
// =========================
function drawSkeleton(landmarks) {
  canvasCtx.save();
  const w = canvasEl.width;
  const h = canvasEl.height;

  // Draw connections
  canvasCtx.strokeStyle = "rgba(0, 255, 136, 0.8)";
  canvasCtx.lineWidth = 3;

  for (const [start, end] of HAND_CONNECTIONS) {
    const p1 = landmarks[start];
    const p2 = landmarks[end];
    canvasCtx.beginPath();
    canvasCtx.moveTo(p1.x * w, p1.y * h);
    canvasCtx.lineTo(p2.x * w, p2.y * h);
    canvasCtx.stroke();
  }

  // Draw points
  canvasCtx.fillStyle = "#ff0044";
  for (const lm of landmarks) {
    canvasCtx.beginPath();
    canvasCtx.arc(lm.x * w, lm.y * h, 4, 0, 2 * Math.PI);
    canvasCtx.fill();
  }

  canvasCtx.restore();
}

// =========================
// Main Loop
// =========================
async function frameLoop(now) {
  updateFps(now);

  if (hands && !handsProcessing) {
    handsProcessing = true;
    try {
      await hands.send({ image: videoEl });
    } finally {
      handsProcessing = false;
    }
  }

  // Clear Canvas
  canvasCtx.clearRect(0, 0, canvasEl.width, canvasEl.height);

  const hasHand = handsResults && handsResults.multiHandLandmarks && handsResults.multiHandLandmarks.length > 0;
  const landmarks = hasHand ? handsResults.multiHandLandmarks[0] : null;

  // 1. Draw Skeleton if enabled
  if (hasHand && chkShowLandmarks.checked) {
    drawSkeleton(landmarks);
  }

  // 2. Predict & Update UI
  if (modelLoaded && session) {
    const result = await classifyBySequence(landmarks);

    if (result) {
      bigLabelEl.textContent = result.label;

      // Đổi màu nếu confidence thấp
      if (result.conf < CONF_THRESH) {
        bigLabelEl.style.color = "#777"; // Mờ đi
      } else {
        bigLabelEl.style.color = "#00ffcc"; // Sáng lên
      }

      confBarFillEl.style.width = `${result.uiConf}%`;
    }
  }

  requestAnimationFrame(frameLoop);
}

// Entry
window.addEventListener("load", () => {
  initWebcam();
});