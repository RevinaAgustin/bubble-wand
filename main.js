import { FilesetResolver, HandLandmarker, DrawingUtils } from '@mediapipe/tasks-vision';

let handLandmarker;
let runningMode = "VIDEO";
let webcamRunning = false;

const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const enableWebcamButton = document.getElementById("enableWebcamButton");
const instruction = document.getElementById("instruction");

let bubbles = [];
let lastSpawnTime = 0;

class Bubble {
    constructor(x, y) {
        this.x = x;
        this.y = y;
        this.radius = Math.random() * 20 + 20; 
        this.vx = Math.random() * 4 - 2;
        this.vy = Math.random() * -3 - 2;
        this.alpha = Math.random() * 0.3 + 0.4;
        this.isPopping = false;
        this.popProgress = 0;
        this.popped = false;

        this.popAngle = Math.random() * Math.PI * 2; 

        this.hue1 = Math.random() * 360;
        this.hue2 = (this.hue1 + 60) % 360;

        this.imgCanvas = document.createElement('canvas');
        const padding = 10;
        this.size = (this.radius + padding) * 2;
        this.imgCanvas.width = this.size;
        this.imgCanvas.height = this.size;
        const octx = this.imgCanvas.getContext('2d');
        const cx = this.size / 2;
        const cy = this.size / 2;

        octx.filter = 'blur(0.8px)';

        octx.beginPath();
        octx.arc(cx, cy, this.radius, 0, 2 * Math.PI);
        octx.fillStyle = `rgba(255, 255, 255, ${this.alpha * 0.15})`;
        octx.fill();

        let grad = octx.createRadialGradient(cx, cy, this.radius * 0.5, cx, cy, this.radius);
        grad.addColorStop(0, "rgba(255, 255, 255, 0)");
        grad.addColorStop(0.7, `hsla(${this.hue1}, 100%, 75%, ${this.alpha * 0.5})`);
        grad.addColorStop(0.9, `hsla(${this.hue2}, 100%, 75%, ${this.alpha * 0.7})`);
        grad.addColorStop(1, `rgba(255, 255, 255, ${this.alpha * 0.9})`);

        octx.beginPath();
        octx.arc(cx, cy, this.radius, 0, 2 * Math.PI);
        octx.fillStyle = grad;
        octx.fill();

        octx.filter = 'blur(1.5px)';
        octx.beginPath();
        octx.ellipse(cx - this.radius * 0.35, cy - this.radius * 0.45, this.radius * 0.3, this.radius * 0.15, -Math.PI / 6, 0, 2 * Math.PI);
        octx.fillStyle = `rgba(255, 255, 255, ${this.alpha * 0.8})`;
        octx.fill();

        octx.beginPath();
        octx.ellipse(cx + this.radius * 0.4, cy + this.radius * 0.4, this.radius * 0.15, this.radius * 0.05, -Math.PI / 4, 0, 2 * Math.PI);
        octx.fillStyle = `rgba(255, 255, 255, ${this.alpha * 0.5})`;
        octx.fill();
    }

    update() {
        if (!this.isPopping) {
            this.x += this.vx;
            this.y += this.vy;
            this.vx += (Math.random() - 0.5) * 0.3;
            if (this.vx > 3) this.vx = 3;
            if (this.vx < -3) this.vx = -3;
        } else {
            this.popProgress += 0.15;
            if (this.popProgress >= 1.0) {
                this.popped = true;
            }
        }
    }

    draw(ctx) {
        if (this.popped) return;

        ctx.save();
        ctx.translate(this.x, this.y);

        if (!this.isPopping) {

            ctx.drawImage(this.imgCanvas, -this.size/2, -this.size/2);
        } else {

            const numParticles = 8;
            for (let i = 0; i < numParticles; i++) {
                const angle = (i / numParticles) * 2 * Math.PI;
                const dist = this.radius + (this.popProgress * 30);
                const px = Math.cos(angle) * dist;
                const py = Math.sin(angle) * dist;

                const alpha = Math.max(0, 1.0 - this.popProgress);
                ctx.beginPath();
                ctx.arc(px, py, 2.5, 0, 2 * Math.PI);
                ctx.fillStyle = `rgba(255, 255, 255, ${alpha})`;
                ctx.fill();
            }
        }
        ctx.restore();
    }
}

async function initializeModel() {
    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
    );
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            delegate: "GPU"
        },
        runningMode: runningMode,
        numHands: 2
    });

    instruction.innerText = "Model Siap!";
    enableWebcamButton.style.display = "inline-block";
}

initializeModel();

function hasGetUserMedia() {
    return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

if (hasGetUserMedia()) {
    enableWebcamButton.addEventListener("click", enableCam);
} else {
    console.warn("getUserMedia() is not supported by your browser");
    instruction.innerText = "Kamera tidak didukung di browser ini.";
}

function enableCam(event) {
    if (!handLandmarker) {
        console.log("Menunggu model dimuat...");
        return;
    }

    webcamRunning = true;
    enableWebcamButton.style.display = "none";
    instruction.innerText = "Tunjuk untuk buat bubble, buka telapak untuk pecahin!";

    const constraints = { video: { width: 1280, height: 720, facingMode: "user" } };

    navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
        video.srcObject = stream;
        video.addEventListener("loadeddata", predictWebcam);
    });
}

function getIndexFingerIfPointing(landmarks) {
    const indexTip = landmarks[8].y;
    const indexPip = landmarks[6].y;

    const middleTip = landmarks[12].y;
    const middlePip = landmarks[10].y;

    const ringTip = landmarks[16].y;
    const ringPip = landmarks[14].y;

    const pinkyTip = landmarks[20].y;
    const pinkyPip = landmarks[18].y;

    const isIndexStraight = indexTip < indexPip;
    const areOthersFolded = (middleTip > middlePip) && (ringTip > ringPip) && (pinkyTip > pinkyPip);

    if (isIndexStraight && areOthersFolded) {
        return landmarks[8];
    }
    return null;
}

function isOpenHand(landmarks) {
    const indexTip = landmarks[8].y;
    const indexPip = landmarks[6].y;

    const middleTip = landmarks[12].y;
    const middlePip = landmarks[10].y;

    const ringTip = landmarks[16].y;
    const ringPip = landmarks[14].y;

    const pinkyTip = landmarks[20].y;
    const pinkyPip = landmarks[18].y;

    return (indexTip < indexPip) && (middleTip < middlePip) && (ringTip < ringPip) && (pinkyTip < pinkyPip);
}

let lastVideoTime = -1;
let results = undefined;
const drawingUtils = new DrawingUtils(canvasCtx);

function predictWebcam() {
    if (canvasElement.width !== video.videoWidth) {
        canvasElement.width = video.videoWidth;
        canvasElement.height = video.videoHeight;
    }

    if (lastVideoTime !== video.currentTime) {
        lastVideoTime = video.currentTime;
        results = handLandmarker.detectForVideo(video, performance.now());
    }

    canvasCtx.drawImage(video, 0, 0, canvasElement.width, canvasElement.height);

    let pointingTips = [];
    let openHand = false;

    if (results && results.landmarks) {
        for (const landmarks of results.landmarks) {
            // MENGGAMBAR KERANGKA TANGAN SECARA MANUAL (SANGAT CEPAT)
            // Menggantikan DrawingUtils bawaan yang lambat dan memakan banyak CPU
            canvasCtx.strokeStyle = "rgba(255, 255, 255, 0.4)";
            canvasCtx.lineWidth = 2;
            canvasCtx.beginPath();
            for (const connection of HandLandmarker.HAND_CONNECTIONS) {
                const start = landmarks[connection.start];
                const end = landmarks[connection.end];
                canvasCtx.moveTo(start.x * canvasElement.width, start.y * canvasElement.height);
                canvasCtx.lineTo(end.x * canvasElement.width, end.y * canvasElement.height);
            }
            canvasCtx.stroke();

            canvasCtx.fillStyle = "rgba(255, 255, 255, 0.8)";
            canvasCtx.beginPath();
            for (const lm of landmarks) {
                const px = lm.x * canvasElement.width;
                const py = lm.y * canvasElement.height;
                canvasCtx.moveTo(px + 3, py);
                canvasCtx.arc(px, py, 3, 0, 2 * Math.PI);
            }
            canvasCtx.fill();

            if (isOpenHand(landmarks)) {
                openHand = true;
            } 

            else {
                let indexTip = getIndexFingerIfPointing(landmarks);
                if (indexTip) {
                    pointingTips.push(indexTip);
                }
            }
        }
    }

    const currentTime = performance.now();

    if (openHand) {
        for (let b of bubbles) {
            if (!b.isPopping) b.isPopping = true;
        }
        instruction.innerText = "POP! 💥";
        instruction.style.color = "#ff6b6b";
    } else if (pointingTips.length > 0) {
        if (currentTime - lastSpawnTime > 60) {

            pointingTips.forEach(tip => {
                let px = tip.x * canvasElement.width;
                let py = tip.y * canvasElement.height;
                bubbles.push(new Bubble(px, py));
            });
            lastSpawnTime = currentTime;
        }
        instruction.innerText = "Spawning Bubbles... 🫧";
        instruction.style.color = "#a1c4fd";
    } else {
        instruction.innerText = "Tunjuk untuk buat bubble, buka telapak untuk pecahin!";
        instruction.style.color = "white";
    }

    for (let b of bubbles) {
        b.update();
        b.draw(canvasCtx);
    }

    bubbles = bubbles.filter(b => !b.popped && b.y + b.radius > -50 && b.x > -100 && b.x < canvasElement.width + 100);

    if (webcamRunning === true) {
        window.requestAnimationFrame(predictWebcam);
    }
}
