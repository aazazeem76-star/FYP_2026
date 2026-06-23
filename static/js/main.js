/**
 * Facial & Retina Recognition Attendance System
 * Main JavaScript File
 */

// Global Variables
let cameraStream = null;
let cameraVideo = null;
let cameraCanvas = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', function () {
    initializeApp();
});

/**
 * Initialize Application
 */
function initializeApp() {
    // Add fade-in animation to cards
    const cards = document.querySelectorAll('.dashboard-card, .stat-card');
    cards.forEach((card, index) => {
        card.style.animationDelay = `${index * 0.1}s`;
        card.classList.add('fade-in');
    });

    // Auto-hide alerts after 5 seconds
    const alerts = document.querySelectorAll('.alert');
    alerts.forEach(alert => {
        setTimeout(() => {
            const bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        }, 5000);
    });
}

/**
 * Camera Functions
 */

/**
 * Start Camera
 */
async function startCamera(videoElementId, canvasElementId = null) {
    try {
        cameraVideo = document.getElementById(videoElementId);
        if (canvasElementId) {
            cameraCanvas = document.getElementById(canvasElementId);
        }

        if (!cameraVideo) {
            console.error('Video element not found');
            return false;
        }

        // Stop any previously running stream first
        if (cameraStream) {
            cameraStream.getTracks().forEach(t => t.stop());
            cameraStream = null;
        }

        // Request camera — prefer HD but fall back gracefully
        const constraints = {
            video: {
                width:  { ideal: 1280 },
                height: { ideal: 720  },
                facingMode: 'user'
            }
        };

        cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
        cameraVideo.srcObject = cameraStream;

        // Wait until the video is actually delivering frames
        await new Promise((resolve, reject) => {
            const onReady = () => {
                cameraVideo.removeEventListener('canplay',  onReady);
                cameraVideo.removeEventListener('error',    onError);
                resolve();
            };
            const onError = (e) => {
                cameraVideo.removeEventListener('canplay',  onReady);
                cameraVideo.removeEventListener('error',    onError);
                reject(e);
            };
            // Already ready (e.g. re-used element)
            if (cameraVideo.readyState >= 2) { resolve(); return; }
            cameraVideo.addEventListener('canplay', onReady);
            cameraVideo.addEventListener('error',   onError);
        });

        return true;
    } catch (error) {
        console.error('Error accessing camera:', error);
        return false;
    }
}

/**
 * Stop Camera
 */
function stopCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }

    if (cameraVideo) {
        cameraVideo.srcObject = null;
    }
}

/**
 * Capture Image from Camera
 * Captures the raw (unmirrored) video frame — consistent with how
 * face samples are stored during registration (also raw).
 * Guards against capturing before the video stream is ready.
 */
function captureImage() {
    if (!cameraVideo || !cameraCanvas) {
        console.error('Camera or canvas not initialized');
        return null;
    }

    const w = cameraVideo.videoWidth;
    const h = cameraVideo.videoHeight;

    // Guard: video must have actual dimensions and be playing
    if (!w || !h || cameraVideo.readyState < 2) {
        console.error('Video stream not ready yet (w=' + w + ' h=' + h + ' state=' + cameraVideo.readyState + ')');
        return null;
    }

    cameraCanvas.width  = w;
    cameraCanvas.height = h;

    const context = cameraCanvas.getContext('2d');
    context.drawImage(cameraVideo, 0, 0, w, h);

    // Export at high quality for reliable face encoding
    return cameraCanvas.toDataURL('image/jpeg', 0.98);
}

/**
 * Face Recognition Functions
 */

/**
 * Get CSRF Token from cookie
 */
function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

/**
 * Process Face Recognition
 */
async function processFaceRecognition(subjectId = null) {
    try {
        showLoading('Recognizing face...');

        // Capture image
        const imageData = captureImage();

        if (!imageData) {
            hideLoading();
            showAlert('error', 'Failed to capture image');
            return;
        }

        // Get CSRF token
        const csrftoken = getCookie('csrftoken');

        // Send to server
        const response = await fetch('/api/recognize-face/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': csrftoken
            },
            body: JSON.stringify({
                image: imageData,
                subject_id: subjectId
            })
        });

        const data = await response.json();

        hideLoading();

        if (data.success) {
            showAlert('success', data.message);

            // Show confidence score
            if (data.confidence) {
                showConfidence(data.confidence);
            }

            // Redirect after 2 seconds
            setTimeout(() => {
                window.location.href = '/dashboard/';
            }, 2000);
        } else {
            showAlert('error', data.message);
        }
    } catch (error) {
        hideLoading();
        console.error('Error:', error);
        showAlert('error', 'An error occurred during recognition');
    }
}

/**
 * Process Retina Recognition
 */
async function processRetinaRecognition(subjectId = null) {
    try {
        showLoading('Recognizing retina pattern...');

        // Capture image
        const imageData = captureImage();

        if (!imageData) {
            hideLoading();
            showAlert('error', 'Failed to capture image');
            return;
        }

        // Get CSRF token
        const csrftoken = getCookie('csrftoken');

        // Send to server
        const response = await fetch('/api/recognize-retina/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': csrftoken
            },
            body: JSON.stringify({
                image: imageData,
                subject_id: subjectId
            })
        });

        const data = await response.json();

        hideLoading();

        if (data.success) {
            showAlert('success', data.message);

            // Show confidence score
            if (data.confidence) {
                showConfidence(data.confidence);
            }

            // Redirect after 2 seconds
            setTimeout(() => {
                window.location.href = '/dashboard/';
            }, 2000);
        } else {
            showAlert('error', data.message);
        }
    } catch (error) {
        hideLoading();
        console.error('Error:', error);
        showAlert('error', 'An error occurred during recognition');
    }
}

/**
 * UI Helper Functions
 */

/**
 * Show Alert
 */
function showAlert(type, message) {
    const alertTypes = {
        'success': 'alert-success',
        'error': 'alert-danger',
        'warning': 'alert-warning',
        'info': 'alert-info'
    };

    const icons = {
        'success': 'fa-check-circle',
        'error': 'fa-exclamation-circle',
        'warning': 'fa-exclamation-triangle',
        'info': 'fa-info-circle'
    };

    const alertHtml = `
        <div class="alert ${alertTypes[type]} alert-dismissible fade show" role="alert">
            <i class="fas ${icons[type]} me-2"></i>
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;

    // Find or create alert container
    let alertContainer = document.getElementById('alert-container');
    if (!alertContainer) {
        alertContainer = document.createElement('div');
        alertContainer.id = 'alert-container';
        alertContainer.className = 'container mt-3';
        document.querySelector('main').prepend(alertContainer);
    }

    alertContainer.innerHTML = alertHtml;

    // Auto-hide after 5 seconds
    setTimeout(() => {
        const alert = alertContainer.querySelector('.alert');
        if (alert) {
            const bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        }
    }, 5000);
}

/**
 * Show Loading
 */
function showLoading(message = 'Processing...') {
    // Remove existing loading
    hideLoading();

    const loadingHtml = `
        <div id="loading-overlay" style="position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.7); z-index: 9999; display: flex; align-items: center; justify-content: center;">
            <div style="background: white; padding: 2rem 3rem; border-radius: 12px; text-align: center;">
                <div class="spinner-border text-primary mb-3" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <p class="mb-0 fw-bold">${message}</p>
            </div>
        </div>
    `;

    document.body.insertAdjacentHTML('beforeend', loadingHtml);
}

/**
 * Hide Loading
 */
function hideLoading() {
    const loading = document.getElementById('loading-overlay');
    if (loading) {
        loading.remove();
    }
}

/**
 * Show Confidence Score
 */
function showConfidence(score) {
    const percentage = (score * 100).toFixed(1);
    const confidenceHtml = `
        <div class="alert alert-info mt-3">
            <i class="fas fa-chart-line me-2"></i>
            <strong>Confidence Score:</strong> ${percentage}%
        </div>
    `;

    let alertContainer = document.getElementById('alert-container');
    if (alertContainer) {
        alertContainer.insertAdjacentHTML('beforeend', confidenceHtml);
    }
}

/**
 * Form Validation
 */
function validateForm(formId) {
    const form = document.getElementById(formId);
    if (!form) return false;

    let isValid = true;
    const inputs = form.querySelectorAll('input[required], select[required], textarea[required]');

    inputs.forEach(input => {
        if (!input.value.trim()) {
            isValid = false;
            input.classList.add('is-invalid');
        } else {
            input.classList.remove('is-invalid');
        }
    });

    return isValid;
}

/**
 * Format Date
 */
function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
    });
}

/**
 * Format Time
 */
function formatTime(timeString) {
    const time = new Date('2000-01-01 ' + timeString);
    return time.toLocaleTimeString('en-US', {
        hour: 'numeric',
        minute: '2-digit',
        hour12: true
    });
}

/**
 * Export functions for global use
 */
window.startCamera = startCamera;
window.stopCamera = stopCamera;
window.captureImage = captureImage;
window.processFaceRecognition = processFaceRecognition;
window.processRetinaRecognition = processRetinaRecognition;
window.showAlert = showAlert;
window.showLoading = showLoading;
window.hideLoading = hideLoading;
