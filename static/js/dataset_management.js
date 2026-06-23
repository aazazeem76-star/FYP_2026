/**
 * Dataset Management JavaScript
 * Handles biometric sample collection, dataset operations, and model training
 * Updated to use Event Delegation for better reliability
 */

console.log('Dataset Management JS Loaded v1.0');

// Utility function to get CSRF token
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

// Global state for collection
let collectionState = {
    face: { stream: null, active: false, count: 0, target: 0 },
    retina: { stream: null, active: false, count: 0, target: 0 }
};

document.addEventListener('DOMContentLoaded', function () {
    console.log('DOM Content Loaded - Attaching Listeners');

    // Attach Event Listeners for Forms
    const createDatasetForm = document.getElementById('createDatasetForm');
    if (createDatasetForm) createDatasetForm.addEventListener('submit', handleCreateDataset);

    const trainFaceForm = document.getElementById('trainFaceForm');
    if (trainFaceForm) trainFaceForm.addEventListener('submit', (e) => handleTrainModel(e, 'face'));

    const trainRetinaForm = document.getElementById('trainRetinaForm');
    if (trainRetinaForm) trainRetinaForm.addEventListener('submit', (e) => handleTrainModel(e, 'retina'));

    // Attach Event Listeners for Collection Buttons
    const startFaceBtn = document.getElementById('startFaceCollection');
    if (startFaceBtn) startFaceBtn.addEventListener('click', startFaceCollection);

    const startRetinaBtn = document.getElementById('startRetinaCollection');
    if (startRetinaBtn) startRetinaBtn.addEventListener('click', startRetinaCollection);

    // Event Delegation for Table Actions
    document.addEventListener('click', function (e) {
        const btn = e.target.closest('.action-btn');
        if (!btn) return;

        const action = btn.dataset.action;
        const id = btn.dataset.id;

        console.log(`Action triggered: ${action} on ID: ${id}`);

        if (action === 'view') viewDataset(id);
        else if (action === 'delete') deleteDataset(id);
        else if (action === 'test') testModel(id);
        else if (action === 'retrain') retrainModel(id, btn);
        else if (action === 'export') exportModel(id);
    });
});

// --- Dataset Operations ---

function viewDataset(datasetId) {
    fetch(`/api/dataset/${datasetId}/view/`)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                const d = data.dataset;
                const status = d.is_trained ? 'Trained' : 'Not Trained';
                const msg = `Dataset Details:\n\nName: ${d.name}\nType: ${d.type}\nSamples: ${d.samples}\nStatus: ${status}\nAccuracy: ${d.accuracy}%\nCreated: ${d.created_at}`;
                alert(msg);
            } else {
                alert('Error loading dataset details: ' + data.message);
            }
        })
        .catch(err => {
            console.error(err);
            alert('Error communicating with server');
        });
};

function deleteDataset(datasetId) {
    if (!confirm('Are you sure you want to delete this dataset from the database? This cannot be undone.')) return;

    fetch(`/api/dataset/${datasetId}/delete/`, {
        method: 'DELETE',
        headers: { 'X-CSRFToken': getCookie('csrftoken') }
    })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                alert('Dataset deleted successfully');
                location.reload();
            } else {
                alert('Error deleting dataset: ' + data.message);
            }
        })
        .catch(err => {
            console.error(err);
            alert('Error communicating with server');
        });
};

function testModel(modelId) {
    fetch(`/api/model/${modelId}/test/`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCookie('csrftoken')
        }
    })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                alert(`Test Results:\n\nModel: ${data.model}\nAccuracy: ${data.test_accuracy}%`);
            } else {
                alert('Test failed: ' + data.message);
            }
        })
        .catch(err => alert('Error testing model'));
};

function retrainModel(modelId, btn) {
    if (!confirm('Retrain this model? This will take a moment.')) return;

    const originalText = btn.innerHTML;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
    btn.disabled = true;

    fetch(`/api/model/${modelId}/retrain/`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCookie('csrftoken')
        }
    })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                alert(`Retraining Complete!\nNew Accuracy: ${data.accuracy}%`);
                location.reload();
            } else {
                alert('Retraining failed: ' + data.message);
                btn.innerHTML = originalText;
                btn.disabled = false;
            }
        })
        .catch(err => {
            alert('Error retraining model');
            btn.innerHTML = originalText;
            btn.disabled = false;
        });
};

function exportModel(modelId) {
    fetch(`/api/model/${modelId}/export/`)
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                alert(`Model Export Ready:\n\nFile: ${data.model_path}`);
            } else {
                alert('Export failed: ' + data.message);
            }
        })
        .catch(err => alert('Error exporting model'));
};


// --- Form Handlers ---

function handleCreateDataset(e) {
    e.preventDefault();
    const formData = new FormData(e.target);
    const data = {
        name: formData.get('name'),
        dataset_type: formData.get('dataset_type'),
        description: formData.get('description')
    };

    fetch('/api/dataset/create/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCookie('csrftoken')
        },
        body: JSON.stringify(data)
    })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                alert('Dataset created successfully!');
                location.reload();
            } else {
                alert('Error: ' + data.message);
            }
        })
        .catch(err => console.error(err));
}

function handleTrainModel(e, type) {
    e.preventDefault();
    const form = e.target;
    const formData = new FormData(form);
    const data = {
        dataset_id: formData.get('dataset_id'),
        epochs: formData.get('epochs'),
        batch_size: formData.get('batch_size')
    };

    // UI Updates
    const btn = form.querySelector('button[type="submit"]');
    const progressDiv = document.getElementById(`${type}TrainingProgress`);
    const progressBar = progressDiv ? progressDiv.querySelector('.progress-bar') : null;
    const originalText = btn.innerHTML;

    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Training...';
    
    console.log('Training started - showing progress bar');
    console.log('Progress div:', progressDiv);
    console.log('Progress bar element:', progressBar);
    
    // Show progress bar with indeterminate animation (100% width)
    if (progressDiv) {
        progressDiv.style.display = 'block';
        if (progressBar) {
            progressBar.style.width = '100%';
            progressBar.innerText = 'Training in progress...';
            console.log('Progress bar updated to 100% width');
        }
    }

    fetch('/api/dataset/train/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCookie('csrftoken')
        },
        body: JSON.stringify(data)
    })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                if (progressBar) {
                    progressBar.style.width = '100%';
                    progressBar.innerText = 'Complete!';
                    progressBar.classList.remove('progress-bar-animated');
                    progressBar.classList.add('bg-success');
                }
                setTimeout(() => {
                    alert(`Training Successful!\nAccuracy: ${data.accuracy}%`);
                    location.reload();
                }, 500);
            } else {
                alert('Training Failed: ' + data.message);
                btn.disabled = false;
                btn.innerHTML = originalText;
                if (progressDiv) progressDiv.style.display = 'none';
            }
        })
        .catch(err => {
            console.error(err);
            alert('An error occurred during training');
            btn.disabled = false;
            btn.innerHTML = originalText;
            if (progressDiv) progressDiv.style.display = 'none';
        });
}


// --- Sample Collection Logic ---

async function startFaceCollection() {
    const userId = document.getElementById('faceUserId').value;
    const count = parseInt(document.getElementById('faceSampleCount').value);

    if (!userId) return alert('Please select a user first.');

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        const video = document.getElementById('faceVideo');
        video.srcObject = stream;

        // Update UI
        const btn = document.getElementById('startFaceCollection');
        btn.disabled = true;
        btn.innerHTML = '<i class="fas fa-camera"></i> Collecting...';

        collectionState.face = {
            stream: stream,
            active: true,
            count: 0,
            target: count
        };

        collectOneSample('face', userId, video);

    } catch (err) {
        console.error(err);
        alert('Could not access camera. Please allow permissions.');
    }
}

async function startRetinaCollection() {
    const userId = document.getElementById('retinaUserId').value;
    const count = parseInt(document.getElementById('retinaSampleCount').value);

    if (!userId) return alert('Please select a user first.');

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        const video = document.getElementById('retinaVideo');
        video.srcObject = stream;

        // Update UI
        const btn = document.getElementById('startRetinaCollection');
        btn.disabled = true;
        btn.innerHTML = '<i class="fas fa-camera"></i> Collecting...';

        collectionState.retina = {
            stream: stream,
            active: true,
            count: 0,
            target: count
        };

        collectOneSample('retina', userId, video);

    } catch (err) {
        console.error(err);
        alert('Could not access camera. Please allow permissions.');
    }
}

function collectOneSample(type, userId, videoElement) {
    const state = collectionState[type];

    // Safety check: stop if inactive or target reached
    if (!state.active || state.count >= state.target) {
        stopCollection(type);
        return;
    }

    const canvas = document.getElementById(`${type}Canvas`);
    const progressBar = document.getElementById(`${type}Progress`);

    // Ensure video is playing
    if (videoElement.readyState === videoElement.HAVE_ENOUGH_DATA) {
        canvas.width = videoElement.videoWidth;
        canvas.height = videoElement.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(videoElement, 0, 0);

        const imageData = canvas.toDataURL('image/jpeg');

        fetch('/api/dataset/collect-samples/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCookie('csrftoken')
            },
            body: JSON.stringify({
                user_id: userId,
                sample_type: type, // 'face' or 'retina'
                image: imageData
            })
        })
            .then(res => res.json())
            .then(data => {
                if (data.success) {
                    state.count++;

                    // Update Progress
                    const pct = Math.round((state.count / state.target) * 100);
                    if (progressBar) {
                        progressBar.style.width = `${pct}%`;
                        progressBar.innerText = `${state.count}/${state.target}`;
                    }

                    // Check if done
                    if (state.count >= state.target) {
                        stopCollection(type);
                    } else {
                        // Collect next sample with slight delay
                        setTimeout(() => collectOneSample(type, userId, videoElement), 200);
                    }
                } else {
                    console.error('Sample Error:', data.message);
                    alert('Error processing sample: ' + data.message);
                    stopCollection(type);
                }
            })
            .catch(err => {
                console.error(err);
                stopCollection(type);
            });

    } else {
        // Video not ready, wait a bit
        setTimeout(() => collectOneSample(type, userId, videoElement), 100);
    }
}

function stopCollection(type) {
    const state = collectionState[type];
    // prevent multiple stops
    if (!state.active && !state.stream) return;

    state.active = false;

    if (state.stream) {
        state.stream.getTracks().forEach(track => track.stop());
        state.stream = null;
    }

    // Reset UI
    const btn = document.getElementById(type === 'face' ? 'startFaceCollection' : 'startRetinaCollection');
    if (btn) {
        btn.disabled = false;
        btn.innerHTML = `<i class="fas fa-play"></i> Start ${type === 'face' ? 'Face' : 'Retina'} Collection`;
    }

    if (state.count >= state.target) {
        alert(`${type} collection complete! ${state.count} samples collected.`);
        location.reload();
    }
}
