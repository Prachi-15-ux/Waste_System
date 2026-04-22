document.addEventListener("DOMContentLoaded", () => {
    const dropArea = document.getElementById("drop-area");
    const fileInput = document.getElementById("file-input");
    const uploadContent = document.getElementById("upload-content");
    const imagePreview = document.getElementById("image-preview");
    const classifyBtn = document.getElementById("classify-btn");
    const resultSection = document.getElementById("result-section");
    const loading = document.getElementById("loading");
    const errorMessage = document.getElementById("error-message");

    let selectedFile = null;

    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });

    // Highlight drop area when item is dragged over it
    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, unhighlight, false);
    });

    // Handle dropped files
    dropArea.addEventListener('drop', handleDrop, false);

    // Handle click to browse
    dropArea.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', function() {
        if (this.files && this.files[0]) {
            handleFiles(this.files[0]);
        }
    });

    classifyBtn.addEventListener('click', uploadAndClassify);

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    function highlight(e) {
        dropArea.classList.add('dragover');
    }

    function unhighlight(e) {
        dropArea.classList.remove('dragover');
    }

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;

        if (files && files[0]) {
            handleFiles(files[0]);
        }
    }

    function handleFiles(file) {
        if (!file.type.startsWith('image/')) {
            showError("Please upload an image file.");
            return;
        }

        selectedFile = file;
        hideError();
        
        // Show preview
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onloadend = function() {
            imagePreview.src = reader.result;
            imagePreview.classList.remove('hidden');
            uploadContent.classList.add('hidden');
            classifyBtn.classList.remove('hidden');
            classifyBtn.disabled = false;
            
            // Hide previous results
            resultSection.classList.add('hidden');
        }
    }

    async function uploadAndClassify() {
        if (!selectedFile) return;

        // UI states
        classifyBtn.disabled = true;
        loading.classList.remove('hidden');
        resultSection.classList.add('hidden');
        hideError();

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || "Something went wrong.");
            }

            displayResult(data);
        } catch (error) {
            showError(error.message);
        } finally {
            loading.classList.add('hidden');
            classifyBtn.disabled = false;
        }
    }

    function displayResult(data) {
        const className = data.class;
        const confidence = data.confidence;
        const suggestions = data.suggestions;

        // Set Badge
        const badge = document.getElementById('class-badge');
        badge.textContent = className;
        badge.className = `badge ${className.toLowerCase()}`;

        // Set Confidence
        document.getElementById('confidence-val').textContent = confidence;

        // Set Suggestions
        document.getElementById('method-val').textContent = suggestions.recycling_method || "N/A";
        document.getElementById('tips-val').textContent = suggestions.disposal_tips || "N/A";
        document.getElementById('impact-val').textContent = suggestions.environmental_impact || "N/A";

        // Show result section
        resultSection.classList.remove('hidden');
    }

    function showError(msg) {
        errorMessage.textContent = msg;
        errorMessage.classList.remove('hidden');
    }

    function hideError() {
        errorMessage.classList.add('hidden');
    }
});
