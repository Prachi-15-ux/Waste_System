document.addEventListener('DOMContentLoaded', () => {
    const dropArea = document.getElementById('drop-area');
    const fileElem = document.getElementById('fileElem');
    const browseBtn = document.getElementById('browse-btn');
    
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const clearBtn = document.getElementById('clear-btn');
    const predictBtn = document.getElementById('predict-btn');
    
    const loading = document.getElementById('loading');
    const resultCard = document.getElementById('result-card');
    
    // Result elements
    const itemDetected = document.getElementById('item-detected');
    const wasteCategory = document.getElementById('waste-category');
    const confidenceScore = document.getElementById('confidence-score');
    const recyclingSuggestion = document.getElementById('recycling-suggestion');
    
    let selectedFile = null;

    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    // Highlight drop area
    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => dropArea.classList.add('highlight'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => dropArea.classList.remove('highlight'), false);
    });

    // Handle dropped files
    dropArea.addEventListener('drop', handleDrop, false);
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length) handleFiles(files[0]);
    }

    // Handle browse button
    browseBtn.addEventListener('click', () => fileElem.click());
    fileElem.addEventListener('change', function() {
        if (this.files.length) handleFiles(this.files[0]);
    });

    function handleFiles(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file.');
            return;
        }
        
        selectedFile = file;
        
        // Show preview
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onloadend = function() {
            imagePreview.src = reader.result;
            dropArea.classList.add('hidden');
            previewContainer.classList.remove('hidden');
            resultCard.classList.add('hidden');
        }
    }

    // Clear selection
    clearBtn.addEventListener('click', () => {
        selectedFile = null;
        fileElem.value = '';
        imagePreview.src = '';
        previewContainer.classList.add('hidden');
        dropArea.classList.remove('hidden');
        resultCard.classList.add('hidden');
    });

    // Predict
    predictBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        // UI State
        predictBtn.disabled = true;
        clearBtn.disabled = true;
        loading.classList.remove('hidden');
        resultCard.classList.add('hidden');

        const formData = new FormData();
        formData.append('image', selectedFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                displayResult(data);
            } else {
                alert('Error: ' + (data.error || 'Prediction failed'));
            }
        } catch (error) {
            console.error('Error:', error);
            alert('Server error occurred. Please try again later.');
        } finally {
            predictBtn.disabled = false;
            clearBtn.disabled = false;
            loading.classList.add('hidden');
        }
    });

    function displayResult(data) {
        itemDetected.textContent = data.item_detected;
        wasteCategory.textContent = data.category;
        confidenceScore.textContent = `${data.confidence}%`;
        recyclingSuggestion.textContent = data.suggestion;

        // Setup styles based on category
        wasteCategory.className = 'value'; // Reset
        let catClass = 'cat-non-recyclable';
        let borderColor = 'var(--non-recyclable)';

        switch (data.category) {
            case 'Organic': 
                catClass = 'cat-organic'; borderColor = 'var(--organic)'; break;
            case 'Plastic': 
                catClass = 'cat-plastic'; borderColor = 'var(--plastic)'; break;
            case 'Glass': 
                catClass = 'cat-glass'; borderColor = 'var(--glass)'; break;
            case 'Paper': 
                catClass = 'cat-paper'; borderColor = 'var(--paper)'; break;
            case 'Metal': 
                catClass = 'cat-metal'; borderColor = 'var(--metal)'; break;
            case 'Hazardous': 
                catClass = 'cat-hazardous'; borderColor = 'var(--hazardous)'; break;
        }

        wasteCategory.classList.add(catClass);
        document.querySelector('.suggestion-box').style.borderLeftColor = borderColor;

        resultCard.classList.remove('hidden');
    }
});
