// DOM elements
const newsText = document.getElementById('newsText');
const analyzeBtn = document.getElementById('analyzeBtn');
const clearBtn = document.getElementById('clearBtn');
const sampleBtn = document.getElementById('sampleBtn');
const charCount = document.getElementById('charCount');
const resultSection = document.getElementById('resultSection');
const loadingSection = document.getElementById('loadingSection');
const predictionLabel = document.getElementById('predictionLabel');
const confidenceValue = document.getElementById('confidenceValue');
const confidenceBar = document.getElementById('confidenceBar');
const confidenceLevel = document.getElementById('confidenceLevel');
const wordCount = document.getElementById('wordCount');
const realProb = document.getElementById('realProb');
const fakeProb = document.getElementById('fakeProb');
const processedText = document.getElementById('processedText');
const detailsIcon = document.getElementById('detailsIcon');
const detailsContent = document.getElementById('detailsContent');

// Sample news articles for quick testing
const sampleTexts = [
    "BREAKING: Scientists at NASA have confirmed that the Moon is actually made of cheese. This discovery was made after analyzing lunar samples brought back by the Apollo missions.",
    "Newly elected officials in Congress have proposed a bill to address climate change by reducing carbon emissions through investment in renewable energy sources and electric vehicle infrastructure.",
    "LOCAL NEWS: City council approves new budget for downtown revitalization project aimed at improving infrastructure and attracting new businesses to the area."
];

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    // Update character count on input
    newsText.addEventListener('input', updateCharCount);
    
    // Set up button click handlers
    analyzeBtn.addEventListener('click', analyzeText);
    clearBtn.addEventListener('click', clearText);
    sampleBtn.addEventListener('click', loadSample);
    
    // Initialize character count
    updateCharCount();
});

// Update the character count display
function updateCharCount() {
    const count = newsText.value.length;
    charCount.textContent = `${count} character${count !== 1 ? 's' : ''}`;
    
    // Enable/disable analyze button based on input
    if (count < 10) {
        analyzeBtn.disabled = true;
        analyzeBtn.classList.add('disabled');
    } else {
        analyzeBtn.disabled = false;
        analyzeBtn.classList.remove('disabled');
    }
}

// Load a random sample text
function loadSample() {
    const randomIndex = Math.floor(Math.random() * sampleTexts.length);
    newsText.value = sampleTexts[randomIndex];
    updateCharCount();
    
    // Scroll to input area if needed
    newsText.scrollIntoView({ behavior: 'smooth' });
    newsText.focus();
}

// Clear the input text
function clearText() {
    newsText.value = '';
    updateCharCount();
    hideResults();
    newsText.focus();
}

// Hide the results section
function hideResults() {
    resultSection.style.display = 'none';
}

// Show the loading indicator
function showLoading() {
    loadingSection.style.display = 'block';
    hideResults();
}

// Hide the loading indicator
function hideLoading() {
    loadingSection.style.display = 'none';
}

// Main function to analyze text
function analyzeText() {
    const text = newsText.value.trim();
    
    // Validate input
    if (text.length < 10) {
        showError('Please enter at least 10 characters of text to analyze.');
        return;
    }
    
    // Show loading indicator
    showLoading();
    
    // Make API call to server
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: text })
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(err => {
                throw new Error(err.message || err.error || 'Server error');
            });
        }
        return response.json();
    })
    .then(data => {
        // Process successful response
        displayResults(data);
    })
    .catch(error => {
        // Handle errors
        console.error('Error:', error);
        showError(error.message || 'An error occurred while analyzing the text.');
    })
    .finally(() => {
        // Hide loading indicator
        hideLoading();
    });
}

// Display results in the UI
function displayResults(data) {
    // Show result section
    resultSection.style.display = 'block';
    
    // Set prediction label
    predictionLabel.textContent = data.prediction;
    predictionLabel.className = 'prediction-label';
    
    // Add appropriate class based on prediction
    if (data.prediction === 'Real') {
        predictionLabel.classList.add('real');
    } else if (data.prediction === 'Fake') {
        predictionLabel.classList.add('fake');
    } else {
        predictionLabel.classList.add('uncertain');
    }
    
    // Update confidence information
    const confidencePercent = Math.round(data.confidence * 100);
    confidenceValue.textContent = `${confidencePercent}%`;
    confidenceBar.style.width = `${confidencePercent}%`;
    
    // Change progress bar color based on confidence
    if (confidencePercent > 80) {
        confidenceBar.style.backgroundColor = 'var(--secondary-color)';
    } else if (confidencePercent > 60) {
        confidenceBar.style.backgroundColor = 'var(--primary-color)';
    } else {
        confidenceBar.style.backgroundColor = 'var(--warning-color)';
    }
    
    // Update detailed info
    wordCount.textContent = data.word_count || 0;
    confidenceLevel.textContent = data.confidence_level || 
        (data.confidence > 0.8 ? 'High' : data.confidence > 0.6 ? 'Medium' : 'Low');
    
    // Update probabilities
    const probabilities = data.probabilities || {
        fake: 1 - data.confidence,
        real: data.confidence
    };
    
    realProb.textContent = `${Math.round(probabilities.real * 100)}%`;
    fakeProb.textContent = `${Math.round(probabilities.fake * 100)}%`;
    
    // Update processed text
    processedText.textContent = data.processed_text || '';
    
    // Scroll to results
    resultSection.scrollIntoView({ behavior: 'smooth' });
}

// Toggle details section
function toggleDetails() {
    const detailsToggle = document.querySelector('.details-toggle');
    detailsToggle.classList.toggle('active');
    
    if (detailsContent.classList.contains('active')) {
        detailsContent.classList.remove('active');
        detailsContent.style.maxHeight = null;
    } else {
        detailsContent.classList.add('active');
        detailsContent.style.maxHeight = detailsContent.scrollHeight + 'px';
    }
}

// Show error message
function showError(message) {
    // You could create a dedicated error UI component
    // For now, we'll use an alert
    alert(message);
} 