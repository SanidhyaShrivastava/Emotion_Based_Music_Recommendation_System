// Example JavaScript for interacting with the API Gateway endpoints

// Update these variables with your actual API Gateway endpoint URLs
const analyzeEmotionUrl = 'https://2mu9j9umph.execute-api.us-east-1.amazonaws.com/prod/analyze-emotion';
const analyzeSentimentUrl = 'https://2mu9j9umph.execute-api.us-east-1.amazonaws.com/prod/sentiment-analysis';
const recommendMusicUrl = 'https://2mu9j9umph.execute-api.us-east-1.amazonaws.com/prod/music-recommendation';

function analyzeImage() {
    const imageInput = document.getElementById('imageInput');
    if (imageInput.files.length > 0) {
        const file = imageInput.files[0];
        // Assuming you have a function to convert the image to a suitable format for your API
        uploadImage(file).then(emotion => getMusicRecommendations(emotion));
    }
}

function analyzeText() {
    const textInput = document.getElementById('textInput').value;
    fetch(analyzeSentimentUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: textInput }),
    })
    .then(response => response.json())
    .then(data => getMusicRecommendations(data.emotion))
    .catch(error => console.error('Error:', error));
}

function getMusicRecommendations(emotion) {
    fetch(recommendMusicUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ emotion }),
    })
    .then(response => response.json())
    .then(data => {
        const recommendationsDiv = document.getElementById('recommendations');
        recommendationsDiv.innerHTML = ''; // Clear previous recommendations
        data.recommendations.forEach(song => {
            const p = document.createElement('p');
            p.textContent = song; // Assuming the response contains a list of song names
            recommendationsDiv.appendChild(p);
        });
    })
    .catch(error => console.error('Error:', error));
}

// Example function to upload an image
// You need to implement this based on how your API expects the image to be sent
function uploadImage(file) {
    // Convert file to a format your API expects (e.g., Base64) and POST to analyzeEmotionUrl
    // Return a promise that resolves with the analyzed emotion
}
