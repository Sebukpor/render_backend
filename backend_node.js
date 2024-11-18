require('dotenv').config(); // Load environment variables
const express = require('express');
const multer = require('multer');
const { ComputerVisionClient } = require('@azure/cognitiveservices-computervision');
const { CognitiveServicesCredentials } = require('@azure/ms-rest-azure-js');
const { OpenAIClient, AzureKeyCredential } = require('@azure/openai');

const app = express();
const upload = multer();

// Azure Computer Vision setup
const computerVisionKey = process.env.AZURE_COMPUTER_VISION_KEY;
const computerVisionEndpoint = process.env.AZURE_COMPUTER_VISION_ENDPOINT;
const computerVisionClient = new ComputerVisionClient(
    new CognitiveServicesCredentials(computerVisionKey),
    computerVisionEndpoint
);

// Azure OpenAI setup
const openAiKey = process.env.AZURE_OPENAI_KEY;
const openAiEndpoint = process.env.AZURE_OPENAI_ENDPOINT;
const deploymentId = process.env.AZURE_OPENAI_DEPLOYMENT_ID;
const openai = new OpenAIClient(openAiEndpoint, new AzureKeyCredential(openAiKey));
const OPENAI_MODEL = deploymentId;

// Analyze route
app.post('/analyze', upload.single('image'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).send({ error: 'No image file uploaded.' });
        }

        // Extract TensorFlow.js prediction from request
        const tfPrediction = req.body.tfPrediction;

        // Step 1: Analyze image using Azure Computer Vision
        const analysis = await computerVisionClient.analyzeImageInStream(req.file.buffer, {
            visualFeatures: ['Description']
        });

        const description = analysis.description.captions[0]?.text || 'No description available';

        // Step 2: Combine TensorFlow.js prediction with Azure description
        const combinedInput = `
            TensorFlow.js Prediction: ${tfPrediction}
            Image Description: ${description}
        `;

        // Step 3: Generate findings with Azure OpenAI (ChatGPT)
        const openaiResponse = await openai.chatCompletions.create({
            model: OPENAI_MODEL,
            messages: [
                {
                    role: 'system',
                    content: "You are a radiologist assistant. Combine the TensorFlow.js prediction with the image description and provide findings and impressions."
                },
                {
                    role: 'user',
                    content: combinedInput
                }
            ]
        });

        const findings = openaiResponse.choices[0].message.content;

        // Step 4: Return results to the frontend
        res.send({
            description,
            findings
        });

    } catch (error) {
        console.error(error);
        res.status(500).send({ error: 'Error processing the request.' });
    }
});

// Start the server
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
