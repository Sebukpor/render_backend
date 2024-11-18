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
        const openaiResponse = await openai.ChatCompletion.create({
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
