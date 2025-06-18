import express from 'express';
import cors from 'cors';
import path from 'path';
import { fileURLToPath } from 'url';
import { GoogleGenAI } from "@google/genai";

const app = express();
const PORT = 3010;

// Resolve __dirname for ES module
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Middleware
app.use(cors());
app.use(express.json());

// Serve static HTML file
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'templates', 'home(main).html'));
});

// Gemini AI setup
const genAI = new GoogleGenAI({ apiKey: "AIzaSyAAyqNaWhLg4ulsj0QV5pidLVyxrm-4bdw" });

app.post('/api/correct', async (req, res) => {
  try {
    const { text1 } = req.body;

    if (!text1) {
      return res.status(400).json({ error: "No text provided" });
    }

    const response = await genAI.models.generateContent({
      model: "gemini-2.0-flash",
      contents: `You will be given a sentence with potential errors in capitalization, grammar, and punctuation. Your task is to correct these errors and provide all corrected possible sentences with the mistakes I made along with the explanation in detail. The output should not have any introduction sentences like here is the answer for your question.: ${text1}`,
    });

    const correctedText = response.text;
    res.json({ correctedText });
  } catch (error) {
    console.error("Gemini API error:", error);
    res.status(500).json({
      error: "Failed to process text",
      details: error.message
    });
  }
});

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
