// import dotenv from 'dotenv';
import express from 'express';
import cors from 'cors';
import { GoogleGenAI } from "@google/genai";

const app = express();
const PORT = 3010;

// Middleware
app.use(cors());
app.use(express.json());

const genAI = new GoogleGenAI({ apiKey: "AIzaSyAAyqNaWhLg4ulsj0QV5pidLVyxrm-4bdw" });

app.post('/api/correct', async (req, res) => {
  try {
    const { text1 } = req.body;

    if (!text1) {
      return res.status(400).json({ error: "No text provided" });
    }

    const response = await genAI.models.generateContent({
            model: "gemini-2.0-flash",
            contents: `You will be given a sentence with potential errors in capitalization, grammar, and punctuation. Your task is to correct these errors and provide the corrected sentence with the mistakes I made. The output should not have any introduction sentences like here is the answer for your question.: ${text1}`,
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
