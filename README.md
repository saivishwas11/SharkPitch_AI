# Athena Task - Shark Tank AI Pitch Analyzer

A full-stack application for analyzing startup pitches using AI-powered multi-agent systems. The project consists of a Python backend for pitch analysis and a React frontend for visualization.

## Project Structure

```
Athena_Task/
├── backend/              # Python backend application
│   ├── agents/          # Multi-agent system components
│   ├── content/         # Content analysis agents
│   ├── graph/           # LangGraph workflow definitions
│   ├── sharks/          # Shark Tank evaluator agents
│   ├── utils/           # Utility functions and helpers
│   ├── main.py          # Main entry point
│   ├── config.py        # Configuration settings
│   └── requirements.txt # Python dependencies
│
├── frontend/            # React frontend application
│   ├── src/            # Source code
│   ├── public/         # Static assets
│   ├── package.json    # Node.js dependencies
│   └── vite.config.ts  # Vite configuration
│
└── README.md           # This file
```

## Backend Setup

### Prerequisites

- Python 3.8+
- pip or conda

### Installation

1. Navigate to the backend directory:

   ```bash
   cd backend
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   Or using conda:

   ```bash
   conda env create -f environment.yml
   conda activate athena_assignment
   ```

3. Set up environment variables:
   Create a `.env` file in the `backend` directory with:
   ```env
   GROQ_API_KEY=your_groq_api_key
   GEMINI_API_KEY=your_gemini_api_key
   GROQ_MODEL=openai/gpt-oss-20b
   GEMINI_MODEL=models/gemini-2.5-pro
   ```

### Running the Backend

From the `backend` directory:

```bash
python main.py <path_to_media_file>
```

Supported formats: `.mp4`, `.avi`, `.mov`, `.wav`, `.mp3`

Example:

```bash
python main.py ../test_files/pitch_video.mp4
```

## Frontend Setup

### Prerequisites

- Node.js 18+ or Bun 1.3.4+

### Installation

1. Navigate to the frontend directory:

   ```bash
   cd frontend
   ```

2. Install dependencies:

   ```bash
   # Using Bun (recommended)
   bun install

   # Or using npm
   npm install
   ```

### Running the Frontend

From the `frontend` directory:

```bash
# Using Bun
bun run dev

# Or using npm
npm run dev
```

The frontend will be available at `http://localhost:5173` (or the port shown in the terminal).

## Development

### Backend Development

- All backend code is in the `backend/` directory
- The main entry point is `backend/main.py`
- Agents, utilities, and graph definitions are organized in their respective folders
- Logs are written to `backend/athena_task.log`

### Frontend Development

- All frontend code is in the `frontend/` directory
- Built with React, TypeScript, Vite, and shadcn-ui
- Source code is in `frontend/src/`
- Components are in `frontend/src/components/`

## Architecture

### Backend Architecture

- **Multi-Agent System**: Uses LangGraph to orchestrate multiple AI agents
- **Agents**: Audio processing, voice analysis, ASR (Automatic Speech Recognition)
- **Content Analysis**: Master agent analyzes pitch content and structure
- **Shark Evaluators**: Multiple specialized agents evaluate different aspects (visionary, finance, customer, skeptic)
- **Aggregator**: Combines all evaluations into a final verdict

### Frontend Architecture

- **React 19**: Modern React with hooks
- **TypeScript**: Type-safe development
- **Vite**: Fast build tool and dev server
- **shadcn-ui**: Component library
- **Tailwind CSS**: Utility-first styling

## Deployment

### Backend Deployment

- Ensure all environment variables are set
- Install dependencies in production environment
- Run as a service or containerized application

### Frontend Deployment

- Build for production:
  ```bash
  cd frontend
  bun run build
  ```
- Deploy the `dist/` folder to a static hosting service (Vercel, Netlify, etc.)

## Environment Variables

### Backend (.env in backend/)

- `GROQ_API_KEY`: API key for Groq LLM services
- `GEMINI_API_KEY`: API key for Google Gemini
- `GROQ_MODEL`: Model identifier for Groq (default: `openai/gpt-oss-20b`)
- `GEMINI_MODEL`: Model identifier for Gemini (default: `models/gemini-2.5-pro`)
- `GROQ_ASR_MODEL`: ASR model identifier (default: `whisper-large-v3`)
- `GROQ_RATE_LIMIT_PER_MINUTE`: Rate limit for API calls (default: `30`)

## License

[Add your license here]

## Contributing

[Add contributing guidelines here]
