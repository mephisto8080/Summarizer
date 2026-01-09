# Quick Start Guide

Get started with the Hierarchical PDF Summarizer in 5 minutes!

## Step 1: Setup

```bash
cd summarizer-project
chmod +x setup.sh
./setup.sh
```

## Step 2: Configure API Key

Edit the `.env` file:

```bash
nano .env
```

Add your Groq API key:
```
GROQ_API_KEY=your_groq_api_key_here
```

**Don't have a Groq API key?** Get one free at [console.groq.com](https://console.groq.com)

## Step 3: Run Your First Summary

```bash
# Activate environment
source venv/bin/activate

# Summarize a PDF
python -m src.main /path/to/your/document.pdf
```

## Step 4: Check Output

Your summary will be saved in `data/output/summary_TIMESTAMP.md`

## Next Steps

- Try saving intermediate results: `--save-intermediate`
- Use local models with Ollama: `--provider ollama`
- Read the full [README.md](README.md) for advanced usage

## Common Commands

```bash
# Basic usage
python -m src.main document.pdf

# With intermediate results
python -m src.main document.pdf --save-intermediate

# Using Ollama (local)
python -m src.main document.pdf --provider ollama --model llama3

# Custom output location
python -m src.main document.pdf --output my_summary.md

# Get help
python -m src.main --help
```

## Troubleshooting

**Import errors?** Make sure to use `python -m src.main` (not `python src/main.py`)

**API key not found?** Check that `.env` file exists and contains `GROQ_API_KEY`

**Module not found?** Activate virtual environment: `source venv/bin/activate`
