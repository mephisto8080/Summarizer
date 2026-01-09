import argparse
import os
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

from .utils.config_loader import ConfigLoader
from .llm_clients.llm_factory import LLMFactory
from .summarizers.hierarchical_summarizer import HierarchicalSummarizer


def setup_argparse():
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Hierarchical PDF Summarizer - A modular document summarization tool"
    )

    parser.add_argument(
        "pdf_path",
        type=str,
        help="Path to the PDF file to summarize"
    )

    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        choices=["groq", "ollama"],
        help="LLM provider to use (default: from config/env)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name to use (default: from config)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: ./data/output/summary_<timestamp>.md)"
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom config file"
    )

    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Save intermediate results (chunks, meta-sections, etc.)"
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for the provider (overrides env variable)"
    )

    return parser


def get_output_path(custom_path: str = None) -> str:
    """
    Get output file path.

    Args:
        custom_path: Custom output path if provided

    Returns:
        Output file path
    """
    if custom_path:
        return custom_path

    output_dir = Path(__file__).parent.parent / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(output_dir / f"summary_{timestamp}.md")


def save_results(results: dict, output_path: str, save_intermediate: bool = False):
    """
    Save summarization results to file.

    Args:
        results: Results dictionary
        output_path: Output file path
        save_intermediate: Whether to save intermediate results
    """
    output_path = Path(output_path)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save main summary
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Document Summary\n\n")
        f.write(results["global_summary"])
        f.write("\n\n---\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"\nSummary saved to: {output_path}")

    # Save intermediate results if requested
    if save_intermediate:
        base_name = output_path.stem
        intermediate_dir = output_dir / f"{base_name}_intermediate"
        intermediate_dir.mkdir(exist_ok=True)

        # Save meta-summaries
        meta_path = intermediate_dir / "meta_summaries.json"
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(results["meta_summaries"], f, indent=2)

        # Save chunks
        chunks_path = intermediate_dir / "chunks.csv"
        results["chunks"].to_csv(chunks_path, index=False)

        print(f"Intermediate results saved to: {intermediate_dir}")


def main():
    """Main entry point."""
    # Load environment variables
    load_dotenv()

    # Parse arguments
    parser = setup_argparse()
    args = parser.parse_args()

    # Load configuration
    config_loader = ConfigLoader(args.config)

    # Determine provider
    provider = (args.provider or
                config_loader.get_env("DEFAULT_MODEL_PROVIDER") or
                "groq")

    print(f"\n{'=' * 60}")
    print(f"Hierarchical PDF Summarizer")
    print(f"{'=' * 60}")
    print(f"Provider: {provider}")
    print(f"PDF: {args.pdf_path}")
    print(f"{'=' * 60}\n")

    # Check if PDF exists
    if not os.path.exists(args.pdf_path):
        print(f"Error: PDF file not found: {args.pdf_path}")
        return

    # Get provider configuration
    model_config = config_loader.get_model_config(provider)

    # Override with command line arguments
    if args.model:
        model_config["model_name"] = args.model

    # Get API key
    if provider == "groq":
        api_key = (args.api_key or
                   config_loader.get_env("GROQ_API_KEY"))
        if not api_key:
            print("Error: GROQ_API_KEY not found. Set it in .env file or use --api-key")
            return
        model_config["api_key"] = api_key

    # Create LLM client
    try:
        llm_client = LLMFactory.create_client(provider, model_config)
        print(f"LLM client initialized: {provider}")
    except Exception as e:
        print(f"Error creating LLM client: {e}")
        return

    # Get processing configuration
    processing_config = config_loader.get_processing_config()

    # Merge model config for max_tokens
    processing_config.update({
        "max_tokens_meta": model_config.get("max_tokens_meta", 3500),
        "max_tokens_global": model_config.get("max_tokens_global", 1800)
    })

    # Create summarizer
    summarizer = HierarchicalSummarizer(llm_client, processing_config)

    # Process PDF
    try:
        results = summarizer.process_pdf(args.pdf_path, args.save_intermediate)

        # Save results
        output_path = get_output_path(args.output)
        save_results(results, output_path, args.save_intermediate)

        print(f"\n{'=' * 60}")
        print("Summarization completed successfully!")
        print(f"{'=' * 60}\n")

    except Exception as e:
        print(f"\nError during summarization: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
