#!/bin/bash
set -e

MODEL_DIR="${DIPRO_MODEL_DIR:-/app/generation/model}"

download_model() {
    if [ ! -f "$MODEL_DIR/DiPro" ] || [ ! -f "$MODEL_DIR/DiPro_ema" ]; then
        echo "=========================================="
        echo "  Downloading pretrained models..."
        echo "=========================================="
        mkdir -p "$MODEL_DIR"
        huggingface-cli download lixinxin/DiPro --local-dir "$MODEL_DIR"
        echo "Model download complete."
    fi
}

show_help() {
    echo ""
    echo "DiPro: Diffusion-based DNA Promoter Sequence Generator"
    echo "======================================================="
    echo ""
    echo "Usage: docker run [OPTIONS] dipro COMMAND [ARGS...]"
    echo ""
    echo "Commands:"
    echo "  generate    Generate novel DNA promoter sequences"
    echo "  inpaint     Complete partial DNA sequences"
    echo "  train       Train or fine-tune the model"
    echo "  --help      Show this help message"
    echo ""
    echo "Examples:"
    echo ""
    echo "  # Generate 100 sequences"
    echo "  docker run --gpus all -v dipro_models:/app/generation/model \\"
    echo "      -v \$(pwd)/output:/app/output dipro \\"
    echo "      generate --num_examples 100 --output_csv_path output/generated.csv"
    echo ""
    echo "  # Inpaint partial sequences"
    echo "  docker run --gpus all -v dipro_models:/app/generation/model \\"
    echo "      -v \$(pwd)/data:/app/data dipro \\"
    echo "      inpaint --input_csv data/partial.csv --output_csv data/completed.csv"
    echo ""
    echo "  # Train on custom dataset"
    echo "  docker run --gpus all -v dipro_models:/app/generation/model \\"
    echo "      -v \$(pwd)/data:/app/data dipro \\"
    echo "      train --data_path data/my_promoters.csv --epochs 500"
    echo ""
}

case "${1:-}" in
    generate)
        download_model
        shift
        exec python DiPro_generate.py "$@"
        ;;
    inpaint)
        download_model
        shift
        exec python DiPro_inpaint.py "$@"
        ;;
    train)
        download_model
        shift
        exec python DiPro_train.py "$@"
        ;;
    template)
        shift
        exec python create_inpaint_template.py "$@"
        ;;
    --help|"")
        show_help
        ;;
    *)
        exec "$@"
        ;;
esac
