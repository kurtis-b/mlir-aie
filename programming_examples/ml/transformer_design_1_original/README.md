# Transformer Layer on Ryzen NPU

This repository contains an implementation of a transformer layer optimized for execution on a Ryzen Neural Processing Unit (NPU). The implementation is designed to leverage the computational capabilities of the NPU for efficient deep learning workloads.

## Features

- **Optimized Performance**: Utilizes Ryzen NPU for accelerated matrix operations and attention mechanisms.
- **Modular Design**: Easy to integrate into larger transformer architectures.
- **Scalability**: Supports varying model sizes and configurations.
- **Customizable**: Parameters such as hidden size, number of heads, and sequence length can be adjusted.

## Requirements

- **Hardware**: AMD Ryzen processor with NPU support.
- **Software**:
  - Python 3.8+
  - MLIR-AIE framework
  - PyTorch or TensorFlow (optional for testing)
  - NumPy

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-repo/ryzen-transformer-layer.git
   cd ryzen-transformer-layer
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Build the MLIR-AIE components:
   ```bash
   ./build_mlir_aie.sh
   ```

## Usage

1. Configure the transformer layer parameters in `config.json`.
2. Run the implementation:
   ```bash
   python run_transformer_layer.py
   ```

## Benchmarking

To benchmark the performance of the transformer layer on the Ryzen NPU, use the provided benchmarking script:

```bash
python benchmark.py
```

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any bugs or feature requests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

Special thanks to the MLIR-AIE community and AMD for their support and tools.
