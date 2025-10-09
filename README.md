# Griptape Nodes: Hyper3D Rodin Library

A Griptape Nodes library for generating 3D models using the Hyper3D Rodin API. This library provides nodes for both Text-to-3D and Image-to-3D generation with advanced customization options.

## Features

- **Text-to-3D Generation**: Create 3D models from text descriptions
- **Image-to-3D Generation**: Convert images into 3D models  
- **Multi-Image Support**: Use up to 5 images for enhanced generation
- **Format Support**: GLB and USDZ output formats
- **Quality Control**: Multiple quality levels and mesh optimization
- **Advanced Options**: Bounding box constraints, pose control, and material settings

## Installation

1. Clone this repository into your Griptape Nodes workspace directory:
   ```bash
   cd $(gtn config | grep workspace_directory | cut -d'"' -f4)
   git clone https://github.com/griptape-ai/griptape-nodes-library-rodin.git
   ```

2. Install dependencies:
   ```bash
   cd griptape-nodes-library-rodin
   uv sync
   ```

## Configuration

### Environment Variables

This library requires a Hyper3D Rodin API key. Set the following environment variable:

```bash
export RODIN_API_KEY="your_rodin_api_key_here"
```

You can obtain a Hyper3D Rodin API key from [Hyper3D](https://hyper3d.ai/).

### Griptape Nodes Setup

The library will automatically register the `RODIN_API_KEY` environment variable when loaded. Ensure your environment variable is set before starting Griptape Nodes.

## Nodes

### Rodin 3D Generator

Generates 3D models from text prompts or images using the Rodin AI API.

**Inputs:**
- **Images** (optional): Up to 5 images for Image-to-3D generation
- **Prompt** (optional): Text description for Text-to-3D generation
- **Tier**: Generation tier (Regular/Priority)
- **File Format**: Output format (GLB/USDZ)
- **Material**: Material type (PBR/Blinn-Phong)
- **Quality**: Generation quality (low/medium/high/extra-low)
- **Mesh Mode**: Mesh type (Quad/Triangle)
- **Mesh Simplify**: Enable mesh simplification
- **Condition Mode**: Multi-image condition mode
- **T-A Pose**: Enable pose guidance
- **Seed**: Random seed for reproducible results
- **Bounding Box**: Size constraints (Y,Z,X format)

**Outputs:**
- **3D Model**: Generated GLTF model artifact
- **Task UUID**: Rodin task identifier
- **All Files**: List of all generated file URLs

## Usage Examples

### Text-to-3D Generation

1. Add the "Rodin 3D Generator" node to your workflow
2. Set the **Prompt** parameter (e.g., "A cute wooden toy robot")
3. Configure quality and format settings as needed
4. Run the workflow
5. The generated 3D model will be available as a GLTF artifact

### Image-to-3D Generation

1. Add the "Rodin 3D Generator" node to your workflow
2. Connect one or more images to the **Images** input
3. Optionally add a descriptive prompt
4. Configure generation settings
5. Run the workflow
6. The generated 3D model will be available as a GLTF artifact

### Multi-Image Generation

1. Connect multiple images (up to 5) to the **Images** input
2. Set the **Condition Mode** for how multiple images should be processed
3. Configure other parameters as needed
4. The node will automatically handle multi-image generation

## API Reference

This library uses the Hyper3D Rodin AI API v2. For more information about the underlying API, visit the [Hyper3D API documentation](https://developer.hyper3d.ai/).

## Support

For issues related to this library, please open an issue on the GitHub repository.
For Hyper3D API-related questions, please refer to the official Hyper3D documentation.

## License

This library is licensed under the Apache License 2.0. See the LICENSE file for details.
