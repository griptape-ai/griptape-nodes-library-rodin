import os
import time
import tempfile
import json
import requests
from typing import Any
from griptape.artifacts import ImageUrlArtifact, UrlArtifact, BaseArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode, ParameterList
from griptape_nodes.exe_types.node_types import DataNode, ControlNode, AsyncResult
from griptape_nodes.traits.options import Options
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes, logger

SERVICE = "Rodin"
API_KEY_ENV_VAR = "RODIN_API_KEY"
BASE_URL = "https://api.hyper3d.com/api/v2"

class GLTFUrlArtifact(UrlArtifact):
    """A GLTF file URL artifact with metadata support."""

    def __init__(self, value: str, name: str | None = None, metadata: dict | None = None) -> None:
        # Store URL directly as value like standard UrlArtifact
        super().__init__(value=value, name=name or self.__class__.__name__)
        # Store metadata separately for LoadGLTF compatibility
        self._metadata = metadata or {}
        
    def get(self, key: str, default=None):
        """Make artifact behave like a dictionary for LoadGLTF compatibility."""
        if key == "metadata":
            return self._metadata
        elif key == "url" or key == "value":
            return self.value
        elif key == "type":
            return "GLTFUrlArtifact"
        return default

class Rodin3DGenerator(ControlNode):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # Generation mode and inputs
        self.add_parameter(
            ParameterList(
                name="images",
                input_types=["ImageUrlArtifact", "list[ImageUrlArtifact]"],
                default_value=[],
                tooltip="Connect individual images or a list of images (up to 5) for Image-to-3D generation",
                allowed_modes={ParameterMode.INPUT},
                ui_options={"display_name": "Images", "clickable_file_browser": True}
            )
        )

        self.add_parameter(
            Parameter(
                name="prompt",
                input_types=["str"],
                type="str",
                tooltip="Text prompt for Text-to-3D generation, or optional description for Image-to-3D",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"multiline": True, "placeholder_text": "Describe the 3D model you want to generate..."},
                default_value=""
            )
        )

        self.add_parameter(
            Parameter(
                name="condition_mode",
                input_types=["str"],
                type="str",
                tooltip="Mode for multi-image generation: 'concat' for multi-view of same object, 'fuse' for combining multiple objects",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                default_value="concat",
                traits={Options(choices=["concat", "fuse"])},
                ui_options={"display_name": "Multi-Image Mode"}
            )
        )

        # Quality and output settings
        self.add_parameter(
            Parameter(
                name="tier",
                input_types=["str"],
                type="str",
                tooltip="Generation tier: Sketch (fast), Regular (balanced), Detail (enhanced), Smooth (sharper)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                default_value="Regular",
                traits={Options(choices=["Sketch", "Regular", "Detail", "Smooth"])},
                ui_options={"display_name": "Tier"}
            )
        )

        self.add_parameter(
            Parameter(
                name="geometry_file_format",
                input_types=["str"],
                type="str",
                tooltip="Output 3D file format",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                default_value="glb",
                traits={Options(choices=["glb", "usdz", "fbx", "obj", "stl"])},
                ui_options={"display_name": "File Format"}
            )
        )

        self.add_parameter(
            Parameter(
                name="material",
                input_types=["str"],
                type="str",
                tooltip="Material type: PBR (physically based), Shaded (baked lighting), All (both)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                default_value="PBR",
                traits={Options(choices=["PBR", "Shaded", "All"])},
                ui_options={"display_name": "Material"}
            )
        )

        self.add_parameter(
            Parameter(
                name="quality",
                input_types=["str"],
                type="str",
                tooltip="Mesh quality: high (50k faces), medium (18k), low (8k), extra-low (4k)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                default_value="medium",
                traits={Options(choices=["high", "medium", "low", "extra-low"])},
                ui_options={"display_name": "Quality"}
            )
        )

        # Advanced parameters
        self.add_parameter(
            Parameter(
                name="seed",
                input_types=["int"],
                type="int",
                tooltip="Random seed for reproducible results (0-65535)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"display_name": "Seed", "placeholder_text": "Enter seed number"},
                default_value=None
            )
        )

        self.add_parameter(
            Parameter(
                name="mesh_mode",
                input_types=["str"],
                type="str",
                tooltip="Face type: Quad (quadrilateral), Raw (triangular)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                default_value="Quad",
                traits={Options(choices=["Quad", "Raw"])},
                ui_options={"display_name": "Mesh Mode"}
            )
        )

        self.add_parameter(
            Parameter(
                name="mesh_simplify",
                input_types=["bool"],
                type="bool",
                tooltip="Simplify mesh to reduce polygon count while preserving quality",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                default_value=False,
                ui_options={"display_name": "Mesh Simplify"}
            )
        )

        self.add_parameter(
            Parameter(
                name="TAPose",
                input_types=["bool"],
                type="bool",
                tooltip="Force T/A pose for human-like models",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                default_value=False,
                ui_options={"display_name": "T-A Pose"}
            )
        )

        self.add_parameter(
            Parameter(
                name="bbox_condition",
                input_types=["str"],
                type="str",
                tooltip="Bounding box dimensions as comma-separated values: Width(Y), Height(Z), Length(X). Example: 1.0,2.0,3.0",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"display_name": "Bounding Box (Y,Z,X)"},
                default_value=""
            )
        )

        # Output parameters
        self.add_parameter(
            Parameter(
                name="gltf_model",
                output_type="GLTFUrlArtifact",
                type="GLTFUrlArtifact",
                tooltip="Generated 3D model in the requested format",
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={"display_name": "3D Model", "is_full_width": True}
            )
        )

        self.add_parameter(
            Parameter(
                name="all_files",
                output_type="list",
                type="list",
                tooltip="URLs of all generated files",
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={"display_name": "All Files"}
            )
        )

        self.add_parameter(
            Parameter(
                name="task_uuid",
                output_type="str",
                type="str",
                tooltip="Task UUID for reference",
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={"display_name": "Task ID"}
            )
        )

    def validate_node(self) -> list[Exception] | None:
        """Validates that the Rodin API key is configured.
        Returns:
            list[Exception] | None: List of exceptions if validation fails, None if validation passes.
        """
        api_key = self.get_config_value(service=SERVICE, value=API_KEY_ENV_VAR)

        errors = []
        if not api_key:
            errors.append(
                ValueError(f"Rodin API key not found. Please set the {API_KEY_ENV_VAR} environment variable.")
            )

        return errors if errors else None

    def process(self) -> AsyncResult:
        def generate_3d_model() -> GLTFUrlArtifact:
            try:
                logger.debug("🚀 generate_3d_model function ENTERED - starting execution")
                logger.debug("🔧 Initializing outputs and parameters...")
                
                # Initialize parameter outputs
                logger.debug("📝 Setting parameter_output_values...")
                self.parameter_output_values["gltf_model"] = None
                self.parameter_output_values["all_files"] = []
                self.parameter_output_values["task_uuid"] = ""
                logger.debug("✅ Initialization complete")

                # Get API key from environment
                logger.debug("🔑 Getting API key from config...")
                api_key = self.get_config_value(service=SERVICE, value=API_KEY_ENV_VAR)
                logger.debug(f"🔑 API key retrieved: {bool(api_key)} (length: {len(api_key) if api_key else 0})")
                if not api_key or not api_key.strip():
                    logger.debug("❌ API key validation failed")
                    raise ValueError("Rodin API key is required. Please set the RODIN_API_KEY environment variable.")
                logger.debug("✅ API key validated")

                # Get inputs
                logger.debug("📷 Getting images parameter...")
                images_input = self.get_parameter_list_value("images")
                logger.debug(f"📷 Images retrieved: {type(images_input)} with {len(images_input) if images_input else 0} items")
                
                logger.debug("📝 Getting prompt parameter...")
                prompt = self.get_parameter_value("prompt")
                logger.debug(f"📝 Prompt retrieved: {bool(prompt)} (length: {len(prompt) if prompt else 0})")

                # Validate generation mode
                logger.debug("🔍 Validating generation mode...")
                has_images = images_input and len(images_input) > 0
                has_prompt = prompt and prompt.strip()
                logger.debug(f"🔍 Validation: has_images={has_images}, has_prompt={has_prompt}")

                if not has_images and not has_prompt:
                    logger.debug("❌ Validation failed: no images or prompt")
                    raise ValueError("Either images or prompt must be provided")
                logger.debug("✅ Generation mode validated")

                # Prepare request
                logger.debug("📤 Publishing 'Preparing request' status...")
                logger.debug("✅ Status published")
                
                # Submit task
                logger.debug("🚀 About to call _submit_task...")
                task_response = self._submit_task(api_key, images_input, prompt)
                logger.debug(f"✅ _submit_task returned: {type(task_response)}")
                logger.debug(f"📋 Response keys: {list(task_response.keys()) if isinstance(task_response, dict) else 'not dict'}")
                
                logger.debug("🔍 Extracting task_uuid...")
                task_uuid = task_response['uuid']
                logger.debug(f"✅ Task UUID extracted: {task_uuid[:8]}...")
                
                logger.debug("🔍 Extracting subscription_key from jobs...")
                logger.debug(f"📋 Jobs content: {task_response.get('jobs', 'jobs key missing')}")
                subscription_key = task_response['jobs']['subscription_key']
                logger.debug(f"✅ subscription_key extracted: {subscription_key[:8] if subscription_key else 'None'}...")

                # Set task UUID output
                logger.debug("📝 Setting task_uuid in parameter_output_values...")
                self.parameter_output_values["task_uuid"] = task_uuid
                logger.debug("✅ Task submission status published")

                # Poll for completion with real-time updates
                logger.debug("📊 Publishing polling start status...")
                logger.debug("🔄 Published polling status - entering polling loop...")
                
                # Poll status endpoint until completion
                status_url = f"{BASE_URL}/status"
                status_headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                
                max_retries = 120  # 120 * 5 seconds = 10 minutes timeout
                retry_count = 0
                logger.debug(f"🔄 Starting polling loop with max_retries={max_retries}")
                
                while retry_count < max_retries:
                    logger.debug(f"⏱️ Sleeping 5 seconds before polling attempt {retry_count + 1}...")
                    time.sleep(5)  # This is now inside the async function
                    retry_count += 1
                    logger.debug(f"🔄 Starting polling attempt {retry_count}/{max_retries}")
                    
                    try:
                        logger.debug("📡 Making status POST request...")
                        status_data = {"subscription_key": subscription_key}
                        status_response = requests.post(status_url, headers=status_headers, json=status_data, timeout=30)
                        logger.debug(f"📡 Status API responded with {status_response.status_code}")
                        status_response.raise_for_status()
                        
                        logger.debug("📋 Parsing status response...")
                        status_result = status_response.json()
                        jobs = status_result.get('jobs', [])
                        logger.debug(f"📋 Found {len(jobs)} jobs in status response")
                        
                        if not jobs:
                            logger.debug("⏳ No jobs found, continuing polling...")
                            logger.debug(f"⏳ Waiting for job status... (attempt {retry_count}/{max_retries})")
                            continue
                        
                        # Check if all jobs are done
                        logger.debug("🔍 Checking job statuses...")
                        job_statuses = [job.get('status', 'unknown') for job in jobs]
                        logger.debug(f"🔍 Job statuses: {job_statuses}")
                        all_done = all(job['status'] in ['Done', 'Failed'] for job in jobs)
                        logger.debug(f"🔍 All jobs done: {all_done}")
                        
                        # Update status with job details
                        statuses = [f"Job {job['uuid'][:8]}: {job['status']}" for job in jobs]
                        status_text = f"🔄 {' | '.join(statuses)} (attempt {retry_count}/{max_retries})"
                        logger.debug(f"📢 Publishing status: {status_text}")
                        
                        if all_done:
                            # Check for failures
                            failed_jobs = [job for job in jobs if job['status'] == 'Failed']
                            if failed_jobs:
                                logger.debug(f"❌ Found failed jobs: {failed_jobs}")
                                raise Exception(f"Generation failed: {failed_jobs}")
                            logger.debug("✅ All jobs completed successfully")
                            logger.debug("✅ Generation completed! All jobs done - preparing to download...")
                            logger.debug(f"🎯 Exiting polling loop after {retry_count} attempts...")
                            logger.debug(f"🎯 Breaking from polling loop after {retry_count} attempts")
                            break
                        else:
                            logger.debug(f"⏳ Jobs not done yet, continuing polling...")
                            
                    except requests.exceptions.RequestException as e:
                        logger.debug(f"⚠️ Status check failed (attempt {retry_count}/{max_retries}): {str(e)}")
                        if retry_count >= max_retries:
                            raise Exception(f"Failed to check status after {max_retries} attempts: {e}")
                        continue
                
                if retry_count >= max_retries:
                    raise Exception(f"Generation timed out after {max_retries * 5 / 60:.1f} minutes")

                # Download results
                logger.debug("📥 Starting download URL fetch process...")
                logger.debug(f"🔑 Using API key: {api_key[:10]}...{api_key[-4:]}")
                logger.debug(f"🆔 Task UUID: {task_uuid}")
                
                try:
                    logger.debug("🌐 About to call _download_results method")
                    logger.debug("🌐 Calling _download_results method...")
                    download_response = self._download_results(api_key, task_uuid)
                    logger.debug(f"✅ _download_results returned: {type(download_response)} with keys: {list(download_response.keys()) if isinstance(download_response, dict) else 'not dict'}")
                    logger.debug(f"✅ Download URLs received! Response keys: {list(download_response.keys()) if isinstance(download_response, dict) else type(download_response)}")
                except Exception as e:
                    logger.debug(f"❌ _download_results failed: {str(e)}")
                    raise

                # Process downloads
                logger.debug("🔍 Processing download files - entering _process_downloads...")
                try:
                    model_artifact = self._process_downloads(download_response)
                    logger.debug("✅ _process_downloads completed successfully!")
                except Exception as e:
                    logger.debug(f"❌ _process_downloads failed: {str(e)}")
                    raise
                
                logger.debug("✅ generate_3d_model function completed successfully")
                return model_artifact

            except Exception as e:
                logger.debug(f"💥 generate_3d_model function failed: {str(e)}")
                raise Exception(f"Rodin 3D generation failed: {str(e)}")

        logger.debug("🚀 About to yield generate_3d_model function")
        yield generate_3d_model

    def _submit_task(self, api_key: str, images_input: list[ImageUrlArtifact], prompt: str) -> dict:
        """Submit the generation task to Rodin API."""
        url = f"{BASE_URL}/rodin"
        headers = {'Authorization': f'Bearer {api_key}'}
        
        # Prepare form data
        data = {}
        files = []

        # Handle images (images_input is now always a list from ParameterList)
        if images_input:
            if len(images_input) > 5:
                raise ValueError("Maximum 5 images allowed")

            # Download images and prepare for multipart upload
            for i, image_artifact in enumerate(images_input):
                if not isinstance(image_artifact, ImageUrlArtifact):
                    raise ValueError(f"Image {i+1} must be an ImageUrlArtifact")
                
                # Download image content
                image_response = requests.get(image_artifact.value, timeout=30)
                image_response.raise_for_status()
                
                # Add to files for multipart upload
                files.append(('images', (f'image_{i+1}.jpg', image_response.content, 'image/jpeg')))

            # Add condition_mode for multi-image
            if len(images_input) > 1:
                data['condition_mode'] = self.get_parameter_value("condition_mode")

        # Add other parameters
        if prompt:
            data['prompt'] = prompt.strip()

        # Required parameters  
        data['tier'] = self.get_parameter_value("tier")
        data['geometry_file_format'] = self.get_parameter_value("geometry_file_format")
        data['material'] = self.get_parameter_value("material")
        data['quality'] = self.get_parameter_value("quality")

        # Optional parameters
        seed = self.get_parameter_value("seed")
        if seed is not None:
            data['seed'] = seed

        data['mesh_mode'] = self.get_parameter_value("mesh_mode")
        
        mesh_simplify = self.get_parameter_value("mesh_simplify")
        if mesh_simplify is not None:
            data['mesh_simplify'] = 'true' if mesh_simplify else 'false'

        tapose = self.get_parameter_value("TAPose")
        if tapose is not None:
            data['TAPose'] = 'true' if tapose else 'false'

        bbox_condition = self.get_parameter_value("bbox_condition")
        if bbox_condition:
            try:
                width, height, length = map(float, bbox_condition.split(','))
                data['bbox_condition'] = json.dumps([width, height, length])
            except ValueError:
                raise ValueError("Bounding box dimensions must be comma-separated numbers (e.g., 1.0,2.0,3.0)")

        # Debug logging before request
        logger.debug(f"🌐 Request URL: {url}")
        logger.debug(f"📋 Request data: {data}")
        logger.debug(f"📎 Files count: {len(files)}")
        logger.debug(f"📁 Files details: {[(name, filename, content_type) for name, (filename, content, content_type) in files]}")
        logger.debug(f"🔑 Headers: {headers}")
        
        # Submit request
        response = requests.post(url, files=files, data=data, headers=headers)
        
        # Debug response
        logger.debug(f"📊 Response status: {response.status_code}")
        logger.debug(f"📄 Response headers: {dict(response.headers)}")
        if response.status_code != 200:
            logger.debug(f"❌ Response content: {response.text}")
        
        response.raise_for_status()
        
        return response.json()

    def _download_results(self, api_key: str, task_uuid: str) -> dict:
        """Download the generated results."""
        logger.debug("🚀 _download_results method ENTRY")
        
        url = f"{BASE_URL}/download"
        headers = {'Authorization': f'Bearer {api_key}'}
        
        logger.debug(f"📡 About to POST to {url}")
        logger.debug(f"📡 Requesting download URLs for task {task_uuid[:8]}...")
        logger.debug(f"🌐 Download URL: {url}")
        data = {"task_uuid": task_uuid}
        
        try:
            logger.debug("📡 Making download API request...")
            response = requests.post(url, headers=headers, json=data, timeout=60)  # Increased timeout
            logger.debug(f"📡 Download API responded with status {response.status_code}")
            logger.debug("📡 Download API responded, processing response...")
            response.raise_for_status()
            
            download_response = response.json()
            logger.debug(f"📋 Download response parsed successfully")
            file_count = len(download_response.get('list', []))
            logger.debug(f"📋 Successfully parsed {file_count} files")
            logger.debug(f"📋 Received {file_count} files for download")
            
        except requests.exceptions.Timeout:
            logger.debug("⏰ Download API TIMEOUT")
            raise Exception("Download API timeout after 60 seconds")
        except requests.exceptions.RequestException as e:
            logger.debug(f"❌ Download API REQUEST ERROR: {str(e)}")
            raise Exception(f"Download API failed: {str(e)}")
        
        logger.debug("✅ _download_results method EXIT")
        return download_response

    def _process_downloads(self, download_response: dict) -> GLTFUrlArtifact:
        """Process and save downloaded files."""
        logger.debug("🚀 _process_downloads method ENTRY")
        
        download_items = download_response.get('list', [])
        logger.debug(f"📋 Found {len(download_items)} download items")
        logger.debug(f"📋 Found {len(download_items)} download items")
        
        if not download_items:
            raise Exception("No files available for download")

        logger.debug(f"🔍 Analyzing {len(download_items)} generated files...")
        
        all_file_urls = []
        file_names = []
        
        for item in download_items:
            all_file_urls.append(item['url'])
            file_names.append(item['name'])

        # Process and download files
        logger.debug(f"📝 Processing {len(download_items)} files...")
        primary_model_url = None
        primary_model_name = None
        
        # Find the preferred format file
        requested_format = self.get_parameter_value("geometry_file_format")
        for item in download_items:
            file_name = item['name']
            if file_name.lower().endswith(f'.{requested_format}'):
                primary_model_url = item['url']
                primary_model_name = file_name
                logger.debug(f"🎯 Found primary model: {file_name}")
                break

        # If no exact format match, use the first model file
        if not primary_model_url:
            logger.debug(f"🔍 No {requested_format} found, searching for any 3D model...")
            model_extensions = ['.glb', '.usdz', '.fbx', '.obj', '.stl']
            for item in download_items:
                for ext in model_extensions:
                    if item['name'].lower().endswith(ext):
                        primary_model_url = item['url']
                        primary_model_name = item['name']
                        logger.debug(f"🎯 Using fallback model: {item['name']}")
                        break
                if primary_model_url:
                    break

        if not primary_model_url:
            raise Exception("No 3D model file found in generated results")

        # Download the primary model file
        logger.debug(f"⬇️ Starting download of {primary_model_name}")
        logger.debug(f"⬇️ Starting download of {primary_model_name}...")
        
        try:
            # Start download with progress
            logger.debug(f"🌐 Making GET request to {primary_model_url}")
            logger.debug(f"🌐 Requesting {primary_model_name} from server...")
            model_response = requests.get(primary_model_url, timeout=120, stream=True)  # Stream for large files
            logger.debug(f"📡 File download responded with status {model_response.status_code}")
            model_response.raise_for_status()
            
            # Check file size
            content_length = model_response.headers.get('content-length')
            if content_length:
                total_size = int(content_length)
                logger.debug(f"📊 File size: {total_size / (1024 * 1024):.1f}MB - downloading...")
            else:
                logger.debug(f"📊 Downloading {primary_model_name} (size unknown)...")
            
            # Download in chunks
            model_bytes = b''
            downloaded_size = 0
            
            for chunk in model_response.iter_content(chunk_size=8192):
                if chunk:
                    model_bytes += chunk
                    downloaded_size += len(chunk)
                    
                    # Progress update every 1MB
                    if downloaded_size % (1024 * 1024) == 0:
                        mb_downloaded = downloaded_size / (1024 * 1024)
                        if content_length:
                            progress = (downloaded_size / total_size) * 100
                            logger.debug(f"⬇️ Downloaded {mb_downloaded:.1f}MB ({progress:.1f}%)")
                        else:
                            logger.debug(f"⬇️ Downloaded {mb_downloaded:.1f}MB...")
            
            file_size_mb = len(model_bytes) / (1024 * 1024)
            logger.debug(f"✅ Download complete: {primary_model_name} ({file_size_mb:.1f}MB)")
            
        except requests.exceptions.Timeout:
            raise Exception(f"Download timeout for {primary_model_name} after 120 seconds")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to download {primary_model_name}: {str(e)}")

        # Save to static files
        logger.debug(f"💾 Preparing to save {primary_model_name} to local storage")
        logger.debug(f"💾 Preparing to save {primary_model_name} to local storage...")
        
        try:
            # Generate unique filename
            timestamp = int(time.time())
            file_extension = primary_model_name.split('.')[-1]
            static_filename = f"rodin_3d_{timestamp}.{file_extension}"
            logger.debug(f"📝 Generated filename: {static_filename}")
            
            logger.debug("🔧 Creating static file manager...")
            logger.debug("🔧 Creating StaticFilesManager instance...")
            
            # Use GriptapeNodes static file manager
            static_files_manager = GriptapeNodes.StaticFilesManager()
            logger.debug("✅ StaticFilesManager created successfully")
            
            logger.debug(f"💾 Saving {file_size_mb:.1f}MB to static storage as {static_filename}...")
            logger.debug(f"💾 About to call save_static_file with {file_size_mb:.1f}MB")
            static_url = static_files_manager.save_static_file(model_bytes, static_filename)
            logger.debug(f"✅ save_static_file completed: {static_url}")
            
            logger.debug(f"🗂️ Successfully saved as static file: {static_filename}")
            logger.debug(f"🔗 Static URL: {static_url}")
            
        except Exception as e:
            logger.debug(f"❌ Failed to save static file: {str(e)}")
            raise Exception(f"Failed to save model file: {str(e)}")

        # Show file summary
        file_summary = ", ".join(file_names)
        logger.debug(f"📁 Generated files: {file_summary}")

        # Create the GLTF artifact with static URL and metadata
        metadata = {
            "filename": static_filename,
            "original_name": primary_model_name,
            "file_size_mb": file_size_mb,
            "format": requested_format,
            "generated_files": file_names
        }
        model_artifact = GLTFUrlArtifact(value=static_url, name=static_filename, metadata=metadata)

        # Set outputs
        self.publish_update_to_parameter("gltf_model", model_artifact)
        self.publish_update_to_parameter("all_files", all_file_urls)
        logger.debug(f"✅ Complete! Ready to load 3D model ({len(download_items)} files generated)")
        
        logger.debug("✅ _process_downloads method EXIT - returning artifact")
        return model_artifact 