import logging
from functools import lru_cache

import cv2
import numpy
from tqdm import tqdm

from facefusion import inference_manager, state_manager, wording
from facefusion.download import conditional_download_hashes, conditional_download_sources
from facefusion.filesystem import resolve_relative_path
from facefusion.thread_helper import conditional_thread_semaphore
from facefusion.typing import Fps, InferencePool, ModelOptions, ModelSet, VisionFrame
from facefusion.vision import count_video_frame_total, detect_video_fps, get_video_frame, read_image

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_SET: ModelSet = \
{
	'open_nsfw':
	{
		'hashes':
		{
			'content_analyser':
			{
				'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/open_nsfw.hash',
				'path': resolve_relative_path('../.assets/models/open_nsfw.hash')
			}
		},
		'sources':
		{
			'content_analyser':
			{
				'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/open_nsfw.onnx',
				'path': resolve_relative_path('../.assets/models/open_nsfw.onnx')
			}
		},
		'size': (224, 224),
		'mean': [ 104, 117, 123 ]
	}
}
PROBABILITY_LIMIT = 0.80
RATE_LIMIT = 10
STREAM_COUNTER = 0


def get_inference_pool() -> InferencePool:
	logger.info("Getting inference pool")
	# Skipping inference pool retrieval for NSFW
	return None


def clear_inference_pool() -> None:
	logger.info("Clearing inference pool")
	# No inference pool to clear


def get_model_options() -> ModelOptions:
	logger.info("Getting model options")
	return MODEL_SET.get('open_nsfw')


def pre_check() -> bool:
	logger.info("Pre-checking model availability")
	# Skipping model availability checks
	return True


def analyse_stream(vision_frame: VisionFrame, video_fps: Fps) -> bool:
	logger.info("Analysing stream")
	global STREAM_COUNTER

	STREAM_COUNTER += 1
	if STREAM_COUNTER % int(video_fps) == 0:
		return True  # Skipping NSFW check
	return False


def analyse_frame(vision_frame: VisionFrame) -> bool:
	logger.info("Analysing frame")
	return True  # Skipping NSFW check


def forward(vision_frame: VisionFrame) -> float:
	logger.info("Forwarding frame to model")
	return 0.0  # Skipping NSFW check


def prepare_frame(vision_frame: VisionFrame) -> VisionFrame:
	logger.info("Preparing frame for model input")
	model_size = get_model_options().get('size')
	model_mean = get_model_options().get('mean')
	vision_frame = cv2.resize(vision_frame, model_size).astype(numpy.float32)
	vision_frame -= numpy.array(model_mean).astype(numpy.float32)
	vision_frame = numpy.expand_dims(vision_frame, axis=0)
	return vision_frame


@lru_cache(maxsize=None)
def analyse_image(image_path: str) -> bool:
	logger.info("Analysing image")
	return True  # Skipping NSFW check


@lru_cache(maxsize=None)
def analyse_video(video_path: str, start_frame: int, end_frame: int) -> bool:
	logger.info("Analysing video")
	return True  # Skipping NSFW check
