"""Middle logic layer b/w worker and functions."""

import json
import multiprocessing as mp
import os
import sys
import traceback
from pathlib import Path

from loguru import logger

import photolink.utils.enums as enums
import photolink.workers.functions as functions
from photolink import get_application_path, get_config
from photolink.workers import WorkerSignals
import time


class JobProcessor:
    """Process the jobs for the worker by calling function layer, and emit various signals back to the main application."""

    def __init__(self, stop_event: mp.Event, signals: WorkerSignals):
        self.application_path = get_application_path()
        self.config = get_config()
        self.cache_dir = self.application_path / ".cache"
        self.stop_event = stop_event
        self.jobs_file = self.cache_dir / Path("job.json")
        self.signals = signals

        if not self.cache_dir.exists():
            raise FileNotFoundError(f"Cache directory not found: {self.cache_dir}")

        if not self.jobs_file.exists():
            raise FileNotFoundError(f"Jobs file not found: {self.jobs_file}")

        with open(self.jobs_file, "r") as f:
            self.jobs = json.load(f)

        # other variables
        self.task = self.jobs["task"]
        self.output_path = Path(self.jobs["output"])
        self.source_list_images = None
        self.reference_list_images = None
        self.num_processes = os.cpu_count()
        logger.info(f"Number of CPU cores: {self.num_processes}")
        self.top_n_face = int(os.getenv("TOP_N_FACE", 3))
        self.min_clustering_samples = int(os.getenv("MIN_CLUSTERING_SAMPLES", 2))
        self.source_cache = self.cache_dir / "source"
        self.reference_cache = self.reference_cache = self.cache_dir / "reference"
        self.source_cache.mkdir(parents=True, exist_ok=True)
        self.reference_cache.mkdir(parents=True, exist_ok=True)

        # save the failed processing ones to a separate folder
        self.fail_path = self.output_path / "missed"
        self.fail_path.mkdir(parents=True, exist_ok=True)

    def run(self):
        """Run the job processor."""
        logger.info("Jobs executing. This may take a few minutes.")
        self.source_list_images = self.jobs["source"]

        if self.task == enums.Task.FACE_SEARCH.name:
            self.reference_list_images = self.jobs["reference"]

            # Below function will listen for stop signals
            self.preprocess_sample_matching()

            # check if the stop event is set
            if self.stop_event.is_set():
                return enums.StatusMessage.STOPPED.name

            logger.info("Preprocessing ended. Now postprocessing.")
            self.postprocess_sample_matching()

            # final stop check
            if self.stop_event.is_set():
                return enums.StatusMessage.STOPPED.name

        elif self.task == enums.Task.CLUSTERING.name:

            self.preprocess_clustering()
            if self.stop_event.is_set():
                return enums.StatusMessage.STOPPED.name

            logger.info("Preprocessing ended. Now postprocessing.")
            self.postprocess_clustering()

            # final stop check
            if self.stop_event.is_set():
                return enums.StatusMessage.STOPPED.name

        elif self.task == enums.Task.DP2_MATCH.name:

            # simulate work
            time.sleep(10)

        else:
            raise NotImplementedError(f"Task not implemented: {self.task}")

        return enums.StatusMessage.COMPLETE.name

    def preprocess_sample_matching(self) -> None:
        """Preprocessing for the matching algorithm. Use MP."""

        if len(self.source_list_images) == 0:
            e = enums.ErrorMessage.SOURCE_FOLDER_EMPTY.value
            logger.error(f"preprocessing error during matching: {e}", file=sys.stderr)
            self.signals.error.emit(str(e))

        if len(self.reference_list_images) == 0:
            e = enums.ErrorMessage.REFERENCE_FOLDER_EMPTY.value
            logger.error(f"preprocessing error during matching: {e}")
            self.signals.error.emit(str(e))

        try:
            functions.run_model_mp(
                entries=self.source_list_images,
                num_workers=self.num_processes,
                save_path=self.source_cache,
                fail_path=self.fail_path,
                keep_top_n=self.top_n_face,
                stop_event=self.stop_event,
            )

            functions.run_model_mp(
                entries=self.reference_list_images,
                num_workers=self.num_processes,
                save_path=self.reference_cache,
                fail_path=self.fail_path,
                keep_top_n=self.top_n_face,
                stop_event=self.stop_event,
            )
        except Exception as e:
            logger.error(
                f"Unexpected preprocessing error during matching: {e}", file=sys.stderr
            )
            logger.error(traceback.format_exc())
            self.signals.error.emit(str(e))

    def postprocess_sample_matching(self):
        """Postprocess the matching algorithm."""

        try:
            result = functions.match_embeddings(
                source_cache=self.source_cache,
                reference_cache=self.reference_cache,
                output_path=self.output_path,
                stop_event=self.stop_event,
            )

            if "error" in result:
                logger.error(
                    f"Postprocessing Error during matching: {result['error']}",
                )
                self.signals.error.emit(str((result["error"])))

        except Exception as e:
            # let handle_stderr handle the error
            logger.error(f"Unexpected postprocessing error during matching: {e}")
            logger.error(traceback.format_exc())
            self.signals.error.emit(str(e))

    def preprocess_clustering(self) -> None:
        """Preprocessing for the clustering algorithm. Use MP."""

        if len(self.source_list_images) == 0:
            e = enums.ErrorMessage.SOURCE_FOLDER_EMPTY.value
            logger.error(f"Preprocessing error during clustering: {e}", file=sys.stderr)
            sys.exit(1)

        try:
            functions.run_model_mp(
                entries=self.source_list_images,
                num_workers=self.num_processes,
                save_path=self.source_cache,
                fail_path=self.fail_path,
                keep_top_n=self.top_n_face,
                stop_event=self.stop_event,
            )
        except Exception as e:
            logger.error(
                f"Unexpected preprocessing error during clustering: {e}",
            )
            logger.error(traceback.format_exc())
            self.signals.error.emit(str(e))

    def postprocess_clustering(self) -> None:
        """Postprocess the clustering algorithm."""
        clustering_algorithm = enums.ClusteringAlgorithm.HDBSCAN.value
        eps = 0.5
        min_samples = 2

        try:
            result = functions.cluster_embeddings(
                source_cache=self.source_cache,
                clustering_algorithm=clustering_algorithm,
                eps=eps,
                min_samples=min_samples,
                output_path=self.output_path,
                fail_path=self.fail_path,
                stop_event=self.stop_event,
            )

        except Exception as e:
            # let handle_stderr handle the error
            logger.error(f"Unexpected postprocessing error during clustering: {e}")
            logger.error(traceback.format_exc())
            self.signals.error.emit(str(e))
