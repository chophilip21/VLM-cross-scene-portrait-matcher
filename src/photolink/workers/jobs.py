"""Middle logic layer b/w worker and functions."""

import photolink.workers.functions as functions
import photolink.utils.enums as enums
import json
import sys
import traceback
from photolink import get_application_path, get_config_file
from photolink.utils.function import read_config
import os
from pathlib import Path
import multiprocessing as mp
from photolink.workers import WorkerSignals


class JobProcessor:
    """Process the jobs for the worker by calling function layer, and emit various signals back to the main application."""

    def __init__(self, stop_event: mp.Event, signals: WorkerSignals):
        self.application_path = get_application_path()
        config = get_config_file(self.application_path)
        self.config = read_config(config)
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
        print(f"Number of CPU cores: {self.num_processes}", flush=True)
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
        print("Jobs executing. This may take a few minutes.", flush=True)
        self.source_list_images = self.jobs["source"]

        # TODO: We could launch thread for monitor here. 

        if self.task == enums.Task.SAMPLE_MATCHING.name:
            self.reference_list_images = self.jobs["reference"]

            # Below function will listen for stop signals
            self.preprocess_sample_matching()

            # check if the stop event is set
            if self.stop_event.is_set():
                print(
                    "Job stopped by user during preprocessing. Will not proceed to postprocessing.",
                    flush=True,
                )
                return enums.StatusMessage.STOPPED.name

            print("Preprocessing ended. Now postprocessing.", flush=True)
            self.postprocess_sample_matching()

            # final stop check
            if self.stop_event.is_set():
                print("Job stopped by user during postprocessing", flush=True)
                return enums.StatusMessage.STOPPED.name

        elif self.task == enums.Task.CLUSTERING.name:

            # Below function will listen for stop signals
            self.signals.progress.emit(25)
            self.preprocess_clustering()
            self.signals.progress.emit(50)

            if self.stop_event.is_set():
                print(
                    "Job stopped by user during preprocessing. Will not proceed to postprocessing.",
                    flush=True,
                )
                return enums.StatusMessage.STOPPED.name

            print("Preprocessing ended. Now postprocessing.", flush=True)
            self.postprocess_clustering()

            # final stop check
            if self.stop_event.is_set():
                print("Job stopped by user during postprocessing", flush=True)
                return enums.StatusMessage.STOPPED.name

        else:
            raise NotImplementedError(f"Task not implemented: {self.task}")

        return enums.StatusMessage.COMPLETE.name

    def preprocess_sample_matching(self) -> None:
        """Preprocessing for the matching algorithm. Use MP."""

        if len(self.source_list_images) == 0:
            e = enums.ErrorMessage.SOURCE_FOLDER_EMPTY.value
            print(f"preprocessing error during matching: {e}", file=sys.stderr)
            sys.exit(1)

        if len(self.reference_list_images) == 0:
            e = enums.ErrorMessage.REFERENCE_FOLDER_EMPTY.value
            print(f"preprocessing error during matching: {e}", file=sys.stderr)
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

            functions.run_model_mp(
                entries=self.reference_list_images,
                num_workers=self.num_processes,
                save_path=self.reference_cache,
                fail_path=self.fail_path,
                keep_top_n=self.top_n_face,
                stop_event=self.stop_event,
            )
        except Exception as e:
            print(
                f"Unexpected preprocessing error during matching: {e}", file=sys.stderr
            )
            print(traceback.format_exc())
            sys.exit(1)

    def postprocess_sample_matching(self):
        """Postprocess the matching algorithm."""
        try:
            result = functions.match_embeddings(
                source_cache=self.source_cache,
                reference_cache=self.reference_cache,
                source_list_images=self.source_list_images,
                reference_list_images=self.reference_list_images,
                output_path=self.output_path,
                stop_event=self.stop_event,
            )

            if "error" in result:
                print(
                    f"Postprocessing Error during matching: {result['error']}",
                    file=sys.stderr,
                )
                sys.exit(1)

        except Exception as e:
            # let handle_stderr handle the error
            print(
                f"Unexpected postprocessing error during matching: {e}", file=sys.stderr
            )
            print(traceback.format_exc())
            sys.exit(1)

    def preprocess_clustering(self) -> None:
        """Preprocessing for the clustering algorithm. Use MP."""

        if len(self.source_list_images) == 0:
            e = enums.ErrorMessage.SOURCE_FOLDER_EMPTY.value
            print(f"Preprocessing error during clustering: {e}", file=sys.stderr)
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
            print(
                f"Unexpected preprocessing error during clustering: {e}",
                file=sys.stderr,
            )
            print(traceback.format_exc())
            sys.exit(1)

    def postprocess_clustering(self) -> None:
        """Postprocess the clustering algorithm."""
        clustering_algorithm = enums.ClusteringAlgorithm.HDBSCAN.value
        eps = 0.5
        min_samples = 2

        try:
            result = functions.cluster_embeddings(
                source_cache=self.source_cache,
                source_list_images=self.source_list_images,
                clustering_algorithm=clustering_algorithm,
                eps=eps,
                min_samples=min_samples,
                output_path=self.output_path,
                fail_path=self.fail_path,
                stop_event=self.stop_event,
            )

        except Exception as e:
            # let handle_stderr handle the error
            print(
                f"Unexpected postprocessing error during clustering: {e}",
                file=sys.stderr,
            )
            print(traceback.format_exc())
            sys.exit(1)  # Exit with a non-zero status to indicate an error
