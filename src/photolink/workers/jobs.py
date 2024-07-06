"""Package worker functions into jobs"""
import photolink.workers.worker as worker
import photolink.utils.enums as enums
import json
import sys
import traceback
from photolink import get_application_path, get_config_file
from photolink.utils.function import read_config
import os
from pathlib import Path

class JobProcessor:
    """Preprocessing codes are the heaviest. Based on the jobs generated from main app, run multiprocessing to expediate. Save results to predefined cache path. Postprocessing does not need multiprocessing."""

    def __init__(self):
        self.application_path = get_application_path()
        config = get_config_file(self.application_path)
        self.config = read_config(config)
        self.cache_dir = self.application_path / ".cache"

        # These should never fail validation. If they do, let it die.
        if not self.cache_dir.exists():
            raise FileNotFoundError(f"Cache directory not found: {self.cache_dir}")

        self.jobs_file = self.cache_dir / Path("job.json")

        if not self.jobs_file.exists():
            raise FileNotFoundError(f"Jobs file not found: {self.jobs_file}")

        with open(self.jobs_file, "r") as f:
            self.jobs = json.load(f)

        self.task = self.jobs["task"]
        self.output_path = Path(self.jobs["output"])
        self.source_list_images = None
        self.reference_list_images = None
        self.num_processes = os.cpu_count()
        print(f"Number of CPU cores: {self.num_processes}", flush=True)
        self.chunksize = int(os.getenv("CHUNKSIZE", 10))
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
        print('Jobs executing. This may take a few minutes.', flush=True)
        self.source_list_images = self.jobs["source"]
        if self.task == enums.Task.SAMPLE_MATCHING.name:
            self.reference_list_images = self.jobs["reference"]
            self.preprocess_sample_matching()
            print("Preprocessing ended")
            self.postprocess_sample_matching()
        elif self.task == enums.Task.CLUSTERING.name:
            self.preprocess_clustering()
            print("Preprocessing ended")
            self.postprocess_clustering()

        else:
            raise NotImplementedError(f"Task not implemented: {self.task}")

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
            worker.run_model_mp(
                self.source_list_images,
                self.num_processes,
                self.chunksize,
                self.source_cache,
                self.fail_path,
                self.top_n_face,
            )

            worker.run_model_mp(
                self.reference_list_images,
                self.num_processes,
                self.chunksize,
                self.reference_cache,
                self.fail_path,
                self.top_n_face,
            )
        except Exception as e:
            print(f"Unexpected preprocessing error during matching: {e}", file=sys.stderr)
            print(traceback.format_exc())
            sys.exit(1)

    def postprocess_sample_matching(self):
        """Postprocess the matching algorithm."""
        try:
            result = worker.match_embeddings(
                source_cache=self.source_cache,
                reference_cache=self.reference_cache,
                source_list_images=self.source_list_images,
                reference_list_images=self.reference_list_images,
                output_path=self.output_path,
            )

            if "error" in result:
                print(f"Postprocessing Error during matching: {result['error']}", file=sys.stderr)
                sys.exit(1)

        except Exception as e:
            # let handle_stderr handle the error
            print(f"Unexpected postprocessing error during matching: {e}", file=sys.stderr)
            print(traceback.format_exc())
            sys.exit(1)

    def preprocess_clustering(self) -> None:
        """Preprocessing for the clustering algorithm. Use MP."""

        if len(self.source_list_images) == 0:
            e = enums.ErrorMessage.SOURCE_FOLDER_EMPTY.value
            print(f"Preprocessing error during clustering: {e}", file=sys.stderr)
            sys.exit(1)

        try:
            worker.run_model_mp(
                self.source_list_images,
                self.num_processes,
                self.chunksize,
                self.source_cache,
                self.fail_path,
                self.top_n_face,
            )
        except Exception as e:
            print(f"Unexpected preprocessing error during clustering: {e}", file=sys.stderr)
            print(traceback.format_exc())
            sys.exit(1)

    def postprocess_clustering(self) -> None:
        """Postprocess the clustering algorithm."""
        clustering_algorithm = enums.ClusteringAlgorithm.HDBSCAN.value
        eps = 0.5
        min_samples = 2

        try:
            result = worker.cluster_embeddings(
                source_cache=self.source_cache,
                source_list_images=self.source_list_images,
                clustering_algorithm=clustering_algorithm,
                eps=eps,
                min_samples=min_samples,
                output_path=self.output_path,
                fail_path=self.fail_path,
            )

            if "error" in result:
                print(f"Postprocessing Error during clustering: {result['error']}", file=sys.stderr)
                sys.exit(1)

        except Exception as e:
            # let handle_stderr handle the error
            print(f"Unexpected postprocessing error during clustering: {e}", file=sys.stderr)
            print(traceback.format_exc())
            sys.exit(1)  # Exit with a non-zero status to indicate an error

