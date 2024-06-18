"""Run multiprocessing code based on the jobs generated from app.py"""
import photolink.workers.worker as worker
import photolink.utils.enums as enums
import os
import json

class JobProcessor():
    """Run multiprocessing code based on the jobs generated from app.py"""

    def __init__(self):
        self.cache_dir = os.path.join(os.path.dirname(__file__), ".cache")

        if not os.path.exists(self.cache_dir):
            raise FileNotFoundError(f"Cache directory not found: {self.cache_dir}")

        self.jobs_file = os.path.join(self.cache_dir, "jobs.json")

        if not os.path.exists(self.jobs_file):
            raise FileNotFoundError(f"Jobs file not found: {self.jobs_file}")
        
        with open(self.jobs_file, "r") as f:
            self.jobs = json.load(f)

        self.task = self.jobs["task"]
        self.source_list_images = None
        self.reference_list_images = None
        self.num_processes = os.cpu_count()
        self.chunksize = os.getenv("CHUNKSIZE", 10)
        self.top_n_face = int(os.getenv("TOP_N_FACE", 3))
        self.min_clustering_samples = int(os.getenv("MIN_CLUSTERING_SAMPLES", 2))
        self.source_cache = os.path.join(self.cache_dir, "source")
        self.reference_cache = os.path.join(self.cache_dir, "reference")

    def run(self):
        """Run the job processor."""
        if self.task == enums.Task.SAMPLE_MATCHING.name:
            self.source_path = self.jobs["source_path"]
            self.reference_path = self.jobs["reference_path"]
            self.preprocess_sample_matching()
        elif self.task == enums.Task.CLUSTERING.name:
            self.source_path = self.jobs["source_path"]
            self.preprocess_clustering()
        else:
            raise NotImplementedError(f"Task not implemented: {self.task}")

    def preprocess_sample_matching(self) -> dict:
        """Preprocessing for the matching algorithm."""
        result = {}

        if len(self.source_list_images) == 0:
            result["error"] = enums.ErrorMessage.SOURCE_FOLDER_EMPTY.value
            return result

        if len(self.reference_list_images) == 0:
            result["error"] = enums.ErrorMessage.REFERENCE_FOLDER_EMPTY.value
            return result
        
        worker.run_model_mp(
            self.source_list_images,
            self.num_processes,
            self.source_cache,
            self.fail_path,
            self.top_n_face,
        )

        worker.run_model_mp(
            self.reference_list_images,
            self.num_processes,
            self.reference_cache,
            self.fail_path,
            self.top_n_face,
        )

        inputs = {
            "source_cache": self.source_cache,
            "reference_cache": self.reference_cache,
            "source_list_images": self.source_list_images,
            "reference_list_images": self.reference_list_images,
            "output_path": self.output_path,
        }

        return inputs

    def preprocess_clustering(self) -> dict:
        """Preprocessing for the clustering algorithm."""
        self.source_list_images = utils.search_all_images(self.source_path)

        if len(self.source_list_images) == 0:
            self.main_window.error_dialog(
                "Invalid Command", enums.ErrorMessage.SOURCE_FOLDER_EMPTY.value
            )
            return

        self.progress_bar.value = 25
        self.display_console_message(
            f"Processing {len(self.source_list_images)} source images."
        )

        worker.run_model_mp(
            self.source_list_images,
            self.num_processes,
            self.source_cache,
            self.fail_path,
            self.top_n_face,
            self.task_queue,
            self.result_queue
        )
        self.progress_bar.value = 50
        self.display_console_message(
            "Embedding conversion completed. Now Clustering the results."
        )

        # HDBSCAN outperforms DBSCAN and OPTICS in most cases.
        inputs = {
            "source_cache": self.source_cache,
            "source_list_images": self.source_list_images,
            "clustering_algorithm": enums.ClusteringAlgorithm.HDBSCAN.value,
            "eps": 0.5,
            "min_samples": 2,
            "output_path": self.output_path,
            "fail_path": self.fail_path,
        }

        return inputs


if __name__ == "__main__":
    processor = JobProcessor()
    processor.run()