"""Run multiprocessing code based on the jobs generated from app.py"""
import photolink.workers as workers 

class JobProcessor():
    """Run multiprocessing code based on the jobs generated from app.py"""

    def __init__(self, jobs):
        self.jobs = jobs
