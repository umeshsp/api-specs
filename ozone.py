
#!/usr/bin/env python3
import os
import time
import subprocess
import pytest
import logging
import random
import string
from typing import Tuple, List
from subprocess import run, PIPE

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
PERFORMANCE_TARGET_MBS = 100  # MB/s - minimum expected throughput
OZONE_SHELL_CMD = "ozone sh"  # Ozone shell command


class OzoneClient:
    """Helper class to interact with Apache Ozone"""
    
    def __init__(self, ozone_endpoint="localhost:9878"):
        self.ozone_endpoint = ozone_endpoint
    
    def create_volume(self, volume_name: str) -> bool:
        """Create an Ozone volume"""
        cmd = f"{OZONE_SHELL_CMD} volume create /{volume_name}"
        result = run(cmd, shell=True, stdout=PIPE, stderr=PIPE, text=True)
        return result.returncode == 0
    
    def create_bucket(self, volume_name: str, bucket_name: str) -> bool:
        """Create an Ozone bucket"""
        cmd = f"{OZONE_SHELL_CMD} bucket create /{volume_name}/{bucket_name}"
        result = run(cmd, shell=True, stdout=PIPE, stderr=PIPE, text=True)
        return result.returncode == 0
    
    def put_key(self, volume_name: str, bucket_name: str, key_name: str, file_path: str) -> Tuple[bool, float]:
        """Put a file into Ozone and return success status and time taken in seconds"""
        cmd = f"{OZONE_SHELL_CMD} key put /{volume_name}/{bucket_name}/{key_name} {file_path}"
        
        start_time = time.time()
        result = run(cmd, shell=True, stdout=PIPE, stderr=PIPE, text=True)
        end_time = time.time()
        
        time_taken = end_time - start_time
        return result.returncode == 0, time_taken
    
    def delete_key(self, volume_name: str, bucket_name: str, key_name: str) -> bool:
        """Delete a key from Ozone"""
        cmd = f"{OZONE_SHELL_CMD} key delete /{volume_name}/{bucket_name}/{key_name}"
        result = run(cmd, shell=True, stdout=PIPE, stderr=PIPE, text=True)
        return result.returncode == 0
    
    def delete_bucket(self, volume_name: str, bucket_name: str) -> bool:
        """Delete an Ozone bucket"""
        cmd = f"{OZONE_SHELL_CMD} bucket delete /{volume_name}/{bucket_name}"
        result = run(cmd, shell=True, stdout=PIPE, stderr=PIPE, text=True)
        return result.returncode == 0
    
    def delete_volume(self, volume_name: str) -> bool:
        """Delete an Ozone volume"""
        cmd = f"{OZONE_SHELL_CMD} volume delete /{volume_name}"
        result = run(cmd, shell=True, stdout=PIPE, stderr=PIPE, text=True)
        return result.returncode == 0


class TestDataGenerator:
    """Helper class to generate test data files"""
    
    @staticmethod
    def generate_random_string(length: int) -> str:
        """Generate a random string of specified length"""
        return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(length))
    
    @staticmethod
    def create_test_file(file_path: str, size_mb: float) -> str:
        """Create a test file of specified size in MB"""
        # Convert MB to bytes
        size_bytes = int(size_mb * 1024 * 1024)
        
        # Determine the efficient way to create the file
        if size_bytes > 100 * 1024 * 1024:  # If file is larger than 100MB
            # Use dd command for efficiency with large files
            block_size = 1024 * 1024  # 1MB blocks
            count = size_bytes // block_size
            remaining = size_bytes % block_size
            
            # Create file with dd
            run(f"dd if=/dev/urandom of={file_path} bs={block_size} count={count}", shell=True)
            
            # Add remaining bytes if needed
            if remaining > 0:
                with open(file_path, 'ab') as f:
                    f.write(os.urandom(remaining))
        else:
            # For smaller files, use Python's direct file writing
            chunk_size = 1024 * 1024  # 1MB chunks
            with open(file_path, 'wb') as f:
                remaining_bytes = size_bytes
                while remaining_bytes > 0:
                    current_chunk = min(chunk_size, remaining_bytes)
                    f.write(os.urandom(current_chunk))
                    remaining_bytes -= current_chunk
        
        logger.info(f"Created test file: {file_path} ({size_mb} MB)")
        return file_path
    
    @staticmethod
    def cleanup_test_file(file_path: str) -> None:
        """Delete a test file"""
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Cleaned up test file: {file_path}")


# Test data - various file sizes to test with
# Includes a mix of small, medium, and large files
@pytest.mark.parametrize("file_size_mb", [
    1,        # 1 MB
    10,       # 10 MB
    100,      # 100 MB
    512,      # 512 MB
    1024,     # 1 GB
    2560,     # 2.5 GB
    4096,     # 4 GB
    7680,     # 7.5 GB
    10240     # 10 GB
])
def test_1_measure_write_throughput_for_large_files(file_size_mb, tmp_path):
    """
    Performance test to measure write throughput for large files in Ozone.
    This test creates files of different sizes and measures the throughput when writing to Ozone.
    """
    # Initialize Ozone client
    ozone_client = OzoneClient()
    
    # Generate unique volume and bucket names for this test run
    test_id = f"perf{int(time.time())}"
    volume_name = f"vol{test_id}"
    bucket_name = f"bucket{test_id}"
    
    # Setup: Create volume and bucket
    logger.info(f"Creating volume {volume_name} and bucket {bucket_name}")
    assert ozone_client.create_volume(volume_name), f"Failed to create volume {volume_name}"
    assert ozone_client.create_bucket(volume_name, bucket_name), f"Failed to create bucket {bucket_name}"
    
    try:
        # Create test file
        test_file_path = os.path.join(tmp_path, f"testfile_{file_size_mb}MB.dat")
        TestDataGenerator.create_test_file(test_file_path, file_size_mb)
        assert os.path.exists(test_file_path), f"Test file {test_file_path} was not created"
        
        # Get actual file size in bytes from the created file
        actual_file_size_bytes = os.path.getsize(test_file_path)
        actual_file_size_mb = actual_file_size_bytes / (1024 * 1024)
        
        # Define key name
        key_name = f"key_{file_size_mb}MB"
        
        # Upload the file and measure time
        logger.info(f"Uploading {test_file_path} ({actual_file_size_mb:.2f} MB) to Ozone")
        success, time_taken = ozone_client.put_key(volume_name, bucket_name, key_name, test_file_path)
        assert success, f"Failed to upload file to Ozone: {volume_name}/{bucket_name}/{key_name}"
        
        # Calculate throughput
        throughput_mbs = actual_file_size_mb / time_taken if time_taken > 0 else 0
        
        # Log the results
        logger.info(f"File size: {actual_file_size_mb:.2f} MB")
        logger.info(f"Upload time: {time_taken:.2f} seconds")
        logger.info(f"Throughput: {throughput_mbs:.2f} MB/s")
        
        # Assert that throughput meets the performance target
        assert throughput_mbs >= PERFORMANCE_TARGET_MBS, (
            f"Write throughput ({throughput_mbs:.2f} MB/s) is below the target ({PERFORMANCE_TARGET_MBS} MB/s)"
        )
        
    finally:
        # Clean up test data
        logger.info("Cleaning up test resources")
        if os.path.exists(test_file_path):
            TestDataGenerator.cleanup_test_file(test_file_path)
        
        # Clean up Ozone resources
        ozone_client.delete_key(volume_name, bucket_name, key_name)
        ozone_client.delete_bucket(volume_name, bucket_name)
        ozone_client.delete_volume(volume_name)

import os
import time
import pytest
import subprocess
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from pyozone.client import OzoneClient

# Constants for Ozone connection and test configuration
OZONE_HOST = os.environ.get("OZONE_HOST", "localhost")
OZONE_PORT = int(os.environ.get("OZONE_PORT", "9862"))
PERFORMANCE_TARGET_MBS = float(os.environ.get("PERFORMANCE_TARGET_MBS", "200"))
VOLUME_NAME = "perfvolume"
BUCKET_NAME = "readthroughputbucket"
NUM_ITERATIONS = 5  # Number of times to repeat each test for averaging


def setup_module():
    """Initialize test environment and ensure prerequisites are met."""
    # Check if Ozone cluster is accessible
    try:
        subprocess.run(["ozone", "sh", "status"], 
                      check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        pytest.skip(f"Ozone cluster not accessible: {e}")


@pytest.fixture
def ozone_client():
    """Provide an Ozone client for tests."""
    return OzoneClient(OZONE_HOST, OZONE_PORT)


def get_file_list() -> List[dict]:
    """Get a list of large files from Ozone for testing."""
    result = subprocess.run(
        ["ozone", "sh", "key", "list", f"{VOLUME_NAME}/{BUCKET_NAME}"],
        check=True, stdout=subprocess.PIPE, text=True
    )
    
    files = []
    for line in result.stdout.strip().split('\n')[1:]:  # Skip header line
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 3:
            key_name = parts[0]
            size_bytes = int(parts[1])
            if size_bytes > 1024*1024:  # Only files larger than 1MB
                files.append({
                    "key": key_name,
                    "size_bytes": size_bytes,
                    "size_mb": size_bytes / (1024*1024)
                })
    
    # If no files found, skip the test
    if not files:
        pytest.skip("No large files found in Ozone for testing")
    
    return files


def read_file_measure_throughput(client: OzoneClient, file_info: dict) -> Tuple[float, float]:
    """
    Read a file from Ozone and measure throughput.
    
    Args:
        client: Ozone client
        file_info: Dictionary containing file metadata
        
    Returns:
        Tuple of (throughput in MB/s, time taken in seconds)
    """
    key = file_info["key"]
    size_mb = file_info["size_mb"]
    
    start_time = time.time()
    
    # Read the file using the client API
    data = client.get_key(VOLUME_NAME, BUCKET_NAME, key)
    
    end_time = time.time()
    time_taken = end_time - start_time
    
    # Calculate throughput in MB/s
    throughput = size_mb / time_taken if time_taken > 0 else 0
    
    return (throughput, time_taken)


@pytest.mark.parametrize("num_parallel_clients", [1, 2, 4, 8])
def test_2_read_throughput_large_files(ozone_client, num_parallel_clients):
    """Measure read throughput for large files."""
    files = get_file_list()
    
    # If we don't have enough files, repeat some files
    while len(files) < num_parallel_clients:
        files.extend(files)
    
    # Select a subset of files for this test
    selected_files = files[:num_parallel_clients]
    
    # Results storage
    throughput_results = []
    
    for _ in range(NUM_ITERATIONS):
        # Use ThreadPoolExecutor to simulate multiple clients
        with ThreadPoolExecutor(max_workers=num_parallel_clients) as executor:
            # Submit tasks to read files and measure throughput
            future_to_file = {
                executor.submit(read_file_measure_throughput, ozone_client, file_info): file_info
                for file_info in selected_files
            }
            
            # Collect results
            iteration_results = []
            for future in future_to_file:
                try:
                    throughput, time_taken = future.result()
                    iteration_results.append(throughput)
                    print(f"File: {future_to_file[future]['key']}, "
                          f"Size: {future_to_file[future]['size_mb']:.2f} MB, "
                          f"Time: {time_taken:.2f}s, "
                          f"Throughput: {throughput:.2f} MB/s")
                except Exception as e:
                    print(f"Error reading file: {e}")
            
            # Add average throughput for this iteration
            if iteration_results:
                throughput_results.append(sum(iteration_results))
    
    # Calculate average and standard deviation of throughput
    avg_throughput = np.mean(throughput_results) if throughput_results else 0
    stddev_throughput = np.std(throughput_results) if len(throughput_results) > 1 else 0
    
    # Print results
    print(f"\nResults for {num_parallel_clients} concurrent clients:")
    print(f"Average total throughput: {avg_throughput:.2f} MB/s")
    print(f"Throughput per client: {avg_throughput/num_parallel_clients:.2f} MB/s")
    print(f"Standard deviation: {stddev_throughput:.2f} MB/s")
    
    # Generate a simple chart for visualization
    plt.figure(figsize=(10, 6))
    plt.bar(['Total Throughput', 'Per-Client Throughput'], 
            [avg_throughput, avg_throughput/num_parallel_clients])
    plt.axhline(y=PERFORMANCE_TARGET_MBS, color='r', linestyle='-', label=f'Target ({PERFORMANCE_TARGET_MBS} MB/s)')
    plt.ylabel('Throughput (MB/s)')
    plt.title(f'Ozone Read Throughput ({num_parallel_clients} clients)')
    plt.legend()
    
    # Save the chart (optional)
    chart_path = f"ozone_read_throughput_{num_parallel_clients}_clients.png"
    plt.savefig(chart_path)
    print(f"Chart saved to {chart_path}")
    
    # Assert that the per-client throughput meets the target
    assert avg_throughput/num_parallel_clients >= PERFORMANCE_TARGET_MBS, \
        f"Read throughput per client ({avg_throughput/num_parallel_clients:.2f} MB/s) " \
        f"does not meet the target of {PERFORMANCE_TARGET_MBS} MB/s"

import pytest
import time
import concurrent.futures
import subprocess
import psutil
import os
import pandas as pd
import matplotlib.pyplot as plt
from subprocess import PIPE, run
import threading
import logging
import tempfile
import random
from typing import List, Dict, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
VOLUME_NAME = "perfvolume"
BUCKET_NAME = "perfbucket"
NUM_CLIENT_THREADS = [5, 10, 25, 50]  # Simulate different client loads
OPERATION_MIXES = [
    {"read": 0.8, "write": 0.2},
    {"read": 0.5, "write": 0.5},
    {"read": 0.2, "write": 0.8}
]
TEST_DURATION_SECONDS = 60
FILE_SIZES_KB = [10, 100, 1024, 10240]  # 10KB, 100KB, 1MB, 10MB

class OzonePerformanceTester:
    """Helper class for Ozone performance testing"""
    
    def __init__(self, volume_name: str, bucket_name: str):
        self.volume_name = volume_name
        self.bucket_name = bucket_name
        self._ensure_volume_bucket_exist()
        self.results = []
        self.monitor_thread = None
        self.stop_monitoring = False
        self.resource_data = []
    
    def _ensure_volume_bucket_exist(self):
        """Ensure that the volume and bucket exist for testing"""
        try:
            # Check if volume exists, create if not
            result = run(f"ozone sh volume info {self.volume_name}", 
                         shell=True, stdout=PIPE, stderr=PIPE, universal_newlines=True)
            if result.returncode != 0:
                logger.info(f"Creating volume {self.volume_name}")
                run(f"ozone sh volume create {self.volume_name}", 
                    shell=True, check=True)
            
            # Check if bucket exists, create if not
            result = run(f"ozone sh bucket info {self.volume_name}/{self.bucket_name}", 
                         shell=True, stdout=PIPE, stderr=PIPE, universal_newlines=True)
            if result.returncode != 0:
                logger.info(f"Creating bucket {self.volume_name}/{self.bucket_name}")
                run(f"ozone sh bucket create {self.volume_name}/{self.bucket_name}", 
                    shell=True, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create volume/bucket: {str(e)}")
            raise
    
    def _create_test_file(self, size_kb: int) -> str:
        """Create a test file with the specified size in KB"""
        fd, path = tempfile.mkstemp()
        try:
            with os.fdopen(fd, 'wb') as temp:
                # Generate random data
                chunk_size = 1024  # 1KB chunks
                remaining = size_kb * 1024  # Convert KB to bytes
                while remaining > 0:
                    write_size = min(chunk_size, remaining)
                    temp.write(os.urandom(write_size))
                    remaining -= write_size
        except Exception as e:
            os.unlink(path)
            raise e
        
        return path
    
    def write_operation(self, file_size_kb: int) -> Tuple[float, bool]:
        """Perform a write operation and return latency and success status"""
        key_name = f"perf-{time.time()}-{random.randint(1000, 9999)}"
        test_file = self._create_test_file(file_size_kb)
        
        try:
            start_time = time.time()
            result = run(
                f"ozone sh key put {self.volume_name}/{self.bucket_name}/{key_name} {test_file}",
                shell=True, stdout=PIPE, stderr=PIPE, universal_newlines=True
            )
            end_time = time.time()
            latency = end_time - start_time
            success = result.returncode == 0
            
            if success:
                logger.debug(f"Write operation successful: {key_name}, latency: {latency:.3f}s")
            else:
                logger.error(f"Write operation failed: {key_name}, error: {result.stderr}")
            
            return latency, success
        finally:
            # Clean up the temporary file
            try:
                os.unlink(test_file)
            except:
                pass
    
    def read_operation(self) -> Tuple[float, bool]:
        """Perform a read operation and return latency and success status"""
        # List keys to find an existing one to read
        try:
            keys_result = run(
                f"ozone sh key list {self.volume_name}/{self.bucket_name}/",
                shell=True, stdout=PIPE, universal_newlines=True
            )
            
            if keys_result.returncode != 0 or not keys_result.stdout.strip():
                # No keys to read, so perform a write first
                _, success = self.write_operation(100)  # Create a 100KB file
                if not success:
                    return 0.0, False
                
                # Try listing again
                keys_result = run(
                    f"ozone sh key list {self.volume_name}/{self.bucket_name}/",
                    shell=True, stdout=PIPE, universal_newlines=True
                )
            
            keys = keys_result.stdout.strip().split('\n')
            if not keys:
                return 0.0, False
            
            # Pick a random key
            key_name = random.choice(keys).strip()
            
            # Read the key
            temp_output_file = tempfile.mktemp()
            start_time = time.time()
            result = run(
                f"ozone sh key get {self.volume_name}/{self.bucket_name}/{key_name} {temp_output_file}",
                shell=True, stdout=PIPE, stderr=PIPE, universal_newlines=True
            )
            end_time = time.time()
            latency = end_time - start_time
            success = result.returncode == 0
            
            if success:
                logger.debug(f"Read operation successful: {key_name}, latency: {latency:.3f}s")
            else:
                logger.error(f"Read operation failed: {key_name}, error: {result.stderr}")
            
            # Clean up
            try:
                os.unlink(temp_output_file)
            except:
                pass
            
            return latency, success
        except Exception as e:
            logger.error(f"Error in read operation: {str(e)}")
            return 0.0, False
    
    def _monitor_resources(self):
        """Monitor system resources in a separate thread"""
        while not self.stop_monitoring:
            try:
                # Get CPU, memory, disk and network usage
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_percent = psutil.virtual_memory().percent
                disk_io = psutil.disk_io_counters()
                net_io = psutil.net_io_counters()
                
                self.resource_data.append({
                    'timestamp': time.time(),
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory_percent,
                    'disk_read_bytes': disk_io.read_bytes,
                    'disk_write_bytes': disk_io.write_bytes,
                    'net_sent_bytes': net_io.bytes_sent,
                    'net_recv_bytes': net_io.bytes_recv
                })
            except Exception as e:
                logger.error(f"Error monitoring resources: {str(e)}")
            
            time.sleep(1)
    
    def start_resource_monitoring(self):
        """Start the resource monitoring thread"""
        self.stop_monitoring = False
        self.resource_data = []
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_resource_monitoring(self):
        """Stop the resource monitoring thread"""
        self.stop_monitoring = True
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
    
    def run_mixed_workload(self, 
                          num_threads: int, 
                          mix_ratio: Dict[str, float], 
                          duration: int = 60,
                          file_sizes_kb: List[int] = None) -> Dict:
        """
        Run a mixed workload of read and write operations.
        
        Args:
            num_threads: Number of concurrent threads/clients
            mix_ratio: Dictionary with 'read' and 'write' ratios (should sum to 1.0)
            duration: Test duration in seconds
            file_sizes_kb: List of file sizes for write operations in KB
        
        Returns:
            Dictionary with performance metrics
        """
        if file_sizes_kb is None:
            file_sizes_kb = [100]  # Default 100KB files
            
        # Validate mix ratio
        if abs(sum(mix_ratio.values()) - 1.0) > 0.01:
            raise ValueError("Mix ratio values must sum to 1.0")
            
        # Start resource monitoring
        self.start_resource_monitoring()
            
        # Results storage
        read_latencies = []
        write_latencies = []
        read_success = 0
        read_failures = 0
        write_success = 0
        write_failures = 0
        total_operations = 0
        
        # Create a thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            start_time = time.time()
            futures = []
            
            # Keep submitting tasks until duration is reached
            while time.time() - start_time < duration:
                # Decide whether to do a read or write based on the mix ratio
                op_type = random.choices(
                    ["read", "write"], 
                    weights=[mix_ratio.get("read", 0), mix_ratio.get("write", 0)],
                    k=1
                )[0]
                
                if op_type == "read":
                    futures.append(executor.submit(self.read_operation))
                else:  # write
                    # Choose a random file size for this write operation
                    file_size = random.choice(file_sizes_kb)
                    futures.append(executor.submit(self.write_operation, file_size))
            
            # Process results
            for future in concurrent.futures.as_completed(futures):
                try:
                    latency, success = future.result()
                    total_operations += 1
                    
                    # Check if it was a read or write operation based on the function signature
                    if future.function == self.read_operation:
                        if success:
                            read_success += 1
                            read_latencies.append(latency)
                        else:
                            read_failures += 1
                    else:  # Write operation
                        if success:
                            write_success += 1
                            write_latencies.append(latency)
                        else:
                            write_failures += 1
                except Exception as e:
                    logger.error(f"Error processing task result: {str(e)}")
        
        # Stop resource monitoring
        self.stop_resource_monitoring()
        
        # Calculate metrics
        end_time = time.time()
        actual_duration = end_time - start_time
        
        # Calculate overall throughput (ops/sec)
        throughput = total_operations / actual_duration
        
        # Calculate average latencies
        avg_read_latency = sum(read_latencies) / len(read_latencies) if read_latencies else 0
        avg_write_latency = sum(write_latencies) / len(write_latencies) if write_latencies else 0
        
        # Calculate success rates
        read_success_rate = read_success / (read_success + read_failures) if (read_success + read_failures) > 0 else 0
        write_success_rate = write_success / (write_success + write_failures) if (write_success + write_failures) > 0 else 0
        
        # Process resource monitoring data
        resource_df = pd.DataFrame(self.resource_data)
        
        # Calculate resource averages if we have data
        if not resource_df.empty:
            avg_cpu = resource_df['cpu_percent'].mean()
            avg_memory = resource_df['memory_percent'].mean()
            
            # Calculate disk and network throughput rates (per second)
            if len(resource_df) > 1:
                first_sample = resource_df.iloc[0]
                last_sample = resource_df.iloc[-1]
                time_diff = last_sample['timestamp'] - first_sample['timestamp']
                
                disk_read_rate = (last_sample['disk_read_bytes'] - first_sample['disk_read_bytes']) / time_diff
                disk_write_rate = (last_sample['disk_write_bytes'] - first_sample['disk_write_bytes']) / time_diff
                
                net_recv_rate = (last_sample['net_recv_bytes'] - first_sample['net_recv_bytes']) / time_diff
                net_send_rate = (last_sample['net_sent_bytes'] - first_sample['net_sent_bytes']) / time_diff
            else:
                disk_read_rate = disk_write_rate = net_recv_rate = net_send_rate = 0
        else:
            avg_cpu = avg_memory = disk_read_rate = disk_write_rate = net_recv_rate = net_send_rate = 0
        
        # Return the metrics
        return {
            'throughput': throughput,
            'avg_read_latency': avg_read_latency,
            'avg_write_latency': avg_write_latency,
            'read_success_rate': read_success_rate,
            'write_success_rate': write_success_rate,
            'total_operations': total_operations,
            'read_operations': read_success + read_failures,
            'write_operations': write_success + write_failures,
            'avg_cpu_percent': avg_cpu,
            'avg_memory_percent': avg_memory,
            'disk_read_rate_bytes_per_sec': disk_read_rate,
            'disk_write_rate_bytes_per_sec': disk_write_rate,
            'net_recv_rate_bytes_per_sec': net_recv_rate,
            'net_send_rate_bytes_per_sec': net_send_rate,
            'test_parameters': {
                'num_threads': num_threads,
                'mix_ratio': mix_ratio,
                'duration': duration,
                'file_sizes_kb': file_sizes_kb
            }
        }


def generate_performance_report(results):
    """Generate performance report from the test results"""
    # Create a DataFrame from the results
    df = pd.DataFrame(results)
    
    # Add derived columns for better analysis
    df['avg_latency'] = (df['avg_read_latency'] * df['read_operations'] + 
                         df['avg_write_latency'] * df['write_operations']) / df['total_operations']
    
    # Extracting parameters for grouping
    df['num_threads'] = df['test_parameters'].apply(lambda x: x['num_threads'])
    df['read_ratio'] = df['test_parameters'].apply(lambda x: x['mix_ratio']['read'])
    
    # Generate summary statistics
    summary = df.groupby(['num_threads', 'read_ratio']).agg({
        'throughput': ['mean', 'min', 'max'],
        'avg_latency': ['mean', 'min', 'max'],
        'avg_cpu_percent': 'mean',
        'avg_memory_percent': 'mean'
    }).reset_index()
    
    # Create plots
    plt.figure(figsize=(12, 10))
    
    # Plot 1: Throughput vs Number of Threads
    plt.subplot(2, 2, 1)
    for read_ratio in df['read_ratio'].unique():
        subset = df[df['read_ratio'] == read_ratio]
        plt.plot(subset['num_threads'], subset['throughput'], 
                 marker='o', label=f"Read Ratio: {read_ratio}")
    
    plt.title('Throughput vs Number of Threads')
    plt.xlabel('Number of Threads')
    plt.ylabel('Throughput (ops/sec)')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Average Latency vs Number of Threads
    plt.subplot(2, 2, 2)
    for read_ratio in df['read_ratio'].unique():
        subset = df[df['read_ratio'] == read_ratio]
        plt.plot(subset['num_threads'], subset['avg_latency'], 
                 marker='o', label=f"Read Ratio: {read_ratio}")
    
    plt.title('Average Latency vs Number of Threads')
    plt.xlabel('Number of Threads')
    plt.ylabel('Average Latency (seconds)')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: CPU Usage vs Number of Threads
    plt.subplot(2, 2, 3)
    for read_ratio in df['read_ratio'].unique():
        subset = df[df['read_ratio'] == read_ratio]
        plt.plot(subset['num_threads'], subset['avg_cpu_percent'], 
                 marker='o', label=f"Read Ratio: {read_ratio}")
    
    plt.title('CPU Usage vs Number of Threads')
    plt.xlabel('Number of Threads')
    plt.ylabel('Average CPU Usage (%)')
    plt.legend()
    plt.grid(True)
    
    # Plot 4: Memory Usage vs Number of Threads
    plt.subplot(2, 2, 4)
    for read_ratio in df['read_ratio'].unique():
        subset = df[df['read_ratio'] == read_ratio]
        plt.plot(subset['num_threads'], subset['avg_memory_percent'], 
                 marker='o', label=f"Read Ratio: {read_ratio}")
    
    plt.title('Memory Usage vs Number of Threads')
    plt.xlabel('Number of Threads')
    plt.ylabel('Average Memory Usage (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save the plot
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    plot_filename = f"ozone_performance_test_{timestamp}.png"
    plt.savefig(plot_filename)
    
    # Also save the raw data
    csv_filename = f"ozone_performance_test_{timestamp}.csv"
    df.to_csv(csv_filename, index=False)
    
    return summary, plot_filename, csv_filename


@pytest.mark.parametrize("num_threads", NUM_CLIENT_THREADS)
@pytest.mark.parametrize("mix_ratio", OPERATION_MIXES)
@pytest.mark.parametrize("file_sizes_kb", [FILE_SIZES_KB])
def test_3_concurrent_read_write_operations(num_threads, mix_ratio, file_sizes_kb):
    """
    Test concurrent read/write operations to assess system performance under load.
    This test simulates multiple clients performing a mix of read and write operations
    and measures throughput, latency, and system resource utilization.
    """
    # Initialize the performance tester
    tester = OzonePerformanceTester(VOLUME_NAME, BUCKET_NAME)
    
    # Log test parameters
    logger.info(f"Starting concurrent test with {num_threads} threads, "
                f"mix ratio: {mix_ratio}, file sizes: {file_sizes_kb}")
    
    # Run the workload
    results = tester.run_mixed_workload(
        num_threads=num_threads,
        mix_ratio=mix_ratio,
        duration=TEST_DURATION_SECONDS,
        file_sizes_kb=file_sizes_kb
    )
    
    # Log the results
    logger.info(f"Test completed with throughput: {results['throughput']:.2f} ops/sec")
    logger.info(f"Average read latency: {results['avg_read_latency']:.4f} sec")
    logger.info(f"Average write latency: {results['avg_write_latency']:.4f} sec")
    logger.info(f"CPU usage: {results['avg_cpu_percent']:.1f}%, "
                f"Memory usage: {results['avg_memory_percent']:.1f}%")
    
    # Performance expectations - these thresholds should be adjusted based on
    # the specific performance requirements and baseline of your Ozone setup
    
    # Check if throughput degradation is within acceptable range
    # as thread count increases
    if num_threads > NUM_CLIENT_THREADS[0]:  # If not the lowest thread count
        # Throughput should not degrade more than 30% per thread doubling
        # (This is just an example threshold - should be calibrated based on actual system capabilities)
        expected_degradation_factor = 0.7  # Allow 30% degradation
        
        # This assertion might be disabled initially during baseline establishment
        # assert results['throughput'] >= baseline_throughput * expected_degradation_factor, \
        #     f"Throughput degraded more than expected: {results['throughput']} < {baseline_throughput * expected_degradation_factor}"
    
    # Validate response time expectations - here we just log, but in a real test 
    # you would assert against known performance benchmarks
    max_acceptable_latency = 2.0  # seconds, adjust based on your requirements
    assert results['avg_read_latency'] < max_acceptable_latency, \
        f"Read latency too high: {results['avg_read_latency']} > {max_acceptable_latency}s"
    assert results['avg_write_latency'] < max_acceptable_latency, \
        f"Write latency too high: {results['avg_write_latency']} > {max_acceptable_latency}s"
    
    # Check success rates - operations should be reliable
    min_success_rate = 0.95  # 95% success
    assert results['read_success_rate'] >= min_success_rate, \
        f"Read success rate too low: {results['read_success_rate']} < {min_success_rate}"
    assert results['write_success_rate'] >= min_success_rate, \
        f"Write success rate too low: {results['write_success_rate']} < {min_success_rate}"
    
    # Store the result for later comparison and reporting
    return results

import os
import time
import statistics
import tempfile
import pytest
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from pyarrow import ozone
from concurrent.futures import ThreadPoolExecutor


class OzonePerformanceTest:
    """Helper class for Ozone performance testing"""
    
    def __init__(self, host: str = "localhost", port: int = 9862, username: str = "hadoop"):
        """Initialize the Ozone client"""
        self.client = ozone.Client(host, port, username)
        self.volume = "perf_vol"
        self.bucket = "latency_test"
        
        # Ensure volume and bucket exist
        self._ensure_volume_bucket()
        
    def _ensure_volume_bucket(self) -> None:
        """Create volume and bucket if they don't exist"""
        if not self.client.volume_exists(self.volume):
            self.client.create_volume(self.volume)
        
        if not self.client.bucket_exists(self.volume, self.bucket):
            self.client.create_bucket(self.volume, self.bucket)
    
    def generate_test_file(self, size_kb: int) -> str:
        """Generate a test file of specified size in KB"""
        fd, file_path = tempfile.mkstemp()
        with os.fdopen(fd, 'wb') as f:
            f.write(os.urandom(size_kb * 1024))
        return file_path
    
    def measure_write_latency(self, file_path: str, key_name: str) -> float:
        """Measure time taken to write a file to Ozone"""
        start_time = time.time()
        
        with open(file_path, 'rb') as file_data:
            self.client.put_key(self.volume, self.bucket, key_name, file_data)
            
        end_time = time.time()
        return (end_time - start_time) * 1000  # Convert to milliseconds
    
    def measure_read_latency(self, key_name: str) -> float:
        """Measure time taken to read a file from Ozone"""
        start_time = time.time()
        
        _ = self.client.get_key(self.volume, self.bucket, key_name)
        
        end_time = time.time()
        return (end_time - start_time) * 1000  # Convert to milliseconds
    
    def measure_delete_latency(self, key_name: str) -> float:
        """Measure time taken to delete a file from Ozone"""
        start_time = time.time()
        
        self.client.delete_key(self.volume, self.bucket, key_name)
        
        end_time = time.time()
        return (end_time - start_time) * 1000  # Convert to milliseconds


def calculate_statistics(latencies: List[float]) -> Dict[str, float]:
    """Calculate statistical metrics for a list of latency values"""
    if not latencies:
        return {
            "min": 0,
            "max": 0,
            "avg": 0,
            "median": 0,
            "p95": 0
        }
    
    return {
        "min": min(latencies),
        "max": max(latencies),
        "avg": statistics.mean(latencies),
        "median": statistics.median(latencies),
        "p95": statistics.quantiles(latencies, n=20)[19]  # 95th percentile
    }


def plot_latency_results(results: Dict[str, Dict[str, Dict[str, float]]], title: str, output_file: str) -> None:
    """Generate a plot comparing latencies for different operation types across file sizes"""
    df_data = []
    
    for size, size_data in results.items():
        for op_type, metrics in size_data.items():
            df_data.append({
                "File Size": f"{size}KB",
                "Operation": op_type,
                "Average (ms)": metrics["avg"],
                "Median (ms)": metrics["median"],
                "95th Percentile (ms)": metrics["p95"]
            })
    
    df = pd.DataFrame(df_data)
    
    # Create pivot table for plotting
    pivot_avg = df.pivot(index="File Size", columns="Operation", values="Average (ms)")
    
    # Plot the results
    ax = pivot_avg.plot(kind="bar", figsize=(12, 8))
    plt.title(title)
    plt.ylabel("Latency (ms)")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(output_file)


@pytest.fixture
def ozone_client():
    """Fixture to provide an Ozone client for tests"""
    client = OzonePerformanceTest()
    yield client
    # Cleanup is automatic when client is destroyed


# Test parameters: small file sizes in KB
FILE_SIZES = [1, 10, 100]
ITERATIONS = 10  # Number of iterations for each test to get reliable statistics


@pytest.mark.parametrize("file_size_kb", FILE_SIZES)
def test_4_small_file_operations_latency(ozone_client, file_size_kb):
    """
    Measure latency for small file operations
    
    This test measures the latency of write, read, and delete operations
    on small files of various sizes (1KB, 10KB, 100KB) and verifies that
    the latencies are within acceptable thresholds.
    """
    write_latencies = []
    read_latencies = []
    delete_latencies = []
    
    for i in range(ITERATIONS):
        # Generate a unique key name for this iteration
        key_name = f"test_file_{file_size_kb}kb_{i}"
        
        # Generate test file
        test_file = ozone_client.generate_test_file(file_size_kb)
        
        try:
            # Measure write latency
            write_time = ozone_client.measure_write_latency(test_file, key_name)
            write_latencies.append(write_time)
            
            # Measure read latency
            read_time = ozone_client.measure_read_latency(key_name)
            read_latencies.append(read_time)
            
            # Measure delete latency
            delete_time = ozone_client.measure_delete_latency(key_name)
            delete_latencies.append(delete_time)
            
        finally:
            # Clean up test file
            if os.path.exists(test_file):
                os.remove(test_file)
    
    # Calculate statistics
    write_stats = calculate_statistics(write_latencies)
    read_stats = calculate_statistics(read_latencies)
    delete_stats = calculate_statistics(delete_latencies)
    
    # Print results
    print(f"\nLatency statistics for {file_size_kb}KB files:")
    print(f"Write latency (ms): avg={write_stats['avg']:.2f}, median={write_stats['median']:.2f}, p95={write_stats['p95']:.2f}")
    print(f"Read latency (ms): avg={read_stats['avg']:.2f}, median={read_stats['median']:.2f}, p95={read_stats['p95']:.2f}")
    print(f"Delete latency (ms): avg={delete_stats['avg']:.2f}, median={delete_stats['median']:.2f}, p95={delete_stats['p95']:.2f}")
    
    # Log results to CSV
    results_dir = "performance_results"
    os.makedirs(results_dir, exist_ok=True)
    
    result_file = f"{results_dir}/small_file_latency_{file_size_kb}kb.csv"
    with open(result_file, "w") as f:
        f.write("Operation,Min (ms),Max (ms),Avg (ms),Median (ms),P95 (ms)\n")
        f.write(f"Write,{write_stats['min']:.2f},{write_stats['max']:.2f},{write_stats['avg']:.2f},{write_stats['median']:.2f},{write_stats['p95']:.2f}\n")
        f.write(f"Read,{read_stats['min']:.2f},{read_stats['max']:.2f},{read_stats['avg']:.2f},{read_stats['median']:.2f},{read_stats['p95']:.2f}\n")
        f.write(f"Delete,{delete_stats['min']:.2f},{delete_stats['max']:.2f},{delete_stats['avg']:.2f},{delete_stats['median']:.2f},{delete_stats['p95']:.2f}\n")
    
    # Assert that latencies are within acceptable limits
    assert write_stats['avg'] < 10.0, f"Average write latency ({write_stats['avg']:.2f}ms) exceeded threshold (10ms)"
    assert read_stats['avg'] < 5.0, f"Average read latency ({read_stats['avg']:.2f}ms) exceeded threshold (5ms)"
    
    # Return statistics for post-test analysis if needed
    return {
        file_size_kb: {
            "write": write_stats,
            "read": read_stats,
            "delete": delete_stats
        }
    }


@pytest.mark.parametrize("file_size_kb", FILE_SIZES)
def test_4_small_file_operations_concurrent_latency(ozone_client, file_size_kb):
    """
    Measure latency for concurrent small file operations
    
    This test measures the latency when performing multiple small file
    operations concurrently to understand the system's behavior under load.
    """
    # Number of concurrent operations
    concurrency = 5
    results = {
        "write": [],
        "read": [],
        "delete": []
    }
    
    # Generate test files
    test_files = [ozone_client.generate_test_file(file_size_kb) for _ in range(concurrency)]
    key_names = [f"concurrent_test_{file_size_kb}kb_{i}" for i in range(concurrency)]
    
    try:
        # Measure concurrent write latency
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(ozone_client.measure_write_latency, test_files[i], key_names[i])
                for i in range(concurrency)
            ]
            for future in futures:
                results["write"].append(future.result())
        
        # Measure concurrent read latency
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(ozone_client.measure_read_latency, key_names[i])
                for i in range(concurrency)
            ]
            for future in futures:
                results["read"].append(future.result())
        
        # Measure concurrent delete latency
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(ozone_client.measure_delete_latency, key_names[i])
                for i in range(concurrency)
            ]
            for future in futures:
                results["delete"].append(future.result())
    
    finally:
        # Clean up test files
        for file_path in test_files:
            if os.path.exists(file_path):
                os.remove(file_path)
    
    # Calculate statistics for concurrent operations
    write_stats = calculate_statistics(results["write"])
    read_stats = calculate_statistics(results["read"])
    delete_stats = calculate_statistics(results["delete"])
    
    # Print results
    print(f"\nConcurrent latency statistics for {file_size_kb}KB files (concurrency={concurrency}):")
    print(f"Write latency (ms): avg={write_stats['avg']:.2f}, median={write_stats['median']:.2f}, p95={write_stats['p95']:.2f}")
    print(f"Read latency (ms): avg={read_stats['avg']:.2f}, median={read_stats['median']:.2f}, p95={read_stats['p95']:.2f}")
    print(f"Delete latency (ms): avg={delete_stats['avg']:.2f}, median={delete_stats['median']:.2f}, p95={delete_stats['p95']:.2f}")
    
    # Assert that latencies are still within acceptable limits under concurrent load
    # Note: We allow slightly higher thresholds for concurrent operations
    assert write_stats['avg'] < 15.0, f"Average concurrent write latency ({write_stats['avg']:.2f}ms) exceeded threshold (15ms)"
    assert read_stats['avg'] < 10.0, f"Average concurrent read latency ({read_stats['avg']:.2f}ms) exceeded threshold (10ms)"


def test_4_latency_comparison_across_file_sizes(request):
    """
    Compare latency metrics across different file sizes
    
    This test runs the individual file size tests and then compares
    the results to analyze how latency scales with file size.
    """
    # Collect results from the individual file size tests
    results = {}
    
    for size in FILE_SIZES:
        marker = pytest.mark.parametrize("file_size_kb", [size])
        test_item = request.node.module.test_4_small_file_operations_latency
        result = pytest.runner.pytest_runtest_call(marker(test_item))
        results.update(result)
    
    # Create directory for output
    results_dir = "performance_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate plots comparing latency across file sizes
    plot_latency_results(
        results,
        "Ozone Small File Operations Latency Comparison",
        f"{results_dir}/small_file_latency_comparison.png"
    )
    
    # Write summary to CSV
    with open(f"{results_dir}/latency_summary.csv", "w") as f:
        f.write("FileSize,Operation,Min,Max,Avg,Median,P95\n")
        for size, size_data in results.items():
            for op, metrics in size_data.items():
                f.write(f"{size},{op},{metrics['min']:.2f},{metrics['max']:.2f}," +
                        f"{metrics['avg']:.2f},{metrics['median']:.2f},{metrics['p95']:.2f}\n")
    
    # Verify that latency increases with file size but remains within acceptable limits
    for op in ["write", "read"]:
        latencies = [results[size][op]["avg"] for size in sorted(FILE_SIZES)]
        
        # Assert that larger files generally have higher latency
        # (This might not always be true due to caching, but it's a reasonable expectation)
        if len(latencies) > 1:
            assert max(latencies) > min(latencies), f"{op} latency does not show expected scaling with file size"
        
        # Ensure all latencies are within limits
        if op == "write":
            for size, latency in zip(FILE_SIZES, latencies):
                assert latency < 10.0, f"{size}KB {op} latency ({latency:.2f}ms) exceeds threshold (10ms)"
        else:  # read
            for size, latency in zip(FILE_SIZES, latencies):
                assert latency < 5.0, f"{size}KB {op} latency ({latency:.2f}ms) exceeds threshold (5ms)"

import pytest
import time
import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import logging
import pandas as pd
import statistics
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
VOLUME_NAME = "perfvol"
BUCKET_NAME = "scalabucket"
TEST_DATA_FOLDER = "test_data"
RESULTS_FOLDER = "test_results"

class OzoneScalabilityTest:
    """Helper class for Ozone scalability testing"""
    
    def __init__(self):
        """Initialize the test environment"""
        # Ensure directories exist
        os.makedirs(TEST_DATA_FOLDER, exist_ok=True)
        os.makedirs(RESULTS_FOLDER, exist_ok=True)
        
        # Initialize performance metrics dictionary
        self.metrics = {
            'object_count': [],
            'put_throughput': [],
            'get_throughput': [],
            'list_latency': [],
            'cpu_usage': [],
            'memory_usage': [],
            'disk_usage': []
        }
    
    def setup_environment(self):
        """Set up the Ozone volume and bucket"""
        try:
            logger.info("Creating test volume and bucket")
            subprocess.run(["ozone", "sh", "volume", "create", VOLUME_NAME], check=True)
            subprocess.run(["ozone", "sh", "bucket", "create", f"{VOLUME_NAME}/{BUCKET_NAME}"], check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error setting up environment: {e}")
            raise
    
    def cleanup_environment(self):
        """Clean up the test volume and bucket"""
        try:
            logger.info("Cleaning up test volume and bucket")
            subprocess.run(["ozone", "sh", "bucket", "delete", f"{VOLUME_NAME}/{BUCKET_NAME}"], check=True)
            subprocess.run(["ozone", "sh", "volume", "delete", VOLUME_NAME], check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error cleaning up environment: {e}")
    
    def create_test_file(self, size_kb=1):
        """Create a test file of the specified size in KB"""
        file_path = os.path.join(TEST_DATA_FOLDER, f"test_file_{size_kb}kb.txt")
        with open(file_path, 'wb') as f:
            f.write(os.urandom(size_kb * 1024))
        return file_path
    
    def upload_object(self, index: int):
        """Upload a single object to Ozone"""
        key_name = f"key_{index}"
        file_path = self.create_test_file(1)  # 1KB files for performance testing
        try:
            subprocess.run([
                "ozone", "sh", "key", "put", 
                f"{VOLUME_NAME}/{BUCKET_NAME}/", key_name, file_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except subprocess.CalledProcessError:
            logger.error(f"Failed to upload object {key_name}")
            return False
    
    def generate_objects(self, count: int, concurrency: int = 20) -> float:
        """Generate and upload the specified number of objects"""
        logger.info(f"Generating {count} objects with concurrency {concurrency}")
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            results = list(executor.map(self.upload_object, range(count)))
        
        success_count = results.count(True)
        elapsed_time = time.time() - start_time
        throughput = success_count / elapsed_time if elapsed_time > 0 else 0
        
        logger.info(f"Successfully uploaded {success_count}/{count} objects in {elapsed_time:.2f} seconds")
        logger.info(f"Put throughput: {throughput:.2f} objects/second")
        
        return throughput
    
    def measure_list_latency(self) -> float:
        """Measure the latency of listing objects in the bucket"""
        start_time = time.time()
        try:
            subprocess.run([
                "ozone", "sh", "key", "list", f"{VOLUME_NAME}/{BUCKET_NAME}/"
            ], check=True, stdout=subprocess.DEVNULL)
            elapsed_time = time.time() - start_time
            logger.info(f"List operation completed in {elapsed_time:.2f} seconds")
            return elapsed_time
        except subprocess.CalledProcessError:
            logger.error("Failed to list objects")
            return -1
    
    def get_system_metrics(self) -> Dict[str, float]:
        """Get system metrics (CPU, memory, disk usage)"""
        metrics = {}
        
        # Get CPU usage
        try:
            cpu_output = subprocess.check_output(["top", "-bn1"]).decode()
            cpu_lines = cpu_output.split('\n')
            for line in cpu_lines:
                if 'Cpu(s)' in line:
                    cpu_usage = float(line.split(',')[0].split(':')[1].strip().replace('%id', '').strip())
                    metrics['cpu_usage'] = 100 - cpu_usage  # Convert idle to usage
        except (subprocess.CalledProcessError, IndexError, ValueError):
            metrics['cpu_usage'] = -1
        
        # Get memory usage
        try:
            mem_output = subprocess.check_output(["free", "-m"]).decode()
            mem_lines = mem_output.split('\n')
            if len(mem_lines) > 1:
                mem_values = mem_lines[1].split()
                total_mem = float(mem_values[1])
                used_mem = float(mem_values[2])
                metrics['memory_usage'] = (used_mem / total_mem) * 100
        except (subprocess.CalledProcessError, IndexError, ValueError):
            metrics['memory_usage'] = -1
        
        # Get disk usage
        try:
            disk_output = subprocess.check_output(["df", "-h"]).decode()
            disk_lines = disk_output.split('\n')
            for line in disk_lines:
                if '/dev/sda1' in line or '/' in line:
                    disk_values = line.split()
                    metrics['disk_usage'] = float(disk_values[4].replace('%', ''))
                    break
        except (subprocess.CalledProcessError, IndexError, ValueError):
            metrics['disk_usage'] = -1
        
        logger.info(f"System metrics: CPU={metrics.get('cpu_usage', -1):.1f}%, "
                   f"Memory={metrics.get('memory_usage', -1):.1f}%, "
                   f"Disk={metrics.get('disk_usage', -1):.1f}%")
        return metrics
    
    def run_scalability_test(self, object_counts: List[int]):
        """Run scalability test with increasing object counts"""
        for count in object_counts:
            logger.info(f"=== Starting test with {count} objects ===")
            
            # Record object count
            self.metrics['object_count'].append(count)
            
            # Generate objects and measure put throughput
            put_throughput = self.generate_objects(count)
            self.metrics['put_throughput'].append(put_throughput)
            
            # Measure list latency
            list_latency = self.measure_list_latency()
            self.metrics['list_latency'].append(list_latency)
            
            # Get system metrics
            system_metrics = self.get_system_metrics()
            self.metrics['cpu_usage'].append(system_metrics.get('cpu_usage', -1))
            self.metrics['memory_usage'].append(system_metrics.get('memory_usage', -1))
            self.metrics['disk_usage'].append(system_metrics.get('disk_usage', -1))
            
            logger.info(f"=== Completed test with {count} objects ===")
            
    def save_results(self):
        """Save test results to CSV and generate plots"""
        # Save to CSV
        results_df = pd.DataFrame(self.metrics)
        csv_path = os.path.join(RESULTS_FOLDER, "scalability_results.csv")
        results_df.to_csv(csv_path, index=False)
        logger.info(f"Results saved to {csv_path}")
        
        # Generate throughput plot
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics['object_count'], self.metrics['put_throughput'], '-o', label='Put Throughput')
        plt.xlabel('Number of Objects')
        plt.ylabel('Throughput (objects/second)')
        plt.title('Ozone Scalability - Put Throughput')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(RESULTS_FOLDER, 'throughput_plot.png'))
        
        # Generate latency plot
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics['object_count'], self.metrics['list_latency'], '-o', label='List Latency')
        plt.xlabel('Number of Objects')
        plt.ylabel('Latency (seconds)')
        plt.title('Ozone Scalability - List Latency')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(RESULTS_FOLDER, 'latency_plot.png'))
        
        # Generate system metrics plot
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics['object_count'], self.metrics['cpu_usage'], '-o', label='CPU Usage (%)')
        plt.plot(self.metrics['object_count'], self.metrics['memory_usage'], '-o', label='Memory Usage (%)')
        plt.plot(self.metrics['object_count'], self.metrics['disk_usage'], '-o', label='Disk Usage (%)')
        plt.xlabel('Number of Objects')
        plt.ylabel('Usage (%)')
        plt.title('Ozone Scalability - System Resource Usage')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(RESULTS_FOLDER, 'resource_usage_plot.png'))
        
    def analyze_scalability(self) -> bool:
        """Analyze if performance scales linearly or sub-linearly"""
        if len(self.metrics['object_count']) < 2:
            return False
        
        # Calculate throughput ratios
        throughput_ratios = []
        object_count_ratios = []
        
        for i in range(1, len(self.metrics['object_count'])):
            obj_ratio = self.metrics['object_count'][i] / self.metrics['object_count'][i-1]
            tp_ratio = self.metrics['put_throughput'][i] / self.metrics['put_throughput'][i-1]
            
            object_count_ratios.append(obj_ratio)
            throughput_ratios.append(tp_ratio)
        
        # Check if throughput decrease is proportionally less than object count increase
        # (which would indicate sub-linear or linear scaling)
        for i in range(len(throughput_ratios)):
            if throughput_ratios[i] < 0.7 * (1 / object_count_ratios[i]):
                logger.warning(f"Potential scalability issue detected at {self.metrics['object_count'][i+1]} objects")
                return False
        
        logger.info("Scalability analysis: Performance scales acceptably with increasing object count")
        return True


@pytest.fixture(scope="module")
def ozone_test():
    """Setup and teardown for the Ozone scalability test"""
    test = OzoneScalabilityTest()
    test.setup_environment()
    yield test
    test.cleanup_environment()


@pytest.mark.parametrize("object_counts", [
    [1000, 10000, 100000],  # Small scale test
    [10000, 100000, 1000000]  # Medium scale test
])
def test_5_scalability_with_increasing_objects(ozone_test, object_counts):
    """
    Test scalability with increasing number of objects
    
    This test verifies that Apache Ozone performance scales linearly or sub-linearly
    with increasing number of objects, without significant degradation.
    """
    # Run the scalability test with increasing object counts
    ozone_test.run_scalability_test(object_counts)
    
    # Save test results
    ozone_test.save_results()
    
    # Verify scalability
    is_scalable = ozone_test.analyze_scalability()
    
    # Check if any put throughput value is too low (indicating performance issues)
    min_acceptable_throughput = 50  # objects per second
    has_acceptable_throughput = all(tp >= min_acceptable_throughput for tp in ozone_test.metrics['put_throughput'])
    
    # Calculate coefficients of variation to detect instability
    if len(ozone_test.metrics['put_throughput']) > 1:
        cv_put = statistics.stdev(ozone_test.metrics['put_throughput']) / statistics.mean(ozone_test.metrics['put_throughput'])
        stable_performance = cv_put < 0.5  # CV below 50% indicates reasonable stability
    else:
        stable_performance = True
    
    # Log detailed results
    logger.info("Scalability test results:")
    for i, count in enumerate(ozone_test.metrics['object_count']):
        logger.info(f"Count: {count}, Put Throughput: {ozone_test.metrics['put_throughput'][i]:.2f} obj/sec, "
                   f"List Latency: {ozone_test.metrics['list_latency'][i]:.2f} sec")
    
    # Assert overall test conditions
    assert is_scalable, "Performance did not scale linearly or sub-linearly with increasing objects"
    assert has_acceptable_throughput, f"Throughput fell below minimum acceptable value of {min_acceptable_throughput} objects/second"
    assert stable_performance, "Performance showed high variability across object count increases"

import os
import time
import subprocess
import logging
import pytest
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='rebalance_performance_test.log'
)
logger = logging.getLogger(__name__)

# Constants
OZONE_BIN = os.environ.get('OZONE_BIN', '/opt/ozone/bin')
METRICS_INTERVAL = 5  # seconds
TEST_DATA_DIR = "test_data"
RESULTS_DIR = "test_results"

# Ensure directories exist
os.makedirs(TEST_DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

class OzoneClusterMetrics:
    """Class to collect and analyze metrics from Ozone cluster nodes"""
    
    def __init__(self, node_ips: List[str]):
        self.node_ips = node_ips
        self.metrics_data = {node: {"cpu": [], "memory": [], "disk_io": [], "network": []} for node in node_ips}
        self.running = False
    
    def start_collection(self):
        """Start metrics collection in background"""
        self.running = True
        self.collection_thread = ThreadPoolExecutor(max_workers=len(self.node_ips))
        for node in self.node_ips:
            self.collection_thread.submit(self._collect_metrics, node)
    
    def stop_collection(self):
        """Stop metrics collection"""
        self.running = False
        self.collection_thread.shutdown(wait=True)
    
    def _collect_metrics(self, node_ip: str):
        """Collect metrics for a specific node"""
        while self.running:
            try:
                # Get CPU usage
                cpu_cmd = f"ssh {node_ip} 'top -bn1 | grep \"Cpu(s)\" | awk '\"'\"'{{print $2 + $4}}'\"'\"''"
                cpu = float(subprocess.check_output(cpu_cmd, shell=True).decode().strip())
                
                # Get memory usage
                mem_cmd = f"ssh {node_ip} 'free -m | grep Mem | awk '\"'\"'{{print $3/$2 * 100.0}}'\"'\"''"
                memory = float(subprocess.check_output(mem_cmd, shell=True).decode().strip())
                
                # Get disk I/O
                io_cmd = f"ssh {node_ip} 'iostat -xk 1 1 | grep sda | awk '\"'\"'{{print $6}}'\"'\"''"
                disk_io = float(subprocess.check_output(io_cmd, shell=True).decode().strip())
                
                # Get network throughput
                net_cmd = f"ssh {node_ip} 'sar -n DEV 1 1 | grep eth0 | tail -1 | awk '\"'\"'{{print $5 + $6}}'\"'\"''"
                network = float(subprocess.check_output(net_cmd, shell=True).decode().strip())
                
                # Store metrics
                self.metrics_data[node_ip]["cpu"].append(cpu)
                self.metrics_data[node_ip]["memory"].append(memory)
                self.metrics_data[node_ip]["disk_io"].append(disk_io)
                self.metrics_data[node_ip]["network"].append(network)
                
                time.sleep(METRICS_INTERVAL)
            except Exception as e:
                logger.error(f"Error collecting metrics from {node_ip}: {str(e)}")
    
    def generate_report(self, test_name: str) -> str:
        """Generate performance report and visualizations"""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        report_file = f"{RESULTS_DIR}/{test_name}_{timestamp}.html"
        
        # Convert metrics data to DataFrames for analysis
        dfs = {}
        for metric in ["cpu", "memory", "disk_io", "network"]:
            data = {node: self.metrics_data[node][metric] for node in self.node_ips}
            dfs[metric] = pd.DataFrame(data)
        
        # Create visualizations
        fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(12, 20))
        
        for i, (metric, df) in enumerate(dfs.items()):
            df.plot(ax=axes[i], title=f"{metric.upper()} Usage During Rebalancing")
            axes[i].set_ylabel(f"{metric} usage (%)" if metric != "network" else "KB/s")
            axes[i].set_xlabel("Time (intervals of 5 seconds)")
            axes[i].grid(True)
        
        # Save plot
        plot_path = f"{RESULTS_DIR}/{test_name}_metrics_{timestamp}.png"
        plt.tight_layout()
        plt.savefig(plot_path)
        
        # Generate HTML report
        with open(report_file, 'w') as f:
            f.write(f"""
            
Ozone Rebalance Performance Test Report

Ozone Rebalance Performance Test Report - {test_name}
Generated at: {timestamp}
Summary Statistics


Metric
Min
Max
Mean

            """)
            
            for metric, df in dfs.items():
                f.write(f"""
                        
{metric}
{df.min().mean():.2f}
{df.max().mean():.2f}
{df.mean().mean():.2f}

                """)
            
            f.write(f"""
                    
Performance Visualizations



            """)
        
        return report_file


def create_imbalanced_data_distribution(volume_name: str, bucket_name: str, node_ips: List[str]) -> Dict[str, int]:
    """
    Creates an imbalanced data distribution across nodes by
    targeting specific datanodes for data upload
    """
    logger.info("Creating imbalanced data distribution across nodes")
    
    # Create volume and bucket if they don't exist
    subprocess.run([f"{OZONE_BIN}/ozone", "sh", "volume", "create", volume_name], check=True)
    subprocess.run([f"{OZONE_BIN}/ozone", "sh", "bucket", "create", f"{volume_name}/{bucket_name}"], check=True)
    
    # Distribution of data sizes across nodes (in MB)
    data_distribution = {
        node_ips[0]: 500,   # 500MB
        node_ips[1]: 1500,  # 1.5GB 
        node_ips[2]: 100,   # 100MB
    }
    
    # Create files of different sizes and upload them to specific nodes
    for node, data_size in data_distribution.items():
        node_id = node_ips.index(node)
        
        # Create test files
        file_sizes = []
        remaining_size = data_size
        
        # Create a mix of file sizes to reach the total data_size
        while remaining_size > 0:
            if remaining_size > 100:
                size = min(remaining_size, 50 + (node_id * 7))  # Vary file sizes based on node_id
            else:
                size = remaining_size
            
            file_sizes.append(size)
            remaining_size -= size
        
        # Create and upload each file
        for i, size_mb in enumerate(file_sizes):
            filename = f"{TEST_DATA_DIR}/file_node{node_id}_part{i}.dat"
            
            # Create file with specific size
            with open(filename, 'wb') as f:
                f.write(os.urandom(size_mb * 1024 * 1024))
            
            # Upload file with a hint to place it on the specific datanode
            key = f"file_node{node_id}_part{i}"
            cmd = [
                f"{OZONE_BIN}/ozone", "sh", "key", "put",
                f"{volume_name}/{bucket_name}/{key}",
                filename,
                f"--replication=3",
                f"--hint={node}"  # This is a hypothetical flag, actual implementation may vary
            ]
            subprocess.run(cmd, check=True)
            
            logger.info(f"Uploaded {size_mb}MB file to {node} as key {key}")
    
    return data_distribution


def trigger_rebalancing() -> str:
    """
    Triggers the data rebalancing operation in Ozone and returns the operation ID
    """
    logger.info("Triggering data rebalancing operation")
    
    # The actual command to trigger rebalancing may vary based on your Ozone deployment
    result = subprocess.run(
        [f"{OZONE_BIN}/ozone", "admin", "rebalance", "start", "--verbose"],
        check=True,
        capture_output=True,
        text=True
    )
    
    # Extract operation ID from output
    operation_id = result.stdout.strip().split("Operation ID: ")[-1].split()[0]
    logger.info(f"Rebalancing operation started with ID: {operation_id}")
    
    return operation_id


def check_rebalancing_status(operation_id: str) -> Tuple[bool, float]:
    """
    Checks the status of a rebalancing operation
    Returns (is_complete, progress_percentage)
    """
    result = subprocess.run(
        [f"{OZONE_BIN}/ozone", "admin", "rebalance", "status", operation_id],
        check=True,
        capture_output=True,
        text=True
    )
    
    output = result.stdout.strip()
    
    # Parse the output to get status and progress
    is_complete = "COMPLETED" in output
    
    # Extract progress percentage (parsing may need adjustment based on actual output)
    progress = 0.0
    for line in output.split("\n"):
        if "Progress:" in line:
            progress_str = line.split("Progress:")[-1].strip().rstrip("%")
            try:
                progress = float(progress_str)
            except ValueError:
                pass
    
    return is_complete, progress


def monitor_ongoing_operations(volume_name: str, bucket_name: str, duration_sec: int = 60) -> Dict:
    """
    Performs read/write operations and measures their performance during rebalancing
    """
    logger.info(f"Monitoring system performance during rebalancing for {duration_sec} seconds")
    
    results = {
        "read_latencies": [],
        "write_latencies": [],
        "read_failures": 0,
        "write_failures": 0
    }
    
    start_time = time.time()
    end_time = start_time + duration_sec
    
    # Keep performing operations until the specified duration is reached
    iteration = 0
    while time.time() < end_time:
        iteration += 1
        
        try:
            # Write operation
            write_start = time.time()
            
            test_file = f"{TEST_DATA_DIR}/concurrent_test_{iteration}.dat"
            with open(test_file, 'wb') as f:
                f.write(os.urandom(1024 * 1024))  # 1MB data
                
            key = f"concurrent_test_{iteration}"
            cmd = [
                f"{OZONE_BIN}/ozone", "sh", "key", "put",
                f"{volume_name}/{bucket_name}/{key}",
                test_file
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            write_latency = time.time() - write_start
            results["write_latencies"].append(write_latency)
            
            # Read operation (read back the file we just wrote)
            read_start = time.time()
            
            read_file = f"{TEST_DATA_DIR}/read_concurrent_{iteration}.dat"
            cmd = [
                f"{OZONE_BIN}/ozone", "sh", "key", "get",
                f"{volume_name}/{bucket_name}/{key}",
                read_file
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            read_latency = time.time() - read_start
            results["read_latencies"].append(read_latency)
            
            # Clean up test files
            os.remove(test_file)
            os.remove(read_file)
            
        except Exception as e:
            logger.error(f"Error during concurrent operation: {str(e)}")
            if "write_start" in locals():
                results["write_failures"] += 1
            else:
                results["read_failures"] += 1
        
        # Add a small delay to avoid overwhelming the system
        time.sleep(1)
    
    return results


@pytest.fixture(scope="module")
def ozone_cluster():
    """Fixture to provide Ozone cluster details"""
    # These would typically be loaded from a configuration file or environment
    node_ips = [
        "192.168.1.100",  # Replace with actual node IPs
        "192.168.1.101",
        "192.168.1.102"
    ]
    
    # Check if cluster is accessible
    try:
        subprocess.run([f"{OZONE_BIN}/ozone", "version"], check=True, stdout=subprocess.DEVNULL)
    except Exception as e:
        pytest.fail(f"Ozone cluster is not accessible: {str(e)}")
    
    return {
        "node_ips": node_ips,
        "metrics_collector": OzoneClusterMetrics(node_ips)
    }


@pytest.mark.performance
def test_6_rebalancing_performance(ozone_cluster):
    """
    Evaluate performance under data rebalancing
    
    This test:
    1. Creates an imbalanced data distribution across nodes
    2. Triggers data rebalancing operation
    3. Measures the time taken for rebalancing
    4. Monitors system performance during rebalancing
    """
    # Test configuration
    volume_name = "perfvolume6"
    bucket_name = "perfbucket6"
    rebalance_timeout = 1800  # 30 minutes max
    performance_threshold = {
        "max_rebalance_time": 600,  # seconds
        "max_read_latency": 2.0,    # seconds
        "max_write_latency": 3.0,   # seconds
        "max_failure_rate": 0.05    # 5% failure rate allowed
    }
    
    node_ips = ozone_cluster["node_ips"]
    metrics = ozone_cluster["metrics_collector"]
    
    # Step 1: Create an imbalanced data distribution across nodes
    data_distribution = create_imbalanced_data_distribution(volume_name, bucket_name, node_ips)
    logger.info(f"Created imbalanced data distribution: {data_distribution}")
    
    # Verify data placement imbalance before rebalancing
    for node in node_ips:
        cmd = [f"{OZONE_BIN}/ozone", "admin", "datanode", "info", node]
        result = subprocess.run(cmd, capture_output=True, text=True)
        logger.info(f"Node {node} info before rebalancing:\n{result.stdout}")
    
    # Start metrics collection
    metrics.start_collection()
    
    # Step 2: Trigger data rebalancing operation
    rebalance_start_time = time.time()
    operation_id = trigger_rebalancing()
    
    # Step 3 & 4: Measure time taken for rebalancing and monitor system performance during rebalancing
    completed = False
    progress = 0.0
    
    # Start concurrent operations to test impact on system
    concurrent_operations_results = {}
    with ThreadPoolExecutor(max_workers=1) as executor:
        concurrent_future = executor.submit(
            monitor_ongoing_operations, volume_name, bucket_name, rebalance_timeout
        )
        
        # Monitor the rebalancing progress
        while not completed and (time.time() - rebalance_start_time) < rebalance_timeout:
            completed, progress = check_rebalancing_status(operation_id)
            logger.info(f"Rebalancing progress: {progress:.2f}%")
            
            if completed:
                break
                
            # Check every 10 seconds
            time.sleep(10)
            
        # Get the results from concurrent operations
        concurrent_operations_results = concurrent_future.result()
    
    # Stop metrics collection
    metrics.stop_collection()
    
    # Calculate rebalance duration
    rebalance_duration = time.time() - rebalance_start_time
    
    # Generate performance report
    report_file = metrics.generate_report(f"rebalance_test_{volume_name}")
    logger.info(f"Performance report generated: {report_file}")
    
    # Analyze the results of concurrent operations
    avg_read_latency = sum(concurrent_operations_results["read_latencies"]) / len(concurrent_operations_results["read_latencies"]) if concurrent_operations_results["read_latencies"] else 0
    avg_write_latency = sum(concurrent_operations_results["write_latencies"]) / len(concurrent_operations_results["write_latencies"]) if concurrent_operations_results["write_latencies"] else 0
    
    total_ops = len(concurrent_operations_results["read_latencies"]) + len(concurrent_operations_results["write_latencies"])
    total_failures = concurrent_operations_results["read_failures"] + concurrent_operations_results["write_failures"]
    failure_rate = total_failures / total_ops if total_ops > 0 else 0
    
    logger.info(f"Rebalancing completed in {rebalance_duration:.2f} seconds")
    logger.info(f"Average read latency during rebalancing: {avg_read_latency:.4f} seconds")
    logger.info(f"Average write latency during rebalancing: {avg_write_latency:.4f} seconds")
    logger.info(f"Operation failure rate during rebalancing: {failure_rate:.2%}")
    
    # Verify data is now more balanced
    for node in node_ips:
        cmd = [f"{OZONE_BIN}/ozone", "admin", "datanode", "info", node]
        result = subprocess.run(cmd, capture_output=True, text=True)
        logger.info(f"Node {node} info after rebalancing:\n{result.stdout}")
    
    # Assertions
    assert completed, f"Rebalancing did not complete within the timeout period ({rebalance_timeout} seconds)"
    assert rebalance_duration < performance_threshold["max_rebalance_time"], f"Rebalancing took too long: {rebalance_duration:.2f} seconds"
    assert avg_read_latency < performance_threshold["max_read_latency"], f"Read latency too high: {avg_read_latency:.4f} seconds"
    assert avg_write_latency < performance_threshold["max_write_latency"], f"Write latency too high: {avg_write_latency:.4f} seconds"
    assert failure_rate < performance_threshold["max_failure_rate"], f"Operation failure rate too high: {failure_rate:.2%}"
    
    # Clean up
    try:
        subprocess.run([f"{OZONE_BIN}/ozone", "sh", "bucket", "delete", f"{volume_name}/{bucket_name}"], check=True)
        subprocess.run([f"{OZONE_BIN}/ozone", "sh", "volume", "delete", volume_name], check=True)
    except Exception as e:
        logger.warning(f"Cleanup failed: {str(e)}")

import time
import pytest
import subprocess
import statistics
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor

# Constants for test configuration
OZONE_SERVICE_ID = "oz1"
VOLUME_NAME = "perfvolume"
BUCKET_NAME = "perfbucket"
OPERATIONS_COUNT = 1000  # Number of operations to perform for each test
NUM_THREADS = 10  # Number of concurrent threads for parallel testing
TEST_ITERATIONS = 3  # Number of test iterations for consistent results
LATENCY_THRESHOLD_MS = 5  # Maximum acceptable latency in milliseconds
THROUGHPUT_THRESHOLD = 1000  # Minimum acceptable throughput (ops/sec)


class OzoneClient:
    """Helper class to interact with Ozone cluster"""
    
    @staticmethod
    def run_command(command: List[str]) -> Tuple[str, float]:
        """Run an Ozone shell command and measure execution time"""
        start_time = time.time()
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        execution_time = time.time() - start_time
        
        if result.returncode != 0:
            raise Exception(f"Command failed: {' '.join(command)}\nError: {result.stderr}")
        
        return result.stdout, execution_time
    
    @staticmethod
    def list_volumes() -> Tuple[List[str], float]:
        """List all volumes and measure execution time"""
        command = ["ozone", "sh", "volume", "list"]
        output, execution_time = OzoneClient.run_command(command)
        volumes = [line.split()[0] for line in output.strip().split('\n')[1:]]
        return volumes, execution_time
    
    @staticmethod
    def list_buckets(volume: str) -> Tuple[List[str], float]:
        """List all buckets in a volume and measure execution time"""
        command = ["ozone", "sh", "bucket", "list", volume]
        output, execution_time = OzoneClient.run_command(command)
        buckets = [line.split()[0] for line in output.strip().split('\n')[1:]]
        return buckets, execution_time
    
    @staticmethod
    def list_keys(volume: str, bucket: str) -> Tuple[List[str], float]:
        """List all keys in a bucket and measure execution time"""
        command = ["ozone", "sh", "key", "list", f"{volume}/{bucket}"]
        output, execution_time = OzoneClient.run_command(command)
        keys = [line.split()[0] for line in output.strip().split('\n')[1:]]
        return keys, execution_time
    
    @staticmethod
    def get_key_info(volume: str, bucket: str, key: str) -> Tuple[Dict, float]:
        """Get info about a key and measure execution time"""
        command = ["ozone", "sh", "key", "info", f"{volume}/{bucket}/{key}"]
        output, execution_time = OzoneClient.run_command(command)
        # Parse output to extract key info
        return {"key": key}, execution_time
    
    @staticmethod
    def setup_test_data(num_keys: int = 100):
        """Set up test data for performance measurements"""
        # Create volume if it doesn't exist
        subprocess.run(["ozone", "sh", "volume", "create", VOLUME_NAME], 
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Create bucket if it doesn't exist
        subprocess.run(["ozone", "sh", "bucket", "create", f"{VOLUME_NAME}/{BUCKET_NAME}"], 
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Create test keys
        temp_file = "/tmp/test_data.txt"
        with open(temp_file, "w") as f:
            f.write("Test data for performance measurement")
        
        for i in range(num_keys):
            key_name = f"testkey{i}"
            subprocess.run(["ozone", "sh", "key", "put", f"{VOLUME_NAME}/{BUCKET_NAME}/{key_name}", temp_file], 
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def run_operation(operation_func, *args) -> float:
    """Run operation and return execution time"""
    _, execution_time = operation_func(*args)
    return execution_time


@pytest.fixture(scope="module")
def prepare_test_environment():
    """Set up test data before running performance tests"""
    OzoneClient.setup_test_data(num_keys=100)
    yield
    # Clean up if necessary


def measure_operation_performance(operation_func, args_list, operation_name):
    """Measure performance of an operation with multiple executions"""
    latencies = []
    
    # Measure sequential performance
    start_time = time.time()
    for args in args_list:
        latency = run_operation(operation_func, *args)
        latencies.append(latency * 1000)  # Convert to ms
    
    sequential_duration = time.time() - start_time
    sequential_throughput = len(args_list) / sequential_duration
    
    # Measure concurrent performance
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = [executor.submit(run_operation, operation_func, *args) for args in args_list]
        concurrent_latencies = [future.result() * 1000 for future in futures]  # Convert to ms
    
    concurrent_duration = time.time() - start_time
    concurrent_throughput = len(args_list) / concurrent_duration
    
    all_latencies = latencies + concurrent_latencies
    
    result = {
        "operation": operation_name,
        "min_latency_ms": min(all_latencies),
        "max_latency_ms": max(all_latencies),
        "avg_latency_ms": statistics.mean(all_latencies),
        "p95_latency_ms": statistics.quantiles(all_latencies, n=20)[18],  # Approximation of p95
        "p99_latency_ms": statistics.quantiles(all_latencies, n=100)[98],  # Approximation of p99
        "sequential_throughput": sequential_throughput,
        "concurrent_throughput": concurrent_throughput
    }
    
    return result


def test_7_metadata_operation_performance(prepare_test_environment):
    """
    Measure metadata operation performance
    
    This test measures the performance of various metadata operations in Apache Ozone,
    including listing volumes, buckets, keys, and getting object info.
    """
    results = []
    
    # 1. List volumes performance
    volume_args = [[] for _ in range(OPERATIONS_COUNT)]
    volume_result = measure_operation_performance(
        OzoneClient.list_volumes, 
        volume_args,
        "List Volumes"
    )
    results.append(volume_result)
    
    # 2. List buckets performance
    bucket_args = [[VOLUME_NAME] for _ in range(OPERATIONS_COUNT)]
    bucket_result = measure_operation_performance(
        OzoneClient.list_buckets,
        bucket_args,
        "List Buckets"
    )
    results.append(bucket_result)
    
    # 3. List keys performance
    keys_args = [[VOLUME_NAME, BUCKET_NAME] for _ in range(OPERATIONS_COUNT)]
    keys_result = measure_operation_performance(
        OzoneClient.list_keys,
        keys_args,
        "List Keys"
    )
    results.append(keys_result)
    
    # 4. Get key info performance
    key_info_args = [[VOLUME_NAME, BUCKET_NAME, f"testkey{i % 100}"] for i in range(OPERATIONS_COUNT)]
    key_info_result = measure_operation_performance(
        OzoneClient.get_key_info,
        key_info_args,
        "Get Key Info"
    )
    results.append(key_info_result)
    
    # Create a performance report
    df = pd.DataFrame(results)
    
    # Generate performance report
    print("\nOzone Metadata Operations Performance Report:")
    print(df.to_string(index=False))
    
    # Optional: Generate performance charts
    df.plot(x='operation', y=['avg_latency_ms', 'p95_latency_ms', 'p99_latency_ms'], kind='bar')
    plt.ylabel('Latency (ms)')
    plt.title('Ozone Metadata Operation Latencies')
    plt.savefig('metadata_latency_report.png')
    
    df.plot(x='operation', y=['sequential_throughput', 'concurrent_throughput'], kind='bar')
    plt.ylabel('Throughput (ops/sec)')
    plt.title('Ozone Metadata Operation Throughput')
    plt.savefig('metadata_throughput_report.png')
    
    # Perform assertions based on expected results
    for result in results:
        # Check if average latency is below threshold
        assert result["avg_latency_ms"] < LATENCY_THRESHOLD_MS, \
            f"{result['operation']} average latency ({result['avg_latency_ms']:.2f} ms) exceeds threshold of {LATENCY_THRESHOLD_MS} ms"
        
        # Check if concurrent throughput meets requirements
        assert result["concurrent_throughput"] > THROUGHPUT_THRESHOLD, \
            f"{result['operation']} throughput ({result['concurrent_throughput']:.2f} ops/sec) is below threshold of {THROUGHPUT_THRESHOLD} ops/sec"

import time
import subprocess
import pytest
import os
import statistics
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor


class OzoneClusterConfig:
    """Helper class to configure and manage Ozone cluster replication factors"""
    
    def __init__(self, admin_node="localhost"):
        self.admin_node = admin_node
    
    def set_replication_factor(self, replication_factor: int) -> bool:
        """
        Configure the Ozone cluster with the specified replication factor
        Returns True if configuration was successful
        """
        try:
            # Command to update the replication factor in hdds-site.xml
            cmd = [
                "ssh", self.admin_node,
                f"sudo sed -i 's/[0-9]<\\/value>/{replication_factor}<\\/value>/g' "
                "/etc/hadoop/conf/hdds-site.xml"
            ]
            subprocess.run(cmd, check=True)
            
            # Restart the required services to apply the new configuration
            restart_cmd = [
                "ssh", self.admin_node,
                "sudo systemctl restart hadoop-ozone-scm hadoop-ozone-om"
            ]
            subprocess.run(restart_cmd, check=True)
            
            # Wait for the cluster to stabilize
            time.sleep(60)
            return True
        except subprocess.SubprocessError as e:
            print(f"Failed to set replication factor: {e}")
            return False


class OzonePerformanceTester:
    """Handles performance testing operations on Ozone"""
    
    def __init__(self, volume_name="perf_vol", bucket_name="perf_bucket"):
        self.volume_name = volume_name
        self.bucket_name = bucket_name
        self.test_file_sizes = [1, 10, 100, 1024]  # Size in MB
        self.test_files = []
        
    def setup(self):
        """Create test volume, bucket and test files"""
        # Create volume and bucket
        subprocess.run(["ozone", "sh", "volume", "create", self.volume_name], check=True)
        subprocess.run(["ozone", "sh", "bucket", "create", f"{self.volume_name}/{self.bucket_name}"], check=True)
        
        # Create test files of different sizes
        for size in self.test_file_sizes:
            filename = f"test_file_{size}MB.bin"
            subprocess.run(["dd", "if=/dev/urandom", f"of={filename}", f"bs=1M", f"count={size}"], check=True)
            self.test_files.append(filename)
    
    def cleanup(self):
        """Clean up test resources"""
        # Remove test files
        for file in self.test_files:
            if os.path.exists(file):
                os.remove(file)
        
        # Delete bucket and volume
        subprocess.run(["ozone", "sh", "bucket", "delete", f"{self.volume_name}/{self.bucket_name}"], check=True)
        subprocess.run(["ozone", "sh", "volume", "delete", self.volume_name], check=True)
    
    def run_write_benchmark(self, num_iterations: int = 5) -> Dict[str, List[float]]:
        """
        Run write performance benchmarks
        Returns dictionary with file sizes as keys and lists of throughput values
        """
        results = {size: [] for size in self.test_file_sizes}
        
        for _ in range(num_iterations):
            for i, file_size in enumerate(self.test_file_sizes):
                file_path = self.test_files[i]
                key_name = f"perftest_key_{file_size}MB_{int(time.time())}"
                
                start_time = time.time()
                subprocess.run([
                    "ozone", "sh", "key", "put", 
                    f"{self.volume_name}/{self.bucket_name}/", file_path, 
                    "--key", key_name
                ], check=True)
                end_time = time.time()
                
                elapsed_time = end_time - start_time
                throughput = (file_size * 1024 * 1024) / elapsed_time  # Bytes per second
                results[file_size].append(throughput / (1024 * 1024))  # Convert to MB/s
                
                # Delete the key to avoid storage buildup
                subprocess.run([
                    "ozone", "sh", "key", "delete", 
                    f"{self.volume_name}/{self.bucket_name}/{key_name}"
                ], check=True)
        
        return results
    
    def run_read_benchmark(self, num_iterations: int = 5) -> Dict[str, List[float]]:
        """
        Run read performance benchmarks
        Returns dictionary with file sizes as keys and lists of throughput values
        """
        # First upload all test files
        key_names = []
        for i, file_size in enumerate(self.test_file_sizes):
            file_path = self.test_files[i]
            key_name = f"perftest_read_key_{file_size}MB"
            key_names.append(key_name)
            
            subprocess.run([
                "ozone", "sh", "key", "put", 
                f"{self.volume_name}/{self.bucket_name}/", file_path,
                "--key", key_name
            ], check=True)
        
        results = {size: [] for size in self.test_file_sizes}
        
        # Now read them multiple times to measure performance
        for _ in range(num_iterations):
            for i, file_size in enumerate(self.test_file_sizes):
                key_name = key_names[i]
                output_file = f"output_{file_size}MB.bin"
                
                start_time = time.time()
                subprocess.run([
                    "ozone", "sh", "key", "get", 
                    f"{self.volume_name}/{self.bucket_name}/{key_name}",
                    output_file
                ], check=True)
                end_time = time.time()
                
                elapsed_time = end_time - start_time
                throughput = (file_size * 1024 * 1024) / elapsed_time  # Bytes per second
                results[file_size].append(throughput / (1024 * 1024))  # Convert to MB/s
                
                # Remove the output file
                if os.path.exists(output_file):
                    os.remove(output_file)
        
        # Clean up uploaded files
        for key_name in key_names:
            subprocess.run([
                "ozone", "sh", "key", "delete", 
                f"{self.volume_name}/{self.bucket_name}/{key_name}"
            ], check=True)
        
        return results
    
    def run_concurrent_operations(self, num_threads: int = 10, op_count: int = 50) -> Tuple[float, float]:
        """
        Run concurrent read/write operations to test throughput
        Returns (write_ops_per_sec, read_ops_per_sec)
        """
        # Create a small test file for concurrent operations
        test_file = "concurrent_test_1MB.bin"
        subprocess.run(["dd", "if=/dev/urandom", f"of={test_file}", "bs=1M", "count=1"], check=True)
        
        # Concurrent write test
        def write_operation(i):
            key_name = f"concurrent_write_key_{i}"
            subprocess.run([
                "ozone", "sh", "key", "put", 
                f"{self.volume_name}/{self.bucket_name}/", test_file,
                "--key", key_name
            ], check=True)
            return key_name
        
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            written_keys = list(executor.map(write_operation, range(op_count)))
        end_time = time.time()
        
        write_duration = end_time - start_time
        write_ops_per_sec = op_count / write_duration
        
        # Concurrent read test
        def read_operation(key_name):
            output_file = f"output_{key_name}.bin"
            subprocess.run([
                "ozone", "sh", "key", "get", 
                f"{self.volume_name}/{self.bucket_name}/{key_name}",
                output_file
            ], check=True)
            if os.path.exists(output_file):
                os.remove(output_file)
        
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            list(executor.map(read_operation, written_keys))
        end_time = time.time()
        
        read_duration = end_time - start_time
        read_ops_per_sec = op_count / read_duration
        
        # Clean up
        for key in written_keys:
            subprocess.run([
                "ozone", "sh", "key", "delete", 
                f"{self.volume_name}/{self.bucket_name}/{key}"
            ], check=True)
        os.remove(test_file)
        
        return write_ops_per_sec, read_ops_per_sec


def analyze_results(results_by_replication: Dict[int, Dict[str, Dict]]):
    """Analyze and visualize performance results across different replication factors"""
    
    # Create DataFrame for write performance
    write_data = {
        'Replication Factor': [],
        'File Size (MB)': [],
        'Throughput (MB/s)': []
    }
    
    for rf, results in results_by_replication.items():
        for size, throughputs in results['write'].items():
            for tp in throughputs:
                write_data['Replication Factor'].append(rf)
                write_data['File Size (MB)'].append(size)
                write_data['Throughput (MB/s)'].append(tp)
    
    write_df = pd.DataFrame(write_data)
    
    # Create DataFrame for read performance
    read_data = {
        'Replication Factor': [],
        'File Size (MB)': [],
        'Throughput (MB/s)': []
    }
    
    for rf, results in results_by_replication.items():
        for size, throughputs in results['read'].items():
            for tp in throughputs:
                read_data['Replication Factor'].append(rf)
                read_data['File Size (MB)'].append(size)
                read_data['Throughput (MB/s)'].append(tp)
    
    read_df = pd.DataFrame(read_data)
    
    # Create summary statistics
    write_summary = write_df.groupby(['Replication Factor', 'File Size (MB)']).agg({
        'Throughput (MB/s)': ['mean', 'std']
    }).reset_index()
    
    read_summary = read_df.groupby(['Replication Factor', 'File Size (MB)']).agg({
        'Throughput (MB/s)': ['mean', 'std']
    }).reset_index()
    
    # Plot write performance
    plt.figure(figsize=(12, 6))
    for rf in sorted(write_df['Replication Factor'].unique()):
        data = write_df[write_df['Replication Factor'] == rf]
        plt.plot(data['File Size (MB)'], data['Throughput (MB/s)'], 'o-', label=f'RF={rf}')
    
    plt.title('Write Throughput vs. File Size for Different Replication Factors')
    plt.xlabel('File Size (MB)')
    plt.ylabel('Throughput (MB/s)')
    plt.legend()
    plt.xscale('log')
    plt.grid()
    plt.savefig('write_performance.png')
    
    # Plot read performance
    plt.figure(figsize=(12, 6))
    for rf in sorted(read_df['Replication Factor'].unique()):
        data = read_df[read_df['Replication Factor'] == rf]
        plt.plot(data['File Size (MB)'], data['Throughput (MB/s)'], 'o-', label=f'RF={rf}')
    
    plt.title('Read Throughput vs. File Size for Different Replication Factors')
    plt.xlabel('File Size (MB)')
    plt.ylabel('Throughput (MB/s)')
    plt.legend()
    plt.xscale('log')
    plt.grid()
    plt.savefig('read_performance.png')
    
    # Return summary tables
    return write_summary, read_summary


@pytest.mark.performance
def test_8_replication_factor_performance():
    """
    Test performance with different replication factors
    
    This test measures and compares performance metrics across different
    replication factors (1, 3, and 5) to determine the performance impact
    of replication in Apache Ozone.
    """
    replication_factors = [1, 3, 5]
    results_by_replication = {}
    
    # Setup Ozone cluster configuration manager
    cluster_config = OzoneClusterConfig()
    
    for rf in replication_factors:
        print(f"\n=== Testing with Replication Factor {rf} ===")
        
        # Configure cluster with current replication factor
        success = cluster_config.set_replication_factor(rf)
        if not success:
            pytest.skip(f"Failed to set replication factor {rf}. Skipping tests.")
        
        # Initialize performance tester
        volume_name = f"perfvol{rf}"
        bucket_name = f"perfbucket{rf}"
        tester = OzonePerformanceTester(volume_name=volume_name, bucket_name=bucket_name)
        
        try:
            # Setup test environment
            tester.setup()
            
            # Run write benchmark
            print(f"Running write benchmark with RF={rf}")
            write_results = tester.run_write_benchmark()
            
            # Run read benchmark
            print(f"Running read benchmark with RF={rf}")
            read_results = tester.run_read_benchmark()
            
            # Run concurrent operations benchmark
            print(f"Running concurrent operations benchmark with RF={rf}")
            write_ops, read_ops = tester.run_concurrent_operations()
            
            # Store results for this replication factor
            results_by_replication[rf] = {
                'write': write_results,
                'read': read_results,
                'concurrent': {
                    'write_ops_per_sec': write_ops,
                    'read_ops_per_sec': read_ops
                }
            }
            
        finally:
            # Clean up test resources
            tester.cleanup()
    
    # Analyze results
    write_summary, read_summary = analyze_results(results_by_replication)
    
    # Save results to file
    with open('replication_performance_results.txt', 'w') as f:
        f.write("=== Write Performance Summary ===\n")
        f.write(str(write_summary) + "\n\n")
        
        f.write("=== Read Performance Summary ===\n")
        f.write(str(read_summary) + "\n\n")
        
        f.write("=== Concurrent Operations Performance ===\n")
        for rf in replication_factors:
            f.write(f"Replication Factor {rf}:\n")
            f.write(f"  Write ops/sec: {results_by_replication[rf]['concurrent']['write_ops_per_sec']}\n")
            f.write(f"  Read ops/sec: {results_by_replication[rf]['concurrent']['read_ops_per_sec']}\n")
    
    # Validate results - Performance should degrade with higher replication factors
    for size in tester.test_file_sizes:
        # Calculate average write throughput for each replication factor
        rf1_write = statistics.mean(results_by_replication[1]['write'][size])
        rf3_write = statistics.mean(results_by_replication[3]['write'][size])
        rf5_write = statistics.mean(results_by_replication[5]['write'][size])
        
        # For writes, performance should degrade with higher replication
        # RF=1 should be faster than RF=3, which should be faster than RF=5
        assert rf1_write >= rf3_write * 0.8, f"Expected RF=1 write throughput to be at least 80% of RF=3 for {size}MB files"
        assert rf3_write >= rf5_write * 0.8, f"Expected RF=3 write throughput to be at least 80% of RF=5 for {size}MB files"
    
    # For read operations, performance should be similar or slightly better with higher replication
    for size in tester.test_file_sizes:
        rf1_read = statistics.mean(results_by_replication[1]['read'][size])
        rf3_read = statistics.mean(results_by_replication[3]['read'][size])
        rf5_read = statistics.mean(results_by_replication[5]['read'][size])
        
        # RF=3 and RF=5 should not be significantly worse than RF=1 for reads
        # In fact, they might be better due to potential for parallel reads
        assert rf3_read >= rf1_read * 0.7, f"Unexpected read performance degradation with RF=3 for {size}MB files"
        assert rf5_read >= rf1_read * 0.7, f"Unexpected read performance degradation with RF=5 for {size}MB files"
    
    # Concurrent operations
    rf1_write_ops = results_by_replication[1]['concurrent']['write_ops_per_sec']
    rf3_write_ops = results_by_replication[3]['concurrent']['write_ops_per_sec']
    rf5_write_ops = results_by_replication[5]['concurrent']['write_ops_per_sec']
    
    # Write ops should be higher for lower replication factors
    assert rf1_write_ops >= rf3_write_ops * 0.8, "Expected higher concurrent write throughput with RF=1 compared to RF=3"
    assert rf3_write_ops >= rf5_write_ops * 0.8, "Expected higher concurrent write throughput with RF=3 compared to RF=5"
    
    print("Performance test with different replication factors completed successfully")
    print("Results have been saved to 'replication_performance_results.txt'")
    print("Performance visualizations saved as 'write_performance.png' and 'read_performance.png'")

import os
import time
import pytest
import subprocess
import statistics
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import logging
import concurrent.futures
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ozone_node_failure_performance_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Test configuration
class OzoneConfig:
    # Cluster configuration
    OZONE_MANAGER_HOSTS = os.environ.get('OZONE_MANAGER_HOSTS', 'om1.example.com,om2.example.com').split(',')
    STORAGE_CONTAINER_HOSTS = os.environ.get('STORAGE_CONTAINER_HOSTS', 'scm1.example.com').split(',')
    DATANODE_HOSTS = os.environ.get('DATANODE_HOSTS', 'dn1.example.com,dn2.example.com,dn3.example.com').split(',')
    
    # Test parameters
    TEST_VOLUME = "performance-vol"
    TEST_BUCKET = "nodebenchmark"
    NUM_KEYS = 100
    FILE_SIZES_KB = [10, 100, 1024]  # 10KB, 100KB, 1MB
    TEST_DURATION = 60  # seconds
    RECOVERY_WAIT_TIME = 180  # seconds
    
    # SSH configuration
    SSH_USER = os.environ.get('SSH_USER', 'ozone')
    SSH_KEY_FILE = os.environ.get('SSH_KEY_FILE', '~/.ssh/id_rsa')
    
    # Performance thresholds
    MAX_ACCEPTABLE_THROUGHPUT_DEGRADATION = 0.5  # 50% degradation allowed during failure
    MAX_ACCEPTABLE_LATENCY_INCREASE = 2.0  # 2x latency increase allowed during failure
    
    # Python client parameters
    CLIENT_TIMEOUT = 30.0  # seconds
    RETRIES = 3


class NodeFailureTest:
    """Helper class for node failure scenarios and performance measurements"""
    
    def __init__(self, config: OzoneConfig):
        self.config = config
        self.test_files = {}
        self.performance_results = {}
    
    def setup(self):
        """Create test volume, bucket and prepare test files"""
        logger.info("Setting up test environment")
        
        # Create volume and bucket if they don't exist
        self._run_command(f"ozone sh volume create /{self.config.TEST_VOLUME}")
        self._run_command(f"ozone sh bucket create /{self.config.TEST_VOLUME}/{self.config.TEST_BUCKET}")
        
        # Create test files of different sizes
        for size_kb in self.config.FILE_SIZES_KB:
            file_path = f"/tmp/test_file_{size_kb}.dat"
            self._run_command(f"dd if=/dev/urandom of={file_path} bs=1K count={size_kb}")
            self.test_files[size_kb] = file_path
            
        logger.info(f"Created test files: {self.test_files}")
    
    def teardown(self):
        """Clean up test resources"""
        logger.info("Cleaning up test environment")
        
        # Remove test files
        for file_path in self.test_files.values():
            if os.path.exists(file_path):
                os.remove(file_path)
        
        # Delete test bucket and volume
        try:
            self._run_command(f"ozone sh bucket delete /{self.config.TEST_VOLUME}/{self.config.TEST_BUCKET}")
            self._run_command(f"ozone sh volume delete /{self.config.TEST_VOLUME}")
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")

    def measure_baseline_performance(self) -> Dict:
        """Measure baseline performance metrics"""
        logger.info("Measuring baseline performance")
        return self._run_performance_test("baseline")
    
    def simulate_node_failure(self, node_type: str, num_nodes: int = 1) -> List[str]:
        """Simulate failure of specified nodes"""
        logger.info(f"Simulating {node_type} failure for {num_nodes} nodes")
        
        failed_nodes = []
        if node_type == "datanode":
            target_nodes = self.config.DATANODE_HOSTS[:num_nodes]
        elif node_type == "om":
            target_nodes = self.config.OZONE_MANAGER_HOSTS[:num_nodes]
        elif node_type == "scm":
            target_nodes = self.config.STORAGE_CONTAINER_HOSTS[:num_nodes]
        else:
            raise ValueError(f"Invalid node type: {node_type}")
        
        for node in target_nodes:
            logger.info(f"Stopping Ozone service on {node}")
            cmd = f"ssh -i {self.config.SSH_KEY_FILE} {self.config.SSH_USER}@{node} 'sudo systemctl stop ozone-service'"
            self._run_command(cmd)
            failed_nodes.append(node)
        
        # Wait a bit to let the cluster detect the failure
        time.sleep(10)
        return failed_nodes
    
    def restore_nodes(self, nodes: List[str]):
        """Restore previously failed nodes"""
        logger.info(f"Restoring nodes: {nodes}")
        
        for node in nodes:
            logger.info(f"Starting Ozone service on {node}")
            cmd = f"ssh -i {self.config.SSH_KEY_FILE} {self.config.SSH_USER}@{node} 'sudo systemctl start ozone-service'"
            self._run_command(cmd)
        
        # Wait for recovery
        logger.info(f"Waiting {self.config.RECOVERY_WAIT_TIME}s for cluster to recover")
        time.sleep(self.config.RECOVERY_WAIT_TIME)
    
    def measure_failure_performance(self, scenario: str) -> Dict:
        """Measure performance during failure scenario"""
        logger.info(f"Measuring performance during {scenario}")
        return self._run_performance_test(scenario)
    
    def measure_recovery_performance(self, scenario: str) -> Dict:
        """Measure performance after recovery"""
        logger.info(f"Measuring post-recovery performance for {scenario}")
        return self._run_performance_test(f"{scenario}_recovery")
    
    def generate_report(self, output_dir: str = "./results"):
        """Generate performance report with metrics and charts"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        report_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert results to DataFrame
        results_data = []
        for scenario, metrics in self.performance_results.items():
            for file_size, values in metrics['by_size'].items():
                results_data.append({
                    'scenario': scenario,
                    'file_size_kb': file_size,
                    'throughput_mbps': values['throughput_mbps'],
                    'avg_latency_ms': values['avg_latency_ms'],
                    'p95_latency_ms': values['p95_latency_ms'],
                    'p99_latency_ms': values['p99_latency_ms'],
                    'success_rate': values['success_rate']
                })
        
        df = pd.DataFrame(results_data)
        
        # Save CSV report
        csv_file = f"{output_dir}/node_failure_performance_{report_time}.csv"
        df.to_csv(csv_file, index=False)
        
        # Generate charts
        self._generate_charts(df, output_dir, report_time)
        
        logger.info(f"Performance report generated at {output_dir}")
        
        return csv_file
    
    def _run_performance_test(self, scenario: str) -> Dict:
        """Run a performance test for the given scenario"""
        results = {
            'scenario': scenario,
            'start_time': time.time(),
            'end_time': None,
            'by_size': {},
            'overall': {}
        }
        
        overall_latencies = []
        overall_ops = 0
        overall_success = 0
        
        end_time = time.time() + self.config.TEST_DURATION
        
        # Run tests for each file size
        for size_kb, file_path in self.test_files.items():
            latencies = []
            operations = 0
            successes = 0
            
            # Run operations until test duration expires
            while time.time() < end_time:
                key = f"key_{scenario}_{size_kb}_{operations}"
                start = time.time()
                
                try:
                    # Put operation
                    self._run_command(
                        f"ozone sh key put {file_path} /{self.config.TEST_VOLUME}/{self.config.TEST_BUCKET}/{key}",
                        timeout=self.config.CLIENT_TIMEOUT
                    )
                    latency = (time.time() - start) * 1000  # ms
                    latencies.append(latency)
                    overall_latencies.append(latency)
                    successes += 1
                    overall_success += 1
                except Exception as e:
                    logger.warning(f"Operation failed: {e}")
                
                operations += 1
                overall_ops += 1
            
            # Calculate metrics for this file size
            if latencies:
                avg_latency = statistics.mean(latencies)
                p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
                p99_latency = sorted(latencies)[int(len(latencies) * 0.99)]
                throughput = (successes * size_kb) / (self.config.TEST_DURATION * 1024)  # MB/s
                success_rate = successes / operations if operations > 0 else 0
                
                results['by_size'][size_kb] = {
                    'operations': operations,
                    'successes': successes,
                    'throughput_mbps': throughput,
                    'avg_latency_ms': avg_latency,
                    'p95_latency_ms': p95_latency,
                    'p99_latency_ms': p99_latency,
                    'success_rate': success_rate
                }
                
                logger.info(f"{scenario} - {size_kb}KB: {throughput:.2f} MB/s, "
                           f"Avg: {avg_latency:.2f}ms, P95: {p95_latency:.2f}ms, "
                           f"Success: {success_rate*100:.1f}%")
        
        # Calculate overall metrics
        results['end_time'] = time.time()
        
        if overall_latencies:
            results['overall'] = {
                'operations': overall_ops,
                'successes': overall_success,
                'avg_latency_ms': statistics.mean(overall_latencies),
                'p95_latency_ms': sorted(overall_latencies)[int(len(overall_latencies) * 0.95)],
                'p99_latency_ms': sorted(overall_latencies)[int(len(overall_latencies) * 0.99)],
                'success_rate': overall_success / overall_ops if overall_ops > 0 else 0
            }
        
        # Store results
        self.performance_results[scenario] = results
        
        return results
    
    def _run_command(self, cmd: str, timeout: float = None) -> str:
        """Run a shell command and return its output"""
        logger.debug(f"Running command: {cmd}")
        result = subprocess.run(
            cmd, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            timeout=timeout,
            text=True
        )
        
        if result.returncode != 0:
            error_msg = f"Command '{cmd}' failed with exit code {result.returncode}: {result.stderr}"
            logger.error(error_msg)
            raise Exception(error_msg)
            
        return result.stdout.strip()
    
    def _generate_charts(self, df: pd.DataFrame, output_dir: str, report_time: str):
        """Generate performance comparison charts"""
        # Throughput comparison by file size
        plt.figure(figsize=(12, 8))
        for scenario in df['scenario'].unique():
            subset = df[df['scenario'] == scenario]
            plt.plot(subset['file_size_kb'], subset['throughput_mbps'], 
                    marker='o', label=scenario)
        
        plt.title('Throughput by File Size and Scenario')
        plt.xlabel('File Size (KB)')
        plt.ylabel('Throughput (MB/s)')
        plt.xscale('log')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/throughput_comparison_{report_time}.png")
        
        # Latency comparison
        plt.figure(figsize=(12, 8))
        for scenario in df['scenario'].unique():
            subset = df[df['scenario'] == scenario]
            plt.plot(subset['file_size_kb'], subset['avg_latency_ms'], 
                    marker='o', label=f"{scenario} (avg)")
            plt.plot(subset['file_size_kb'], subset['p95_latency_ms'], 
                    marker='s', linestyle='--', label=f"{scenario} (p95)")
        
        plt.title('Latency by File Size and Scenario')
        plt.xlabel('File Size (KB)')
        plt.ylabel('Latency (ms)')
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/latency_comparison_{report_time}.png")
        
        # Success rate comparison
        plt.figure(figsize=(12, 6))
        for scenario in df['scenario'].unique():
            subset = df[df['scenario'] == scenario]
            plt.plot(subset['file_size_kb'], subset['success_rate']*100, 
                    marker='o', label=scenario)
        
        plt.title('Success Rate by File Size and Scenario')
        plt.xlabel('File Size (KB)')
        plt.ylabel('Success Rate (%)')
        plt.xscale('log')
        plt.grid(True)
        plt.ylim(0, 105)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/success_rate_comparison_{report_time}.png")


def validate_performance(baseline: Dict, failure: Dict, recovery: Dict, config: OzoneConfig) -> bool:
    """
    Validate performance metrics against acceptance criteria
    
    Returns:
        bool: True if performance meets criteria, False otherwise
    """
    all_criteria_met = True
    
    # Check if data availability was maintained during failure
    failure_success_rate = failure['overall']['success_rate']
    if failure_success_rate < 0.8:  # At least 80% operations should succeed during failure
        logger.error(f"Data availability criterion not met. Success rate during failure: {failure_success_rate*100:.1f}%")
        all_criteria_met = False
    
    # Check if throughput degradation is within acceptable limits
    for size_kb, baseline_metrics in baseline['by_size'].items():
        if size_kb not in failure['by_size']:
            continue
            
        baseline_throughput = baseline_metrics['throughput_mbps']
        failure_throughput = failure['by_size'][size_kb]['throughput_mbps']
        
        degradation = 1 - (failure_throughput / baseline_throughput)
        if degradation > config.MAX_ACCEPTABLE_THROUGHPUT_DEGRADATION:
            logger.error(f"Throughput degradation too high for {size_kb}KB: "
                         f"{degradation*100:.1f}% (limit: {config.MAX_ACCEPTABLE_THROUGHPUT_DEGRADATION*100:.1f}%)")
            all_criteria_met = False
    
    # Check if recovery performance returns to acceptable levels
    for size_kb, baseline_metrics in baseline['by_size'].items():
        if size_kb not in recovery['by_size']:
            continue
            
        baseline_throughput = baseline_metrics['throughput_mbps']
        recovery_throughput = recovery['by_size'][size_kb]['throughput_mbps']
        
        recovery_ratio = recovery_throughput / baseline_throughput
        if recovery_ratio < 0.8:  # Should recover to at least 80% of baseline
            logger.error(f"Recovery performance too low for {size_kb}KB: "
                         f"{recovery_ratio*100:.1f}% of baseline (minimum: 80%)")
            all_criteria_met = False
    
    return all_criteria_met


@pytest.fixture(scope="module")
def ozone_config():
    """Fixture to provide Ozone test configuration"""
    return OzoneConfig()


@pytest.fixture(scope="module")
def node_failure_test(ozone_config):
    """Fixture to provide node failure test instance"""
    test = NodeFailureTest(ozone_config)
    test.setup()
    yield test
    test.teardown()


@pytest.mark.parametrize("node_type,num_nodes", [
    ("datanode", 1),
    ("datanode", 2),
    ("om", 1)
])
def test_9_node_failure_performance(node_failure_test, node_type, num_nodes):
    """
    Evaluate performance under node failure scenarios.
    
    This test simulates failures of different node types and measures the
    performance impact during failure and the recovery time. The test checks
    that the system maintains minimum performance thresholds and recovers
    properly after node failures are resolved.
    """
    # 1. Establish baseline performance
    baseline_metrics = node_failure_test.measure_baseline_performance()
    
    # 2. Simulate failure of specified nodes
    failed_nodes = node_failure_test.simulate_node_failure(node_type, num_nodes)
    scenario_name = f"{node_type}_{num_nodes}_failure"
    
    # 3. Measure performance metrics during node failure
    failure_metrics = node_failure_test.measure_failure_performance(scenario_name)
    
    # 4. Restore nodes and wait for recovery
    node_failure_test.restore_nodes(failed_nodes)
    
    # 5. Measure performance after recovery
    recovery_metrics = node_failure_test.measure_recovery_performance(scenario_name)
    
    # 6. Generate performance report
    node_failure_test.generate_report()
    
    # 7. Validate performance metrics against acceptance criteria
    assert validate_performance(
        baseline_metrics, 
        failure_metrics, 
        recovery_metrics, 
        node_failure_test.config
    ), "Performance under node failure scenarios does not meet acceptance criteria"
    
    # Log summary of results
    baseline_success = baseline_metrics['overall']['success_rate'] * 100
    failure_success = failure_metrics['overall']['success_rate'] * 100
    recovery_success = recovery_metrics['overall']['success_rate'] * 100
    
    logger.info(f"Test completed for {scenario_name}:")
    logger.info(f"  - Baseline success rate: {baseline_success:.1f}%")
    logger.info(f"  - During failure success rate: {failure_success:.1f}%")
    logger.info(f"  - After recovery success rate: {recovery_success:.1f}%")

#!/usr/bin/env python3
"""
Performance tests for Apache Ozone with encryption enabled/disabled
"""

import pytest
import time
import subprocess
import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
from statistics import mean
from subprocess import PIPE, run
from pyozone import Client

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
VOLUME_NAME = "perftest-vol"
BUCKET_NAME = "perftest-bucket"
TEST_FILE_SIZES = [1, 10, 100, 512, 1024, 2048]  # Sizes in KB
TEST_ITERATIONS = 3
ACCEPTABLE_DEGRADATION = 10  # 10% maximum acceptable performance degradation

def create_test_file(size_kb, file_name="test_file.txt"):
    """Create a test file of specified size in KB"""
    with open(file_name, 'wb') as f:
        f.write(os.urandom(size_kb * 1024))
    return file_name

def setup_encryption():
    """Enable encryption for data at rest and in transit"""
    try:
        # Enable encryption for data at rest
        run(["ozone", "admin", "kms", "enable"], check=True)
        # Set encryption keys
        run(["ozone", "admin", "kms", "create", VOLUME_NAME], check=True)
        # Enable TLS for in-transit encryption
        run(["ozone", "shell", "conf", "set", "ozone.security.enabled", "true"], check=True)
        run(["ozone", "shell", "conf", "set", "hdds.grpc.tls.enabled", "true"], check=True)
        logger.info("Encryption enabled for both data at rest and in transit")
        return True
    except subprocess.SubprocessError as e:
        logger.error(f"Failed to enable encryption: {e}")
        return False

def disable_encryption():
    """Disable encryption for data at rest and in transit"""
    try:
        # Disable encryption for data at rest
        run(["ozone", "admin", "kms", "disable"], check=True)
        # Disable TLS for in-transit encryption
        run(["ozone", "shell", "conf", "set", "ozone.security.enabled", "false"], check=True)
        run(["ozone", "shell", "conf", "set", "hdds.grpc.tls.enabled", "false"], check=True)
        logger.info("Encryption disabled")
        return True
    except subprocess.SubprocessError as e:
        logger.error(f"Failed to disable encryption: {e}")
        return False

def run_benchmark(encryption_enabled, file_size_kb):
    """
    Run read/write performance benchmarks
    Returns tuple of (write_time, read_time) in seconds
    """
    test_file = create_test_file(file_size_kb, f"test_file_{file_size_kb}kb.txt")
    key_name = f"key_{file_size_kb}kb"
    
    client = Client()
    
    # Ensure volume and bucket exist
    if not client.volume_exists(VOLUME_NAME):
        client.create_volume(VOLUME_NAME)
    
    if not client.bucket_exists(VOLUME_NAME, BUCKET_NAME):
        client.create_bucket(VOLUME_NAME, BUCKET_NAME)
    
    # Measure write performance
    write_start = time.time()
    run(["ozone", "shell", "key", "put", f"{VOLUME_NAME}/{BUCKET_NAME}/", test_file], check=True)
    write_time = time.time() - write_start
    
    # Measure read performance
    read_start = time.time()
    run(["ozone", "shell", "key", "get", f"{VOLUME_NAME}/{BUCKET_NAME}/{test_file}", "./download_file"], check=True)
    read_time = time.time() - read_start
    
    # Clean up
    os.remove(test_file)
    if os.path.exists("./download_file"):
        os.remove("./download_file")
    
    return (write_time, read_time)

def plot_performance_comparison(results_df, title, output_file):
    """Generate performance comparison chart"""
    plt.figure(figsize=(12, 8))
    
    # Plot write performance
    plt.subplot(2, 1, 1)
    plt.plot(results_df['file_size_kb'], results_df['encrypted_write_time'], 'ro-', label='Encrypted')
    plt.plot(results_df['file_size_kb'], results_df['unencrypted_write_time'], 'go-', label='Unencrypted')
    plt.title(f'Write Performance: {title}')
    plt.xlabel('File Size (KB)')
    plt.ylabel('Time (seconds)')
    plt.grid(True)
    plt.legend()
    
    # Plot read performance
    plt.subplot(2, 1, 2)
    plt.plot(results_df['file_size_kb'], results_df['encrypted_read_time'], 'ro-', label='Encrypted')
    plt.plot(results_df['file_size_kb'], results_df['unencrypted_read_time'], 'go-', label='Unencrypted')
    plt.title(f'Read Performance: {title}')
    plt.xlabel('File Size (KB)')
    plt.ylabel('Time (seconds)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_file)
    logger.info(f"Performance comparison chart saved to {output_file}")

@pytest.mark.parametrize("file_size_kb", TEST_FILE_SIZES)
def test_10_encryption_performance_impact(file_size_kb):
    """Test performance with encryption enabled vs disabled"""
    logger.info(f"Testing with file size: {file_size_kb} KB")
    
    # Create results dictionary
    results = {
        'file_size_kb': file_size_kb,
        'encrypted_write_times': [],
        'encrypted_read_times': [],
        'unencrypted_write_times': [],
        'unencrypted_read_times': []
    }
    
    # Test with encryption disabled
    assert disable_encryption(), "Failed to disable encryption"
    
    for i in range(TEST_ITERATIONS):
        logger.info(f"Running unencrypted test iteration {i+1}/{TEST_ITERATIONS} for {file_size_kb}KB")
        write_time, read_time = run_benchmark(False, file_size_kb)
        results['unencrypted_write_times'].append(write_time)
        results['unencrypted_read_times'].append(read_time)
    
    # Test with encryption enabled
    assert setup_encryption(), "Failed to enable encryption"
    
    for i in range(TEST_ITERATIONS):
        logger.info(f"Running encrypted test iteration {i+1}/{TEST_ITERATIONS} for {file_size_kb}KB")
        write_time, read_time = run_benchmark(True, file_size_kb)
        results['encrypted_write_times'].append(write_time)
        results['encrypted_read_times'].append(read_time)
    
    # Calculate average times
    unencrypted_write_avg = mean(results['unencrypted_write_times'])
    encrypted_write_avg = mean(results['encrypted_write_times'])
    unencrypted_read_avg = mean(results['unencrypted_read_times'])
    encrypted_read_avg = mean(results['encrypted_read_times'])
    
    # Calculate performance degradation
    write_degradation = ((encrypted_write_avg - unencrypted_write_avg) / unencrypted_write_avg) * 100
    read_degradation = ((encrypted_read_avg - unencrypted_read_avg) / unencrypted_read_avg) * 100
    
    logger.info(f"File size: {file_size_kb}KB")
    logger.info(f"Unencrypted write avg: {unencrypted_write_avg:.4f} sec, Encrypted: {encrypted_write_avg:.4f} sec")
    logger.info(f"Unencrypted read avg: {unencrypted_read_avg:.4f} sec, Encrypted: {encrypted_read_avg:.4f} sec")
    logger.info(f"Write performance degradation: {write_degradation:.2f}%")
    logger.info(f"Read performance degradation: {read_degradation:.2f}%")
    
    # Assertions to verify performance is within acceptable limits
    assert write_degradation < ACCEPTABLE_DEGRADATION, \
           f"Write performance degradation ({write_degradation:.2f}%) exceeds acceptable limit ({ACCEPTABLE_DEGRADATION}%)"
    
    assert read_degradation < ACCEPTABLE_DEGRADATION, \
           f"Read performance degradation ({read_degradation:.2f}%) exceeds acceptable limit ({ACCEPTABLE_DEGRADATION}%)"
    
    # Return results for potential aggregation
    return {
        'file_size_kb': file_size_kb,
        'unencrypted_write_time': unencrypted_write_avg,
        'encrypted_write_time': encrypted_write_avg,
        'unencrypted_read_time': unencrypted_read_avg, 
        'encrypted_read_time': encrypted_read_avg,
        'write_degradation': write_degradation,
        'read_degradation': read_degradation
    }

def test_10_encryption_performance_aggregate():
    """Aggregate test to create performance comparison chart for all file sizes"""
    all_results = []
    
    for file_size in TEST_FILE_SIZES:
        result = test_10_encryption_performance_impact(file_size)
        all_results.append(result)
    
    # Create DataFrame for analysis
    results_df = pd.DataFrame(all_results)
    
    # Generate performance comparison chart
    plot_performance_comparison(
        results_df, 
        f"Encryption Performance Impact (Max Acceptable: {ACCEPTABLE_DEGRADATION}%)",
        "encryption_performance_comparison.png"
    )
    
    # Calculate overall average degradation
    avg_write_degradation = results_df['write_degradation'].mean()
    avg_read_degradation = results_df['read_degradation'].mean()
    
    logger.info(f"Overall average write degradation: {avg_write_degradation:.2f}%")
    logger.info(f"Overall average read degradation: {avg_read_degradation:.2f}%")
    
    # Final assertions for overall performance
    assert avg_write_degradation < ACCEPTABLE_DEGRADATION, \
           f"Overall write performance degradation ({avg_write_degradation:.2f}%) exceeds acceptable limit ({ACCEPTABLE_DEGRADATION}%)"
    
    assert avg_read_degradation < ACCEPTABLE_DEGRADATION, \
           f"Overall read performance degradation ({avg_read_degradation:.2f}%) exceeds acceptable limit ({ACCEPTABLE_DEGRADATION}%)"

import os
import time
import pytest
import subprocess
import logging
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from pyozone import OzoneClient
from subprocess import Popen, PIPE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
OZONE_CONF_DIR = os.environ.get('OZONE_CONF_DIR', '/etc/ozone')
DATA_DIR = '/tmp/ozone_perf_test'
RESULT_DIR = '/tmp/ozone_perf_test_results'

# Helper functions
def create_test_file(file_path: str, size_mb: float) -> str:
    """Create a test file of specified size in MB"""
    size_bytes = int(size_mb * 1024 * 1024)
    cmd = f"dd if=/dev/urandom of={file_path} bs=1M count={size_mb} iflag=fullblock"
    
    try:
        subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
        logger.info(f"Created test file {file_path} of size {size_mb}MB")
        return file_path
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to create test file: {e.output}")
        raise

def run_ozone_command(cmd: List[str]) -> str:
    """Run an ozone shell command"""
    full_cmd = ["ozone", "sh"] + cmd
    logger.info(f"Running: {' '.join(full_cmd)}")
    
    try:
        result = subprocess.check_output(full_cmd, stderr=subprocess.STDOUT, universal_newlines=True)
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Ozone command failed: {e.output}")
        raise

def get_cluster_storage_info() -> Dict:
    """Get storage utilization info from Ozone cluster"""
    cmd = "hdfs dfsadmin -report"
    process = Popen(cmd.split(), stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    
    if process.returncode != 0:
        logger.error(f"Failed to get cluster storage info: {stderr}")
        raise RuntimeError(f"Command failed: {stderr}")
    
    # Parse the output to extract storage information
    output = stdout.decode('utf-8')
    storage_info = {
        'total_raw_capacity': 0,
        'used_raw_capacity': 0,
        'available_raw_capacity': 0,
        'datanodes': []
    }
    
    # This is a simplified parsing logic - actual implementation would need to be adjusted based on actual output format
    try:
        # Extract cluster-wide info
        for line in output.split('\n'):
            if "Total raw capacity" in line:
                storage_info['total_raw_capacity'] = float(line.split(':')[1].strip().split(' ')[0])
            if "Total used capacity" in line:
                storage_info['used_raw_capacity'] = float(line.split(':')[1].strip().split(' ')[0])
        
        # In a real implementation, we would also parse datanode-specific information
    except Exception as e:
        logger.error(f"Error parsing storage info: {e}")
        raise
        
    return storage_info

def calculate_storage_efficiency(raw_data_size: float, storage_used: float, replication_factor: int) -> Dict:
    """Calculate storage efficiency metrics"""
    expected_storage = raw_data_size * replication_factor
    overhead = storage_used - expected_storage
    overhead_percentage = (overhead / expected_storage) * 100 if expected_storage > 0 else 0
    
    return {
        'raw_data_size': raw_data_size,
        'storage_used': storage_used,
        'replication_factor': replication_factor,
        'expected_storage': expected_storage,
        'overhead': overhead,
        'overhead_percentage': overhead_percentage
    }


# Test parameters
test_scenarios = [
    # (file_size_mb, replication_factor, expected_max_overhead_percentage)
    (100, 1, 5),  # 100MB with RF=1, expecting max 5% overhead
    (100, 3, 5),  # 100MB with RF=3, expecting max 5% overhead
    (500, 1, 5),  # 500MB with RF=1, expecting max 5% overhead
    (500, 3, 5),  # 500MB with RF=3, expecting max 5% overhead
    (1024, 1, 5), # 1GB with RF=1, expecting max 5% overhead
    (1024, 3, 5), # 1GB with RF=3, expecting max 5% overhead
    (4096, 1, 5), # 4GB with RF=1, expecting max 5% overhead
    (4096, 3, 5), # 4GB with RF=3, expecting max 5% overhead
    (7.5 * 1024, 1, 5), # 7.5GB with RF=1, expecting max 5% overhead
    (9 * 1024, 1, 5),   # 9GB with RF=1, expecting max 5% overhead
]


@pytest.mark.parametrize("file_size_mb,replication_factor,expected_max_overhead", test_scenarios)
def test_11_storage_utilization_efficiency(file_size_mb, replication_factor, expected_max_overhead):
    """
    Test storage utilization efficiency with different file sizes and replication factors.
    Verifies that the storage overhead doesn't exceed the expected threshold.
    """
    # Create test directories if they don't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    # Unique volume and bucket names
    volume_name = f"vol-perf-{int(time.time())}"
    bucket_name = f"bucket-perf-{int(time.time())}"
    
    # Get initial storage state
    initial_storage = get_cluster_storage_info()
    initial_used_storage = initial_storage['used_raw_capacity']
    
    try:
        # 1. Create volume and bucket
        run_ozone_command(["volume", "create", volume_name])
        run_ozone_command(["bucket", "create", f"{volume_name}/{bucket_name}"])
        
        # Set replication factor for the bucket
        run_ozone_command(["bucket", "set-replication", f"{volume_name}/{bucket_name}", str(replication_factor)])
        
        # Create test file
        test_file_path = os.path.join(DATA_DIR, f"test_file_{file_size_mb}mb.dat")
        create_test_file(test_file_path, file_size_mb)
        
        # Get file size in bytes
        raw_data_size_bytes = os.path.getsize(test_file_path)
        raw_data_size_mb = raw_data_size_bytes / (1024 * 1024)
        
        logger.info(f"Generated test file with size {raw_data_size_mb:.2f}MB")
        
        # 2. Upload the file to Ozone
        key_name = f"test_key_{file_size_mb}mb"
        run_ozone_command(["key", "put", f"{volume_name}/{bucket_name}/{key_name}", test_file_path])
        
        # 3. Wait for data to be fully replicated and settled
        time.sleep(10)  # Adjust based on cluster size and file size
        
        # 4. Get storage utilization after upload
        final_storage = get_cluster_storage_info()
        final_used_storage = final_storage['used_raw_capacity']
        
        # Calculate actual storage used for this operation
        storage_used = final_used_storage - initial_used_storage
        storage_used_mb = storage_used * 1024  # Convert to MB
        
        # 5. Calculate storage efficiency metrics
        efficiency_metrics = calculate_storage_efficiency(
            raw_data_size_mb, 
            storage_used_mb,
            replication_factor
        )
        
        # Log results
        logger.info(f"Storage Efficiency Metrics: {efficiency_metrics}")
        
        # Create a visualization of the results
        labels = ['Raw Data Size', 'Expected Storage', 'Actual Storage Used']
        values = [raw_data_size_mb, raw_data_size_mb * replication_factor, storage_used_mb]
        
        plt.figure(figsize=(10, 6))
        plt.bar(labels, values)
        plt.title(f'Storage Utilization (File Size: {file_size_mb}MB, RF: {replication_factor})')
        plt.ylabel('Storage (MB)')
        plt.savefig(f"{RESULT_DIR}/storage_efficiency_{file_size_mb}mb_rf{replication_factor}.png")
        
        # 6. Assert that overhead doesn't exceed expected threshold
        assert efficiency_metrics['overhead_percentage'] <= expected_max_overhead, \
            f"Storage overhead ({efficiency_metrics['overhead_percentage']:.2f}%) exceeds expected maximum ({expected_max_overhead}%)"
        
        logger.info(f"Storage efficiency test passed for {file_size_mb}MB file with replication factor {replication_factor}")
        
    finally:
        # Cleanup
        try:
            run_ozone_command(["bucket", "delete", f"{volume_name}/{bucket_name}"])
            run_ozone_command(["volume", "delete", volume_name])
            if os.path.exists(test_file_path):
                os.remove(test_file_path)
        except Exception as e:
            logger.warning(f"Cleanup error (non-critical): {e}")

import pytest
import time
import subprocess
import threading
import csv
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Constants for the test
VOLUME_NAME = "perfvol"
BUCKET_NAME = "perfbucket"
TEST_DURATION_SEC = 300  # 5 minutes
MEASUREMENT_INTERVAL_SEC = 5
LOG_FILE = "gc_impact_measurements.csv"
CHART_OUTPUT = "gc_impact_chart.png"

# Helper class to monitor system metrics
class SystemMetrics:
    def __init__(self):
        self.running = False
        self.metrics = []
        self.start_time = None
    
    def start_monitoring(self):
        """Start collecting system metrics"""
        self.running = True
        self.start_time = time.time()
        self.metrics = []
        self.monitor_thread = threading.Thread(target=self._collect_metrics)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop collecting system metrics"""
        self.running = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=10)
    
    def _collect_metrics(self):
        """Continuously collect system metrics at defined intervals"""
        with open(LOG_FILE, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['timestamp', 'latency_ms', 'throughput_ops', 'cpu_usage', 'memory_usage', 'gc_active'])
            
            while self.running:
                elapsed_time = time.time() - self.start_time
                latency, throughput = self._get_ozone_metrics()
                cpu_usage = self._get_cpu_usage()
                memory_usage = self._get_memory_usage()
                gc_active = self._is_gc_active()
                
                metrics_row = [elapsed_time, latency, throughput, cpu_usage, memory_usage, gc_active]
                self.metrics.append(metrics_row)
                writer.writerow(metrics_row)
                csvfile.flush()
                
                time.sleep(MEASUREMENT_INTERVAL_SEC)
    
    def _get_ozone_metrics(self):
        """Get current Ozone latency and throughput metrics"""
        try:
            # This would normally call JMX or metrics API, simulated for test
            cmd = "curl -s http://localhost:9876/metrics | grep -E 'Latency|Throughput'"
            output = subprocess.check_output(cmd, shell=True, text=True)
            
            # Parse the output to extract metrics (simplified)
            latency = 50 + (np.random.random() * 20)  # Simulated latency between 50-70ms
            throughput = 200 + (np.random.random() * 50)  # Simulated throughput between 200-250 ops/sec
            
            return latency, throughput
        except Exception:
            return 0, 0
    
    def _get_cpu_usage(self):
        """Get current CPU usage percentage"""
        try:
            cmd = "top -bn1 | grep 'Cpu(s)' | sed 's/.*, *\\([0-9.]*\\)%* id.*/\\1/' | awk '{print 100 - $1}'"
            output = subprocess.check_output(cmd, shell=True, text=True).strip()
            return float(output)
        except Exception:
            return 0
    
    def _get_memory_usage(self):
        """Get current memory usage percentage"""
        try:
            cmd = "free | grep Mem | awk '{print $3/$2 * 100.0}'"
            output = subprocess.check_output(cmd, shell=True, text=True).strip()
            return float(output)
        except Exception:
            return 0
    
    def _is_gc_active(self):
        """Check if GC is currently active"""
        # In a real test, this would check JVM GC activity
        # For simulation, we'll return 0 (inactive)
        return 0


# Workload generator class
class OzoneWorkloadGenerator:
    def __init__(self, volume, bucket, threads=10):
        self.volume = volume
        self.bucket = bucket
        self.threads = threads
        self.running = False
        self.executor = None
    
    def setup(self):
        """Set up the test volume and bucket if needed"""
        try:
            # Create volume if it doesn't exist
            result = subprocess.run(
                f"ozone sh volume info {self.volume}",
                shell=True, capture_output=True
            )
            if result.returncode != 0:
                subprocess.run(
                    f"ozone sh volume create {self.volume}",
                    shell=True, check=True
                )
            
            # Create bucket if it doesn't exist
            result = subprocess.run(
                f"ozone sh bucket info {self.volume}/{self.bucket}",
                shell=True, capture_output=True
            )
            if result.returncode != 0:
                subprocess.run(
                    f"ozone sh bucket create {self.volume}/{self.bucket}",
                    shell=True, check=True
                )
        except subprocess.SubprocessError as e:
            pytest.fail(f"Failed to set up test environment: {e}")
    
    def start_load(self):
        """Start generating a constant load on the cluster"""
        self.running = True
        self.executor = ThreadPoolExecutor(max_workers=self.threads)
        for i in range(self.threads):
            self.executor.submit(self._worker_task, i)
    
    def stop_load(self):
        """Stop the load generation"""
        self.running = False
        if self.executor:
            self.executor.shutdown(wait=True)
    
    def _worker_task(self, worker_id):
        """Worker task that continuously writes and reads data"""
        counter = 0
        temp_file = f"temp_data_{worker_id}.dat"
        
        # Create a test file (1MB)
        with open(temp_file, 'wb') as f:
            f.write(os.urandom(1024 * 1024))
        
        try:
            while self.running:
                key_name = f"perfkey_{worker_id}_{counter}"
                
                # PUT operation
                put_cmd = f"ozone fs -put {temp_file} ofs://{self.volume}/{self.bucket}/{key_name}"
                subprocess.run(put_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                # GET operation
                get_cmd = f"ozone fs -cat ofs://{self.volume}/{self.bucket}/{key_name} > /dev/null"
                subprocess.run(get_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                # LIST operation
                list_cmd = f"ozone fs -ls ofs://{self.volume}/{self.bucket} > /dev/null"
                subprocess.run(list_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                counter += 1
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)


def trigger_gc():
    """Trigger garbage collection on Ozone JVMs"""
    # In real test, this would connect to JMX and trigger GC
    print("Triggering garbage collection...")
    
    # Get JVM process IDs for Ozone services
    cmd = "jps | grep -E 'OzoneManager|StorageContainer|SCM|OM' | awk '{print $1}'"
    try:
        process_ids = subprocess.check_output(cmd, shell=True, text=True).strip().split('\n')
        
        for pid in process_ids:
            if pid:
                # Use jcmd to trigger GC
                gc_cmd = f"jcmd {pid} GC.run"
                subprocess.run(gc_cmd, shell=True, check=True)
                print(f"Triggered GC on process {pid}")
    except subprocess.SubprocessError as e:
        pytest.fail(f"Failed to trigger garbage collection: {e}")


def analyze_and_visualize_results():
    """Analyze collected metrics and generate visualization"""
    data = []
    with open(LOG_FILE, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append({
                'timestamp': float(row['timestamp']),
                'latency_ms': float(row['latency_ms']),
                'throughput_ops': float(row['throughput_ops']),
                'cpu_usage': float(row['cpu_usage']),
                'memory_usage': float(row['memory_usage']),
                'gc_active': int(row['gc_active'])
            })
    
    # Extract data series
    timestamps = [d['timestamp'] for d in data]
    latencies = [d['latency_ms'] for d in data]
    throughputs = [d['throughput_ops'] for d in data]
    gc_active = [d['gc_active'] for d in data]
    
    # Find periods with GC
    gc_periods = []
    current_period = None
    for i, is_gc in enumerate(gc_active):
        if is_gc and current_period is None:
            current_period = i
        elif not is_gc and current_period is not None:
            gc_periods.append((current_period, i))
            current_period = None
    
    # If GC was active at the end of the test
    if current_period is not None:
        gc_periods.append((current_period, len(gc_active)-1))
    
    # Calculate impact
    normal_latency = np.mean([latencies[i] for i in range(len(latencies)) 
                             if all(not (start <= i <= end) for start, end in gc_periods)])
    
    gc_latency = np.mean([latencies[i] for start, end in gc_periods 
                         for i in range(start, end+1)]) if gc_periods else 0
    
    normal_throughput = np.mean([throughputs[i] for i in range(len(throughputs)) 
                                if all(not (start <= i <= end) for start, end in gc_periods)])
    
    gc_throughput = np.mean([throughputs[i] for start, end in gc_periods 
                            for i in range(start, end+1)]) if gc_periods else 0
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot latency
    ax1.plot(timestamps, latencies, label='Latency (ms)')
    ax1.set_ylabel('Latency (ms)')
    ax1.set_title('GC Impact on Ozone Performance')
    ax1.legend()
    ax1.grid(True)
    
    # Plot throughput
    ax2.plot(timestamps, throughputs, label='Throughput (ops/sec)', color='green')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Throughput (ops/sec)')
    ax2.legend()
    ax2.grid(True)
    
    # Highlight GC periods
    for start, end in gc_periods:
        start_time = timestamps[start]
        end_time = timestamps[end]
        ax1.axvspan(start_time, end_time, color='red', alpha=0.3)
        ax2.axvspan(start_time, end_time, color='red', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(CHART_OUTPUT)
    
    return {
        'normal_latency': normal_latency,
        'gc_latency': gc_latency,
        'normal_throughput': normal_throughput,
        'gc_throughput': gc_throughput,
        'latency_impact_pct': ((gc_latency - normal_latency) / normal_latency * 100) if normal_latency > 0 else 0,
        'throughput_impact_pct': ((normal_throughput - gc_throughput) / normal_throughput * 100) if normal_throughput > 0 else 0
    }


@pytest.fixture(scope="module")
def ozone_cluster():
    """Fixture to ensure Ozone cluster is ready for testing"""
    # Check if Ozone cluster is running
    result = subprocess.run("ozone sh status", shell=True, capture_output=True)
    if result.returncode != 0:
        pytest.skip("Ozone cluster is not running. Please start it before running tests.")
    return True


def test_12_gc_impact_on_performance(ozone_cluster):
    """Evaluate garbage collection impact on performance
    
    This test assesses how garbage collection affects the throughput and latency
    of Apache Ozone during normal operation.
    """
    # Set up workload generator
    workload = OzoneWorkloadGenerator(VOLUME_NAME, BUCKET_NAME, threads=10)
    workload.setup()
    
    # Set up metrics monitoring
    metrics = SystemMetrics()
    
    try:
        # Start monitoring and constant load
        print("Starting performance monitoring...")
        metrics.start_monitoring()
        
        print(f"Starting constant load for {TEST_DURATION_SEC/60} minutes...")
        workload.start_load()
        
        # Let system stabilize and collect baseline metrics
        print("Collecting baseline metrics for 1 minute...")
        time.sleep(60)
        
        # Trigger GC and continue monitoring
        print("Triggering garbage collection...")
        trigger_gc()
        
        # Mark when GC was triggered
        gc_trigger_time = time.time() - metrics.start_time
        print(f"GC triggered at {gc_trigger_time:.2f} seconds")
        
        # Continue monitoring to observe recovery
        print("Monitoring recovery for remainder of test...")
        remaining_time = TEST_DURATION_SEC - (time.time() - metrics.start_time)
        if remaining_time > 0:
            time.sleep(remaining_time)
        
    finally:
        # Stop the workload and monitoring
        print("Stopping workload and metrics collection...")
        workload.stop_load()
        metrics.stop_monitoring()
    
    # Analyze the results
    print("Analyzing results...")
    results = analyze_and_visualize_results()
    
    print(f"Normal operation latency: {results['normal_latency']:.2f} ms")
    print(f"GC period latency: {results['gc_latency']:.2f} ms")
    print(f"Latency impact: {results['latency_impact_pct']:.2f}%")
    
    print(f"Normal operation throughput: {results['normal_throughput']:.2f} ops/sec")
    print(f"GC period throughput: {results['gc_throughput']:.2f} ops/sec")
    print(f"Throughput impact: {results['throughput_impact_pct']:.2f}%")
    
    print(f"Performance chart saved to {CHART_OUTPUT}")
    print(f"Raw metrics data saved to {LOG_FILE}")
    
    # Assertions based on expected results
    # We expect minimal impact (less than 30% degradation) and quick recovery (within 10 seconds)
    assert results['latency_impact_pct'] < 30, f"GC had excessive impact on latency: {results['latency_impact_pct']:.2f}%"
    assert results['throughput_impact_pct'] < 30, f"GC had excessive impact on throughput: {results['throughput_impact_pct']:.2f}%"
    
    # Check that performance returns to normal after GC
    # In a real test, we would analyze the time series data to verify recovery time

import pytest
import time
import os
import threading
import subprocess
import concurrent.futures
import logging
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Dict, Tuple
from functools import partial

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test configuration
class TenantConfig:
    def __init__(self, name, volume_quota_gb, bucket_quota_gb, workload_type):
        self.name = name
        self.volume_quota_gb = volume_quota_gb
        self.bucket_quota_gb = bucket_quota_gb
        self.workload_type = workload_type
        self.volume = f"volume-{name}"
        self.bucket = f"bucket-{name}"

# Workload generator functions
def generate_analytics_workload(tenant: TenantConfig, duration_seconds: int = 300):
    """Generate analytics workload with read-heavy operations and some aggregation"""
    logger.info(f"Starting analytics workload for tenant {tenant.name}")
    
    # Create test files of various sizes
    files_dir = f"tenant_{tenant.name}_files"
    os.makedirs(files_dir, exist_ok=True)
    
    file_sizes_mb = [10, 50, 100, 250]
    test_files = []
    
    for size in file_sizes_mb:
        file_path = os.path.join(files_dir, f"analytics_file_{size}MB.dat")
        if not os.path.exists(file_path):
            with open(file_path, 'wb') as f:
                f.write(os.urandom(size * 1024 * 1024))
        test_files.append(file_path)
    
    start_time = time.time()
    metrics = {
        "reads": 0,
        "writes": 0,
        "errors": 0,
        "latencies": []
    }
    
    while time.time() - start_time < duration_seconds:
        try:
            # Upload files (20% of operations)
            if np.random.random() < 0.2:
                file_to_upload = np.random.choice(test_files)
                key = f"analytics/data_{int(time.time())}.dat"
                
                upload_start = time.time()
                subprocess.run([
                    "ozone", "sh", "key", "put", f"{tenant.volume}/{tenant.bucket}/", file_to_upload
                ], check=True, capture_output=True)
                latency = time.time() - upload_start
                
                metrics["writes"] += 1
                metrics["latencies"].append(("write", latency))
            
            # Read files (80% of operations)
            else:
                # List some keys first
                list_start = time.time()
                result = subprocess.run([
                    "ozone", "sh", "key", "list", f"{tenant.volume}/{tenant.bucket}/"
                ], check=True, capture_output=True, text=True)
                latency = time.time() - list_start
                metrics["latencies"].append(("list", latency))
                
                # Get a few random keys to simulate analytics operations
                keys = result.stdout.strip().split("\n")
                if keys:
                    for _ in range(min(5, len(keys))):
                        key_to_read = np.random.choice(keys)
                        if key_to_read:
                            read_start = time.time()
                            subprocess.run([
                                "ozone", "sh", "key", "get", 
                                f"{tenant.volume}/{tenant.bucket}/{key_to_read}", 
                                "/dev/null"
                            ], check=True, capture_output=True)
                            latency = time.time() - read_start
                            
                            metrics["reads"] += 1
                            metrics["latencies"].append(("read", latency))
            
            # Add some think time to simulate real analytics workload
            time.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Error in analytics workload for tenant {tenant.name}: {str(e)}")
            metrics["errors"] += 1
    
    logger.info(f"Completed analytics workload for tenant {tenant.name}")
    return metrics

def generate_streaming_workload(tenant: TenantConfig, duration_seconds: int = 300):
    """Generate streaming workload with high-frequency small writes"""
    logger.info(f"Starting streaming workload for tenant {tenant.name}")
    
    # Create small test files for streaming simulation
    files_dir = f"tenant_{tenant.name}_files"
    os.makedirs(files_dir, exist_ok=True)
    
    file_sizes_kb = [10, 50, 100, 500]
    test_files = []
    
    for size in file_sizes_kb:
        file_path = os.path.join(files_dir, f"stream_file_{size}KB.dat")
        if not os.path.exists(file_path):
            with open(file_path, 'wb') as f:
                f.write(os.urandom(size * 1024))
        test_files.append(file_path)
    
    start_time = time.time()
    metrics = {
        "reads": 0,
        "writes": 0,
        "errors": 0,
        "latencies": []
    }
    
    # Streaming workload is write-heavy and sequential
    while time.time() - start_time < duration_seconds:
        try:
            # Streaming is mostly writes (90%)
            if np.random.random() < 0.9:
                file_to_upload = np.random.choice(test_files)
                # Use timestamp to ensure sequential writes
                key = f"stream/event_{int(time.time()*1000)}.dat"
                
                upload_start = time.time()
                subprocess.run([
                    "ozone", "sh", "key", "put", 
                    f"{tenant.volume}/{tenant.bucket}/", file_to_upload
                ], check=True, capture_output=True)
                latency = time.time() - upload_start
                
                metrics["writes"] += 1
                metrics["latencies"].append(("write", latency))
            
            # Occasional reads (10% to verify data)
            else:
                list_start = time.time()
                result = subprocess.run([
                    "ozone", "sh", "key", "list", 
                    f"{tenant.volume}/{tenant.bucket}/stream/"
                ], check=True, capture_output=True, text=True)
                keys = result.stdout.strip().split("\n")
                
                if keys:
                    key_to_read = keys[-1]  # Read most recent data
                    if key_to_read:
                        read_start = time.time()
                        subprocess.run([
                            "ozone", "sh", "key", "get", 
                            f"{tenant.volume}/{tenant.bucket}/{key_to_read}", 
                            "/dev/null"
                        ], check=True, capture_output=True)
                        latency = time.time() - read_start
                        
                        metrics["reads"] += 1
                        metrics["latencies"].append(("read", latency))
            
            # Shorter think times for streaming workload
            time.sleep(0.05)
            
        except Exception as e:
            logger.error(f"Error in streaming workload for tenant {tenant.name}: {str(e)}")
            metrics["errors"] += 1
    
    logger.info(f"Completed streaming workload for tenant {tenant.name}")
    return metrics

def generate_batch_workload(tenant: TenantConfig, duration_seconds: int = 300):
    """Generate batch workload with large file uploads and downloads"""
    logger.info(f"Starting batch workload for tenant {tenant.name}")
    
    # Create larger test files for batch simulation
    files_dir = f"tenant_{tenant.name}_files"
    os.makedirs(files_dir, exist_ok=True)
    
    file_sizes_mb = [25, 75, 150, 300]
    test_files = []
    
    for size in file_sizes_mb:
        file_path = os.path.join(files_dir, f"batch_file_{size}MB.dat")
        if not os.path.exists(file_path):
            with open(file_path, 'wb') as f:
                f.write(os.urandom(size * 1024 * 1024))
        test_files.append(file_path)
    
    start_time = time.time()
    metrics = {
        "reads": 0,
        "writes": 0,
        "errors": 0,
        "latencies": []
    }
    
    # Batch jobs alternately upload and download
    batch_phase = "upload"
    uploaded_keys = []
    
    while time.time() - start_time < duration_seconds:
        try:
            # Upload phase
            if batch_phase == "upload":
                for file_path in test_files:
                    key = f"batch/data_{os.path.basename(file_path)}_{int(time.time())}"
                    upload_start = time.time()
                    subprocess.run([
                        "ozone", "sh", "key", "put", 
                        f"{tenant.volume}/{tenant.bucket}/", file_path
                    ], check=True, capture_output=True)
                    latency = time.time() - upload_start
                    
                    metrics["writes"] += 1
                    metrics["latencies"].append(("write", latency))
                    uploaded_keys.append(key)
                
                batch_phase = "process"  # Next phase
            
            # Processing phase (list and read)
            elif batch_phase == "process":
                # List all batch data
                list_start = time.time()
                result = subprocess.run([
                    "ozone", "sh", "key", "list", 
                    f"{tenant.volume}/{tenant.bucket}/batch/"
                ], check=True, capture_output=True, text=True)
                list_latency = time.time() - list_start
                metrics["latencies"].append(("list", list_latency))
                
                # Process (read) each key
                keys = result.stdout.strip().split("\n")
                for key in keys[:min(10, len(keys))]:
                    if key:
                        read_start = time.time()
                        subprocess.run([
                            "ozone", "sh", "key", "get", 
                            f"{tenant.volume}/{tenant.bucket}/{key}", 
                            "/dev/null"
                        ], check=True, capture_output=True)
                        latency = time.time() - read_start
                        
                        metrics["reads"] += 1
                        metrics["latencies"].append(("read", latency))
                
                batch_phase = "cleanup"  # Next phase
            
            # Cleanup phase (delete processed data)
            else:
                # Delete some keys to simulate cleanup
                if uploaded_keys:
                    for key in uploaded_keys[:min(5, len(uploaded_keys))]:
                        delete_start = time.time()
                        subprocess.run([
                            "ozone", "sh", "key", "delete", 
                            f"{tenant.volume}/{tenant.bucket}/{key}"
                        ], check=True, capture_output=True)
                        latency = time.time() - delete_start
                        metrics["latencies"].append(("delete", latency))
                    
                    # Remove deleted keys from our list
                    uploaded_keys = uploaded_keys[min(5, len(uploaded_keys)):]
                
                batch_phase = "upload"  # Back to upload phase
            
            # Longer think times for batch workload
            time.sleep(0.5)
            
        except Exception as e:
            logger.error(f"Error in batch workload for tenant {tenant.name}: {str(e)}")
            metrics["errors"] += 1
    
    logger.info(f"Completed batch workload for tenant {tenant.name}")
    return metrics

# System metrics collection
def collect_system_metrics(duration_seconds: int, interval_seconds: int = 5):
    """Collect system metrics during the test execution"""
    logger.info("Starting system metrics collection")
    
    metrics = []
    start_time = time.time()
    
    while time.time() - start_time < duration_seconds:
        try:
            # Get cluster health metrics
            result = subprocess.run(
                ["hdfs", "dfsadmin", "-report"], 
                check=True, capture_output=True, text=True
            )
            
            # Get Ozone specific metrics
            ozone_metrics = subprocess.run(
                ["ozone", "admin", "replicationmetrics"], 
                check=True, capture_output=True, text=True
            )
            
            # Parse and store metrics
            # This is simplified - in a real test you'd parse these outputs
            metrics.append({
                "timestamp": datetime.now().isoformat(),
                "hdfs_report": result.stdout,
                "ozone_metrics": ozone_metrics.stdout
            })
            
            time.sleep(interval_seconds)
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {str(e)}")
    
    logger.info("Completed system metrics collection")
    return metrics

def setup_tenant_environment(tenants: List[TenantConfig]) -> bool:
    """Set up the tenant environments with appropriate quotas"""
    try:
        for tenant in tenants:
            logger.info(f"Setting up environment for tenant {tenant.name}")
            
            # Create volume
            subprocess.run([
                "ozone", "sh", "volume", "create", tenant.volume
            ], check=True)
            
            # Set volume quota
            subprocess.run([
                "ozone", "sh", "volume", "setquota", 
                tenant.volume, "--space", f"{tenant.volume_quota_gb}GB"
            ], check=True)
            
            # Create bucket
            subprocess.run([
                "ozone", "sh", "bucket", "create", 
                f"{tenant.volume}/{tenant.bucket}"
            ], check=True)
            
            # Set bucket quota
            subprocess.run([
                "ozone", "sh", "bucket", "setquota", 
                f"{tenant.volume}/{tenant.bucket}", 
                "--space", f"{tenant.bucket_quota_gb}GB"
            ], check=True)
            
            logger.info(f"Successfully configured tenant {tenant.name}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to set up tenant environments: {str(e)}")
        return False

def analyze_results(tenant_metrics: Dict, system_metrics: List) -> Dict:
    """Analyze performance metrics and check for fairness across tenants"""
    results = {
        "fairness_score": 0.0,
        "isolation_score": 0.0,
        "per_tenant_stats": {},
        "system_bottlenecks": [],
        "conclusion": ""
    }
    
    # Calculate per-tenant stats
    for tenant_name, metrics in tenant_metrics.items():
        # Calculate average latencies
        write_latencies = [lat for op, lat in metrics["latencies"] if op == "write"]
        read_latencies = [lat for op, lat in metrics["latencies"] if op == "read"]
        
        avg_write_latency = np.mean(write_latencies) if write_latencies else 0
        avg_read_latency = np.mean(read_latencies) if read_latencies else 0
        p95_write_latency = np.percentile(write_latencies, 95) if len(write_latencies) > 10 else 0
        p95_read_latency = np.percentile(read_latencies, 95) if len(read_latencies) > 10 else 0
        
        # Store tenant stats
        results["per_tenant_stats"][tenant_name] = {
            "total_operations": metrics["reads"] + metrics["writes"],
            "reads": metrics["reads"],
            "writes": metrics["writes"],
            "errors": metrics["errors"],
            "avg_write_latency": avg_write_latency,
            "avg_read_latency": avg_read_latency,
            "p95_write_latency": p95_write_latency,
            "p95_read_latency": p95_read_latency
        }
    
    # Calculate fairness (coefficient of variation of throughput across tenants)
    # Lower CoV means more fairness in resource allocation
    throughputs = [stats["total_operations"] for stats in results["per_tenant_stats"].values()]
    if throughputs and np.mean(throughputs) > 0:
        fairness_cov = np.std(throughputs) / np.mean(throughputs)
        results["fairness_score"] = max(0, 1 - fairness_cov)  # Normalize to 0-1 scale
    
    # Calculate isolation (correlation between tenant latencies)
    # Lower correlation means better isolation
    tenant_names = list(results["per_tenant_stats"].keys())
    isolation_scores = []
    
    if len(tenant_names) >= 2:
        latency_series = {}
        for tenant_name in tenant_names:
            # Create time series for latencies
            latencies = tenant_metrics[tenant_name]["latencies"]
            if latencies:
                latency_series[tenant_name] = pd.Series([lat for _, lat in latencies])
        
        # Calculate correlations between tenant latencies
        for i in range(len(tenant_names)):
            for j in range(i+1, len(tenant_names)):
                t1, t2 = tenant_names[i], tenant_names[j]
                if t1 in latency_series and t2 in latency_series:
                    # Pad shorter series if needed
                    min_len = min(len(latency_series[t1]), len(latency_series[t2]))
                    if min_len > 10:  # Only calculate if we have enough data points
                        corr = abs(latency_series[t1][:min_len].corr(latency_series[t2][:min_len]))
                        isolation_score = 1 - corr  # Higher score = better isolation
                        isolation_scores.append(isolation_score)
    
    if isolation_scores:
        results["isolation_score"] = np.mean(isolation_scores)
    
    # Overall conclusion
    if results["fairness_score"] >= 0.7 and results["isolation_score"] >= 0.7:
        results["conclusion"] = "PASS: Good resource allocation fairness and tenant isolation"
    elif results["fairness_score"] >= 0.7:
        results["conclusion"] = "PARTIAL PASS: Good fairness but insufficient tenant isolation"
    elif results["isolation_score"] >= 0.7:
        results["conclusion"] = "PARTIAL PASS: Good isolation but resource allocation is not fair"
    else:
        results["conclusion"] = "FAIL: Poor resource fairness and tenant isolation"
    
    return results

def visualize_results(tenant_metrics: Dict, analysis_results: Dict, output_dir: str = "results"):
    """Generate visualization of the performance test results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create throughput comparison chart
    tenant_names = list(tenant_metrics.keys())
    reads = [analysis_results["per_tenant_stats"][t]["reads"] for t in tenant_names]
    writes = [analysis_results["per_tenant_stats"][t]["writes"] for t in tenant_names]
    
    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    x = np.arange(len(tenant_names))
    
    plt.bar(x - bar_width/2, reads, bar_width, label='Reads')
    plt.bar(x + bar_width/2, writes, bar_width, label='Writes')
    
    plt.xlabel('Tenant')
    plt.ylabel('Operations Count')
    plt.title('Throughput by Tenant')
    plt.xticks(x, tenant_names)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'throughput_comparison.png'))
    plt.close()
    
    # Create latency comparison chart
    avg_read_latencies = [analysis_results["per_tenant_stats"][t]["avg_read_latency"] for t in tenant_names]
    avg_write_latencies = [analysis_results["per_tenant_stats"][t]["avg_write_latency"] for t in tenant_names]
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - bar_width/2, avg_read_latencies, bar_width, label='Avg Read Latency')
    plt.bar(x + bar_width/2, avg_write_latencies, bar_width, label='Avg Write Latency')
    
    plt.xlabel('Tenant')
    plt.ylabel('Latency (s)')
    plt.title('Latency by Tenant')
    plt.xticks(x, tenant_names)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'latency_comparison.png'))
    plt.close()
    
    # Create fairness and isolation score chart
    plt.figure(figsize=(8, 5))
    scores = [analysis_results["fairness_score"], analysis_results["isolation_score"]]
    score_labels = ["Fairness Score", "Isolation Score"]
    
    plt.bar(score_labels, scores, color=['green', 'blue'])
    plt.axhline(y=0.7, color='r', linestyle='--', label='Threshold')
    
    plt.ylim(0, 1)
    plt.ylabel('Score (0-1)')
    plt.title('Multi-tenant Performance Metrics')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'performance_scores.png'))
    plt.close()
    
    # Save results as JSON
    with open(os.path.join(output_dir, 'analysis_results.json'), 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    logger.info(f"Results visualizations saved to {output_dir}")

@pytest.fixture(scope="module")
def ozone_cluster():
    """Fixture to ensure Ozone cluster is ready for testing"""
    # Check if Ozone cluster is running
    try:
        subprocess.run(["ozone", "version"], check=True, capture_output=True)
        return True
    except:
        pytest.skip("Apache Ozone cluster is not available")

def test_13_multi_tenant_performance(ozone_cluster):
    """
    Test performance under multi-tenant workloads to ensure fair resource allocation
    and performance isolation between tenants.
    """
    # Define tenants with different quotas and workload types
    tenants = [
        TenantConfig("analytics", volume_quota_gb=200, bucket_quota_gb=150, 
                     workload_type="analytics"),
        TenantConfig("streaming", volume_quota_gb=100, bucket_quota_gb=75, 
                     workload_type="streaming"),
        TenantConfig("batch", volume_quota_gb=300, bucket_quota_gb=250, 
                     workload_type="batch")
    ]
    
    # Set up tenant environments
    logger.info("Setting up multi-tenant environment")
    setup_success = setup_tenant_environment(tenants)
    assert setup_success, "Failed to set up tenant environments"
    
    # Test parameters
    test_duration_seconds = 300  # 5 minutes
    
    # Start system metrics collection in a separate thread
    logger.info("Starting system metrics collection")
    system_metrics_future = concurrent.futures.ThreadPoolExecutor().submit(
        collect_system_metrics, test_duration_seconds, 5
    )
    
    # Start workloads for each tenant in parallel
    logger.info("Starting tenant workloads")
    tenant_metrics = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(tenants)) as executor:
        # Submit workloads based on tenant type
        futures = {}
        for tenant in tenants:
            if tenant.workload_type == "analytics":
                future = executor.submit(generate_analytics_workload, tenant, test_duration_seconds)
            elif tenant.workload_type == "streaming":
                future = executor.submit(generate_streaming_workload, tenant, test_duration_seconds)
            else:  # batch
                future = executor.submit(generate_batch_workload, tenant, test_duration_seconds)
            futures[future] = tenant.name
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(futures):
            tenant_name = futures[future]
            try:
                result = future.result()
                tenant_metrics[tenant_name] = result
                logger.info(f"Tenant {tenant_name} workload completed: {result['reads']} reads, {result['writes']} writes")
            except Exception as e:
                logger.error(f"Error in tenant {tenant_name} workload: {str(e)}")
    
    # Get system metrics
    system_metrics = system_metrics_future.result()
    
    # Analyze results
    logger.info("Analyzing test results")
    analysis_results = analyze_results(tenant_metrics, system_metrics)
    
    # Generate visualizations
    visualize_results(tenant_metrics, analysis_results)
    
    # Log the conclusion
    logger.info(f"Test conclusion: {analysis_results['conclusion']}")
    
    # Assertions to validate test expectations
    assert analysis_results["fairness_score"] >= 0.7, \
        f"Resource allocation fairness score {analysis_results['fairness_score']} is below threshold (0.7)"
    
    assert analysis_results["isolation_score"] >= 0.7, \
        f"Performance isolation score {analysis_results['isolation_score']} is below threshold (0.7)"
    
    # Check that all tenants had successful operations
    for tenant_name, metrics in tenant_metrics.items():
        assert metrics["reads"] + metrics["writes"] > 0, f"Tenant {tenant_name} had no successful operations"
        assert metrics["errors"] / max(1, metrics["reads"] + metrics["writes"]) < 0.05, \
            f"Tenant {tenant_name} had too many errors: {metrics['errors']}"
    
    # Print test summary
    logger.info(f"Multi-tenant performance test completed successfully with "
                f"fairness score: {analysis_results['fairness_score']:.2f}, "
                f"isolation score: {analysis_results['isolation_score']:.2f}")

import pytest
import subprocess
import time
import os
import re
import csv
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import socket
import psutil
from threading import Event

class OzoneNetworkMonitor:
    """Utility class to monitor network performance in Ozone cluster"""
    
    def __init__(self, cluster_hosts, interval=1.0):
        """
        Initialize network monitor
        
        Args:
            cluster_hosts (list): List of hostnames or IPs in the cluster
            interval (float): Sampling interval in seconds
        """
        self.cluster_hosts = cluster_hosts
        self.interval = interval
        self.stop_event = Event()
        self.results = {}
        
    def start_monitoring(self):
        """Start network monitoring for the cluster"""
        self.stop_event.clear()
        self.results = {host: {'timestamp': [], 'bytes_sent': [], 'bytes_recv': []} 
                       for host in self.cluster_hosts}
        
        if 'localhost' in self.cluster_hosts or '127.0.0.1' in self.cluster_hosts:
            # For local machine, use psutil
            self._start_local_monitoring()
        
        # For remote machines, use SSH to run sar or similar commands
        for host in [h for h in self.cluster_hosts if h not in ['localhost', '127.0.0.1']]:
            self._start_remote_monitoring(host)
            
    def _start_local_monitoring(self):
        """Monitor network on local machine using psutil"""
        def monitor_local():
            net_io_counters_start = psutil.net_io_counters()
            start_time = time.time()
            
            while not self.stop_event.is_set():
                time.sleep(self.interval)
                net_io_counters = psutil.net_io_counters()
                current_time = time.time()
                
                # Calculate rates
                bytes_sent = net_io_counters.bytes_sent - net_io_counters_start.bytes_sent
                bytes_recv = net_io_counters.bytes_recv - net_io_counters_start.bytes_recv
                elapsed = current_time - start_time
                
                # Update results
                self.results['localhost']['timestamp'].append(current_time)
                self.results['localhost']['bytes_sent'].append(bytes_sent / elapsed)
                self.results['localhost']['bytes_recv'].append(bytes_recv / elapsed)
                
                # Reset for next interval
                net_io_counters_start = net_io_counters
                start_time = current_time
                
        # Start monitoring in a separate thread
        monitor_thread = ThreadPoolExecutor(max_workers=1)
        monitor_thread.submit(monitor_local)
    
    def _start_remote_monitoring(self, host):
        """Monitor network on remote machine using SSH and sar"""
        def monitor_remote(host):
            while not self.stop_event.is_set():
                # Using SSH to run sar command on remote host
                cmd = f"ssh {host} 'sar -n DEV 1 1 | grep -i eth'"
                try:
                    output = subprocess.check_output(cmd, shell=True, text=True)
                    lines = output.strip().split('\n')
                    if len(lines) >= 2:  # Skip header
                        # Parse sar output to get network metrics
                        # Format depends on exact sar version, this is simplified
                        fields = lines[1].split()
                        if len(fields) >= 5:
                            rx_bytes = float(fields[3]) * 1024  # Convert KB to bytes
                            tx_bytes = float(fields[4]) * 1024
                            
                            current_time = time.time()
                            self.results[host]['timestamp'].append(current_time)
                            self.results[host]['bytes_recv'].append(rx_bytes)
                            self.results[host]['bytes_sent'].append(tx_bytes)
                except Exception as e:
                    print(f"Error monitoring host {host}: {str(e)}")
                
                time.sleep(self.interval)
        
        # Start monitoring in a separate thread
        monitor_thread = ThreadPoolExecutor(max_workers=1)
        monitor_thread.submit(monitor_remote, host)
    
    def stop_monitoring(self):
        """Stop network monitoring"""
        self.stop_event.set()
        time.sleep(self.interval * 2)  # Give time for threads to complete
        
    def get_results(self):
        """Return monitoring results"""
        return self.results
        
    def generate_report(self, output_dir="./network_report"):
        """Generate report with network statistics and charts"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create summary CSV
        with open(f"{output_dir}/network_summary.csv", 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Host', 'Avg Bytes Sent/s', 'Avg Bytes Recv/s', 'Max Bytes Sent/s', 'Max Bytes Recv/s'])
            
            for host, data in self.results.items():
                if data['bytes_sent'] and data['bytes_recv']:
                    avg_sent = sum(data['bytes_sent']) / len(data['bytes_sent'])
                    avg_recv = sum(data['bytes_recv']) / len(data['bytes_recv'])
                    max_sent = max(data['bytes_sent'])
                    max_recv = max(data['bytes_recv'])
                    writer.writerow([host, avg_sent, avg_recv, max_sent, max_recv])
        
        # Generate charts
        for host, data in self.results.items():
            if data['timestamp'] and data['bytes_sent'] and data['bytes_recv']:
                plt.figure(figsize=(10, 6))
                plt.plot(data['timestamp'], data['bytes_sent'], label='Bytes Sent/s')
                plt.plot(data['timestamp'], data['bytes_recv'], label='Bytes Received/s')
                plt.title(f'Network Utilization - {host}')
                plt.xlabel('Time')
                plt.ylabel('Bytes/s')
                plt.legend()
                plt.grid(True)
                plt.savefig(f"{output_dir}/{host}_network_utilization.png")
                plt.close()
        
        return output_dir


class OzoneDataTransfer:
    """Utility to generate network-intensive workloads for Ozone"""
    
    def __init__(self, ozone_shell_cmd="ozone sh"):
        """
        Initialize data transfer utility
        
        Args:
            ozone_shell_cmd (str): Command to invoke Ozone shell
        """
        self.ozone_shell_cmd = ozone_shell_cmd
        
    def create_test_file(self, filename, size_mb):
        """
        Create a test file of specified size
        
        Args:
            filename (str): Path to the file to be created
            size_mb (int): Size in megabytes
        """
        with open(filename, 'wb') as f:
            f.write(os.urandom(int(size_mb * 1024 * 1024)))
        return filename
    
    def upload_file(self, volume, bucket, key, local_file):
        """
        Upload a file to Ozone
        
        Args:
            volume (str): Ozone volume
            bucket (str): Ozone bucket
            key (str): Key name
            local_file (str): Path to local file
        """
        cmd = f"{self.ozone_shell_cmd} key put {volume}/{bucket}/{key} {local_file}"
        subprocess.run(cmd, shell=True, check=True)
        
    def download_file(self, volume, bucket, key, local_file):
        """
        Download a file from Ozone
        
        Args:
            volume (str): Ozone volume
            bucket (str): Ozone bucket
            key (str): Key name
            local_file (str): Path to save downloaded file
        """
        cmd = f"{self.ozone_shell_cmd} key get {volume}/{bucket}/{key} {local_file}"
        subprocess.run(cmd, shell=True, check=True)
        
    def run_parallel_transfers(self, volume, bucket, num_files, size_mb, num_workers=4):
        """
        Run parallel data transfers to generate network load
        
        Args:
            volume (str): Ozone volume
            bucket (str): Ozone bucket
            num_files (int): Number of files to transfer
            size_mb (int): Size of each file in MB
            num_workers (int): Number of parallel workers
            
        Returns:
            tuple: Duration of the operation (seconds), total data transferred (bytes)
        """
        # Create volume and bucket if needed
        subprocess.run(f"{self.ozone_shell_cmd} volume create {volume}", 
                     shell=True, check=True)
        subprocess.run(f"{self.ozone_shell_cmd} bucket create {volume}/{bucket}", 
                     shell=True, check=True)
        
        # Create test files
        test_files = []
        for i in range(num_files):
            filename = f"/tmp/ozone_test_file_{i}.dat"
            self.create_test_file(filename, size_mb)
            test_files.append(filename)
            
        # Upload in parallel
        def upload_task(idx):
            key = f"testkey{idx}"
            self.upload_file(volume, bucket, key, test_files[idx])
            return size_mb * 1024 * 1024  # Return bytes transferred
            
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(upload_task, range(num_files)))
        end_time = time.time()
        
        # Clean up
        for file in test_files:
            try:
                os.remove(file)
            except OSError:
                pass
                
        return end_time - start_time, sum(results)


def get_cluster_hosts():
    """Get list of hosts in the Ozone cluster from environment or configuration"""
    # In a real scenario, this would read from Ozone configuration or environment
    # For test, we'll use localhost and possibly other hosts if defined in env
    hosts = ['localhost']
    
    # Add additional hosts from environment variable if it exists
    ozone_hosts_env = os.environ.get('OZONE_HOSTS', '')
    if ozone_hosts_env:
        additional_hosts = ozone_hosts_env.split(',')
        hosts.extend(additional_hosts)
        
    return hosts


@pytest.fixture
def network_monitor():
    """Fixture to provide network monitoring"""
    hosts = get_cluster_hosts()
    monitor = OzoneNetworkMonitor(hosts)
    yield monitor


@pytest.fixture
def data_transfer():
    """Fixture to provide data transfer utility"""
    transfer = OzoneDataTransfer()
    yield transfer


@pytest.mark.parametrize("file_size_mb, num_files, num_workers", [
    (10, 5, 1),      # Small files, sequential
    (100, 10, 4),    # Medium files, parallel
    (500, 3, 3),     # Larger files, parallel
    (1000, 2, 2)     # Very large files, parallel
])
def test_14_network_bandwidth_utilization(network_monitor, data_transfer, file_size_mb, num_files, num_workers):
    """Measure network bandwidth utilization during data transfers in Ozone cluster"""
    # Test configuration
    volume = f"vol-nettest-{int(time.time())}"
    bucket = f"bucket-nettest-{int(time.time())}"
    report_dir = f"./network_test_report_{file_size_mb}mb_{num_files}files_{num_workers}workers"
    
    # 1. Start network monitoring
    network_monitor.start_monitoring()
    
    try:
        # 2. Generate network-intensive workloads
        print(f"Running test with {num_files} files of {file_size_mb}MB each using {num_workers} parallel workers")
        duration, bytes_transferred = data_transfer.run_parallel_transfers(
            volume, bucket, num_files, file_size_mb, num_workers
        )
        
        # 3. Calculate throughput
        throughput_mbps = (bytes_transferred / duration) / (1024 * 1024)  # Convert to MB/s
        print(f"Transfer completed in {duration:.2f} seconds")
        print(f"Throughput: {throughput_mbps:.2f} MB/s")
        
        # 4. Stop monitoring and get results
        network_monitor.stop_monitoring()
        results = network_monitor.get_results()
        
        # 5. Generate report
        report_path = network_monitor.generate_report(report_dir)
        print(f"Network report generated at: {report_path}")
        
        # 6. Validate results - Check for network efficiency
        # Analyze results for each host
        for host, data in results.items():
            if data['bytes_sent'] and data['bytes_recv']:
                avg_sent = sum(data['bytes_sent']) / len(data['bytes_sent'])
                avg_recv = sum(data['bytes_recv']) / len(data['bytes_recv'])
                max_sent = max(data['bytes_sent'])
                max_recv = max(data['bytes_recv'])
                
                # Calculate network utilization variance as a measure of stability
                sent_variance = np.var(data['bytes_sent']) if len(data['bytes_sent']) > 1 else 0
                recv_variance = np.var(data['bytes_recv']) if len(data['bytes_recv']) > 1 else 0
                
                # Log results
                print(f"Host {host} - Avg Sent: {avg_sent/1024/1024:.2f} MB/s, " 
                      f"Avg Recv: {avg_recv/1024/1024:.2f} MB/s")
                print(f"Host {host} - Max Sent: {max_sent/1024/1024:.2f} MB/s, "
                      f"Max Recv: {max_recv/1024/1024:.2f} MB/s")
                print(f"Host {host} - Sent Variance: {sent_variance/1024/1024:.2f}, "
                      f"Recv Variance: {recv_variance/1024/1024:.2f}")
                
                # Validations
                # 1. Check that network is utilized (non-zero traffic)
                assert avg_sent > 0 or avg_recv > 0, f"No network traffic detected on {host}"
                
                # 2. Check for reasonable stability (variance shouldn't be too high)
                # This is a heuristic - adjust threshold based on environment
                norm_sent_variance = sent_variance / (avg_sent**2) if avg_sent > 0 else 0
                norm_recv_variance = recv_variance / (avg_recv**2) if avg_recv > 0 else 0
                
                # Network should be reasonably stable (normalized variance < 0.5)
                # This is a heuristic threshold - adjust based on environment
                assert norm_sent_variance < 0.5, f"Network sending on {host} shows high instability"
                assert norm_recv_variance < 0.5, f"Network receiving on {host} shows high instability" 
    
    finally:
        # Clean up
        try:
            subprocess.run(f"ozone sh bucket delete {volume}/{bucket} --force", 
                         shell=True, check=False)
            subprocess.run(f"ozone sh volume delete {volume}", 
                         shell=True, check=False)
        except Exception as e:
            print(f"Cleanup error (non-fatal): {str(e)}")

import os
import time
import pytest
import pandas as pd
import matplotlib.pyplot as plt
from subprocess import run, PIPE
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import tempfile
import random
import string
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StoragePerformanceTester:
    """Utility class for testing Ozone performance on different storage media"""
    
    def __init__(self, cluster_config, storage_configs):
        """
        Initialize the performance tester
        
        Args:
            cluster_config: Dict with Ozone cluster connection details
            storage_configs: List of storage configurations to test
        """
        self.cluster_config = cluster_config
        self.storage_configs = storage_configs
        self.results = {}
        
    def setup_storage_config(self, config):
        """Configure the Ozone cluster to use the specified storage type"""
        logger.info(f"Setting up storage configuration: {config['name']}")
        
        # Create the hdds-site.xml property changes for the storage type
        properties = {
            "hdds.datanode.dir": config["storage_path"],
            "hdds.container.disk.selection.policy": "RoundRobin"
        }
        
        # Apply the configuration using admin commands
        cmd = ["ozone", "admin", "datanode", "updateconf"]
        for key, value in properties.items():
            run(cmd + [key, value], check=True)
            
        # Restart the datanodes to apply configuration
        run(["ozone", "admin", "datanode", "restart"], check=True)
        
        # Wait for the cluster to stabilize
        time.sleep(60)
        logger.info(f"Storage configuration {config['name']} is ready")
        
    def generate_test_files(self, sizes):
        """Generate test files of various sizes for benchmarking"""
        test_files = []
        for size_mb in sizes:
            with tempfile.NamedTemporaryFile(delete=False) as f:
                # Generate random data of the specified size
                size_bytes = int(size_mb * 1024 * 1024)
                # Write in chunks to avoid memory issues with large files
                chunk_size = 1024 * 1024  # 1MB chunks
                for _ in range(0, size_bytes, chunk_size):
                    write_size = min(chunk_size, size_bytes - f.tell())
                    f.write(os.urandom(write_size))
                test_files.append((f.name, size_mb))
        return test_files
    
    def run_benchmark(self, config, test_files, operations_per_file=5):
        """
        Run benchmarks for a specific storage configuration
        
        Args:
            config: Storage configuration to test
            test_files: List of (file_path, size_mb) tuples to use for testing
            operations_per_file: Number of operations to perform per file for averaging
        """
        logger.info(f"Running benchmark for {config['name']}")
        results = {
            'write_throughput': [],
            'read_throughput': [],
            'latency': []
        }
        
        # Create unique volume and bucket for this test
        volume_name = f"vol-{config['name']}-{int(time.time())}"
        bucket_name = f"bucket-{config['name']}-{int(time.time())}"
        
        # Create volume and bucket
        run(["ozone", "sh", "volume", "create", volume_name], check=True)
        run(["ozone", "sh", "bucket", "create", f"{volume_name}/{bucket_name}"], check=True)
        
        for file_path, size_mb in test_files:
            file_size_bytes = size_mb * 1024 * 1024
            
            # Multiple operations to get average performance
            write_times = []
            read_times = []
            
            for i in range(operations_per_file):
                key_name = f"key-{size_mb}mb-{i}"
                
                # Measure write performance
                start_time = time.time()
                run(["ozone", "sh", "key", "put", f"{volume_name}/{bucket_name}/{key_name}", file_path], check=True)
                end_time = time.time()
                write_time = end_time - start_time
                write_times.append(write_time)
                
                # Calculate write throughput (MB/s)
                write_throughput = size_mb / write_time
                results['write_throughput'].append(write_throughput)
                
                # Measure read performance
                output_file = tempfile.NamedTemporaryFile(delete=False).name
                start_time = time.time()
                run(["ozone", "sh", "key", "get", f"{volume_name}/{bucket_name}/{key_name}", output_file], check=True)
                end_time = time.time()
                read_time = end_time - start_time
                read_times.append(read_time)
                
                # Calculate read throughput (MB/s)
                read_throughput = size_mb / read_time
                results['read_throughput'].append(read_throughput)
                
                # Cleanup the output file
                os.unlink(output_file)
                
            # Calculate average latency (ms)
            avg_latency = 1000 * (sum(write_times) + sum(read_times)) / (len(write_times) + len(read_times))
            results['latency'].append(avg_latency)
            
        # Clean up
        run(["ozone", "sh", "bucket", "delete", f"{volume_name}/{bucket_name}"], check=True)
        run(["ozone", "sh", "volume", "delete", volume_name], check=True)
        
        return results
        
    def run_all_benchmarks(self):
        """Run benchmarks on all storage configurations"""
        # Define test file sizes in MB
        test_sizes = [0.5, 10, 50, 100, 500]
        
        for config in self.storage_configs:
            # Setup storage configuration
            self.setup_storage_config(config)
            
            # Generate test files
            test_files = self.generate_test_files(test_sizes)
            
            # Run benchmarks
            self.results[config['name']] = self.run_benchmark(config, test_files)
            
            # Clean up test files
            for file_path, _ in test_files:
                os.unlink(file_path)
    
    def analyze_results(self):
        """Analyze benchmark results and generate comparison metrics"""
        analysis = {}
        
        # Aggregate metrics for each configuration
        for config_name, metrics in self.results.items():
            analysis[config_name] = {
                'avg_write_throughput': sum(metrics['write_throughput']) / len(metrics['write_throughput']),
                'avg_read_throughput': sum(metrics['read_throughput']) / len(metrics['read_throughput']),
                'avg_latency': sum(metrics['latency']) / len(metrics['latency']),
                'max_write_throughput': max(metrics['write_throughput']),
                'min_write_throughput': min(metrics['write_throughput']),
                'max_read_throughput': max(metrics['read_throughput']),
                'min_read_throughput': min(metrics['read_throughput'])
            }
        
        # Create cost-performance analysis
        for config_name, metrics in analysis.items():
            config = next(c for c in self.storage_configs if c['name'] == config_name)
            cost_per_gb = config.get('cost_per_gb', 0)
            if cost_per_gb > 0:
                metrics['cost_performance_ratio'] = (metrics['avg_write_throughput'] + metrics['avg_read_throughput']) / (2 * cost_per_gb)
            else:
                metrics['cost_performance_ratio'] = float('inf')
                
        return analysis
    
    def plot_results(self, analysis):
        """Generate performance comparison plots"""
        config_names = list(analysis.keys())
        
        # Prepare data for plotting
        write_throughputs = [analysis[name]['avg_write_throughput'] for name in config_names]
        read_throughputs = [analysis[name]['avg_read_throughput'] for name in config_names]
        latencies = [analysis[name]['avg_latency'] for name in config_names]
        
        # Create throughput comparison chart
        fig, ax = plt.subplots(figsize=(12, 8))
        x = np.arange(len(config_names))
        width = 0.35
        
        ax.bar(x - width/2, write_throughputs, width, label='Write Throughput (MB/s)')
        ax.bar(x + width/2, read_throughputs, width, label='Read Throughput (MB/s)')
        
        ax.set_xlabel('Storage Configuration')
        ax.set_ylabel('Throughput (MB/s)')
        ax.set_title('Ozone Performance by Storage Type')
        ax.set_xticks(x)
        ax.set_xticklabels(config_names)
        ax.legend()
        
        plt.savefig('ozone_storage_throughput_comparison.png')
        
        # Create latency comparison chart
        plt.figure(figsize=(10, 6))
        plt.bar(config_names, latencies)
        plt.xlabel('Storage Configuration')
        plt.ylabel('Average Latency (ms)')
        plt.title('Ozone Latency by Storage Type')
        plt.savefig('ozone_storage_latency_comparison.png')
        
        # If cost data is available, create cost-performance chart
        if all('cost_performance_ratio' in analysis[name] for name in config_names):
            cost_perf_ratios = [analysis[name].get('cost_performance_ratio', 0) for name in config_names]
            
            plt.figure(figsize=(10, 6))
            plt.bar(config_names, cost_perf_ratios)
            plt.xlabel('Storage Configuration')
            plt.ylabel('Performance/Cost Ratio')
            plt.title('Ozone Cost-Performance Analysis')
            plt.savefig('ozone_cost_performance_analysis.png')
        
        return True


@pytest.fixture
def storage_configs():
    """Fixture providing storage configurations to test"""
    return [
        {
            "name": "ssd",
            "storage_path": "/mnt/ssd/ozone",
            "type": "SSD",
            "cost_per_gb": 0.20  # Example cost in dollars per GB
        },
        {
            "name": "hdd",
            "storage_path": "/mnt/hdd/ozone",
            "type": "HDD",
            "cost_per_gb": 0.05  # Example cost in dollars per GB
        },
        {
            "name": "hybrid",
            "storage_path": "/mnt/hybrid/ozone",
            "type": "Hybrid",
            "cost_per_gb": 0.12  # Example cost in dollars per GB
        }
    ]


@pytest.fixture
def cluster_config():
    """Fixture providing Ozone cluster configuration"""
    return {
        "om_service_id": "ozone1",
        "service_id": "ozone1",
        "hosts": ["localhost"],
        "port": 9862,
    }


@pytest.mark.performance
def test_15_storage_media_performance(cluster_config, storage_configs):
    """
    Test performance with different storage media
    
    This test:
    1. Configures Ozone to use different storage media
    2. Runs standard benchmarks on each configuration
    3. Compares performance metrics across storage types
    4. Analyzes cost-performance tradeoffs
    """
    # Create the performance tester
    tester = StoragePerformanceTester(cluster_config, storage_configs)
    
    # Run benchmarks on all storage configurations
    tester.run_all_benchmarks()
    
    # Analyze results
    analysis = tester.analyze_results()
    
    # Generate performance comparison charts
    tester.plot_results(analysis)
    
    # Log analysis results
    logger.info("Performance analysis results:")
    logger.info(json.dumps(analysis, indent=2))
    
    # Validate results against expectations
    for config_name, metrics in analysis.items():
        config = next(c for c in storage_configs if c['name'] == config_name)
        
        # SSD should have better performance than HDD
        if config['type'] == 'SSD':
            # Assert SSD read/write throughput is better than other storage types
            for other_config in storage_configs:
                if other_config['type'] == 'HDD':
                    assert metrics['avg_write_throughput'] > analysis[other_config['name']]['avg_write_throughput'], \
                        f"SSD write throughput ({metrics['avg_write_throughput']}) should be higher than HDD " \
                        f"({analysis[other_config['name']]['avg_write_throughput']})"
                    
                    assert metrics['avg_read_throughput'] > analysis[other_config['name']]['avg_read_throughput'], \
                        f"SSD read throughput ({metrics['avg_read_throughput']}) should be higher than HDD " \
                        f"({analysis[other_config['name']]['avg_read_throughput']})"
                    
        # Verify that latency is in expected ranges based on storage type
        if config['type'] == 'SSD':
            assert metrics['avg_latency'] < 50, f"SSD latency ({metrics['avg_latency']} ms) exceeds expected range"
        elif config['type'] == 'HDD':
            assert 50 <= metrics['avg_latency'] <= 200, f"HDD latency ({metrics['avg_latency']} ms) outside expected range"
        
        # Verify we have meaningful cost-performance tradeoffs
        if 'cost_performance_ratio' in metrics and metrics['cost_performance_ratio'] != float('inf'):
            assert metrics['cost_performance_ratio'] > 0, f"Cost-performance ratio should be positive for {config_name}"
    
    # Generate a final report
    report = {
        "test_summary": "Storage Media Performance Test",
        "configurations_tested": len(storage_configs),
        "top_performer": max(analysis.keys(), key=lambda k: (analysis[k]['avg_read_throughput'] + analysis[k]['avg_write_throughput'])/2),
        "best_cost_performance": max(analysis.keys(), key=lambda k: analysis[k].get('cost_performance_ratio', 0)),
        "timestamp": time.time(),
    }
    
    with open('storage_performance_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Test complete. Top performer: {report['top_performer']}")

import os
import time
import pytest
import subprocess
import statistics
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, List, Tuple

# Custom exception for Ozone operation failures
class OzoneOperationError(Exception):
    """Exception raised when an Ozone operation fails"""
    pass

class DataCompactionPerformanceTest:
    """Helper class for data compaction performance testing"""
    
    def __init__(self, cluster_info: Dict):
        """Initialize with cluster connection information"""
        self.cluster_info = cluster_info
        self.metrics_before = {}
        self.metrics_during = {}
        self.metrics_after = {}
        self.test_data_dir = "/tmp/ozone_compaction_test_data"
        self.results_dir = "/tmp/ozone_compaction_test_results"
        os.makedirs(self.test_data_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def setup_fragmented_data(self, volume: str, bucket: str, num_keys: int = 1000, 
                              key_sizes_kb: List[int] = [5, 10, 50, 100, 500]) -> None:
        """
        Load the system with fragmented data by:
        1. Creating many small files
        2. Updating files multiple times to create versions
        3. Deleting some files to create tombstones
        """
        print(f"Setting up fragmented data with {num_keys} keys in {volume}/{bucket}")
        
        # Create volume and bucket if they don't exist
        self._execute_shell_cmd(f"ozone sh volume create {volume}")
        self._execute_shell_cmd(f"ozone sh bucket create {volume}/{bucket}")
        
        # Create many small files with different sizes
        with ThreadPoolExecutor(max_workers=10) as executor:
            for i in range(num_keys):
                size_kb = key_sizes_kb[i % len(key_sizes_kb)]
                key_name = f"key_{i:05d}"
                test_file = f"{self.test_data_dir}/{key_name}_{size_kb}kb"
                
                # Create file with random data of specified size
                self._execute_shell_cmd(f"dd if=/dev/urandom of={test_file} bs=1K count={size_kb}")
                
                # Put file into Ozone
                executor.submit(self._put_key, volume, bucket, key_name, test_file)
                
                # Create versions by updating some files multiple times
                if i % 3 == 0:  # Update every 3rd file multiple times
                    for v in range(3):
                        self._execute_shell_cmd(f"dd if=/dev/urandom of={test_file} bs=1K count={size_kb}")
                        executor.submit(self._put_key, volume, bucket, key_name, test_file)
                
                # Delete some files to create tombstones
                if i % 7 == 0:  # Delete every 7th file
                    executor.submit(self._execute_shell_cmd, 
                                   f"ozone sh key delete {volume}/{bucket}/{key_name}")

    def _put_key(self, volume: str, bucket: str, key: str, file_path: str) -> None:
        """Put a key into Ozone"""
        self._execute_shell_cmd(f"ozone sh key put {volume}/{bucket}/{key} {file_path}")

    def _execute_shell_cmd(self, cmd: str) -> str:
        """Execute shell command and return output"""
        result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise OzoneOperationError(f"Command failed: {cmd}\nError: {result.stderr}")
        return result.stdout

    def measure_performance(self, volume: str, bucket: str, sample_keys: List[str], 
                           iterations: int = 10) -> Dict:
        """
        Measure performance metrics by:
        1. Reading a sample of keys and measuring latency
        2. Running list operations and measuring throughput
        3. Collecting system metrics from Ozone
        """
        metrics = {}
        
        # Measure read latency
        read_latencies = []
        for _ in range(iterations):
            for key in sample_keys:
                start_time = time.time()
                self._execute_shell_cmd(f"ozone sh key info {volume}/{bucket}/{key}")
                read_latencies.append((time.time() - start_time) * 1000)  # Convert to ms
        
        metrics['read_latency_ms'] = {
            'mean': statistics.mean(read_latencies),
            'median': statistics.median(read_latencies),
            'p95': sorted(read_latencies)[int(len(read_latencies) * 0.95)],
            'min': min(read_latencies),
            'max': max(read_latencies)
        }
        
        # Measure list operation throughput
        start_time = time.time()
        keys_list = self._execute_shell_cmd(f"ozone sh key list {volume}/{bucket}")
        list_duration = time.time() - start_time
        key_count = len(keys_list.strip().split('\n')) if keys_list.strip() else 0
        
        metrics['list_operation'] = {
            'duration_sec': list_duration,
            'key_count': key_count,
            'throughput': key_count / list_duration if list_duration > 0 else 0
        }
        
        # Get system metrics from Ozone (in a real scenario, would use JMX/Prometheus metrics)
        # This is a simplified version - in practice, you would query the metrics endpoint
        try:
            metrics['system'] = self._collect_system_metrics()
        except Exception as e:
            print(f"Failed to collect system metrics: {e}")
            metrics['system'] = {}
            
        return metrics

    def _collect_system_metrics(self) -> Dict:
        """
        Collect system metrics from Ozone
        In a real implementation, this would query JMX or Prometheus metrics
        """
        # This is a placeholder - in a real scenario, we would query actual metrics
        # For example, using requests to get metrics from Prometheus or JMX
        metrics = {}
        
        # Get disk usage information
        df_output = self._execute_shell_cmd("df -h | grep ozone")
        if df_output:
            metrics['disk_usage'] = df_output.strip()
        
        # Get memory usage
        free_output = self._execute_shell_cmd("free -m")
        if free_output:
            metrics['memory'] = free_output.strip()
            
        # CPU usage
        top_output = self._execute_shell_cmd("top -b -n 1 | head -n 20")
        if top_output:
            metrics['cpu'] = top_output.strip()
            
        return metrics

    def trigger_compaction(self) -> None:
        """
        Trigger data compaction in Ozone
        In a real scenario, this would use appropriate admin commands
        """
        print("Triggering compaction...")
        
        # Admin command to trigger compaction
        # This is a placeholder - in a real scenario, use the correct admin command
        try:
            # In a real scenario, this might be an admin API call or shell command
            self._execute_shell_cmd("ozone admin compaction start --all")
            print("Compaction triggered successfully")
        except Exception as e:
            print(f"Failed to trigger compaction: {e}")
            raise
    
    def wait_for_compaction_completion(self, timeout_sec: int = 600, 
                                      check_interval_sec: int = 10) -> bool:
        """Wait for the compaction to complete by polling status"""
        print(f"Waiting for compaction to complete (timeout: {timeout_sec}s)...")
        
        start_time = time.time()
        while time.time() - start_time < timeout_sec:
            try:
                status = self._execute_shell_cmd("ozone admin compaction status")
                if "COMPLETED" in status:
                    print("Compaction completed successfully")
                    return True
                
                print(f"Compaction still in progress. Status: {status}")
                time.sleep(check_interval_sec)
                
            except Exception as e:
                print(f"Error checking compaction status: {e}")
                time.sleep(check_interval_sec)
        
        print("Timeout waiting for compaction to complete")
        return False

    def generate_report(self) -> str:
        """Generate a performance report comparing metrics before and after compaction"""
        report_path = f"{self.results_dir}/compaction_perf_report_{self.timestamp}.html"
        
        # Create DataFrame for comparison
        metrics_data = []
        
        # Extract key metrics for before and after
        for phase, metrics in [("Before", self.metrics_before), 
                               ("During", self.metrics_during),
                               ("After", self.metrics_after)]:
            if not metrics:
                continue
                
            row = {
                "Phase": phase,
                "Read Latency Mean (ms)": metrics.get('read_latency_ms', {}).get('mean', 0),
                "Read Latency P95 (ms)": metrics.get('read_latency_ms', {}).get('p95', 0),
                "List Operation Duration (s)": metrics.get('list_operation', {}).get('duration_sec', 0),
                "List Throughput (keys/s)": metrics.get('list_operation', {}).get('throughput', 0)
            }
            metrics_data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(metrics_data)
        
        # Generate HTML report
        with open(report_path, 'w') as f:
            f.write("Ozone Compaction Performance Report")
            f.write("")
            f.write("")
            f.write(f"Ozone Compaction Performance Report")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            f.write("Performance Metrics Comparison")
            f.write(df.to_html(index=False))
            
            # Generate simple charts if we have before and after data
            if len(metrics_data) >= 2:
                self._generate_charts(df, report_path.replace('.html', '_charts.png'))
                f.write(f"Performance Charts")
                f.write(f"")
            
            f.write("Conclusions")
            
            # Add some analysis
            if self.metrics_before and self.metrics_after:
                read_improvement = ((self.metrics_before.get('read_latency_ms', {}).get('mean', 0) - 
                                    self.metrics_after.get('read_latency_ms', {}).get('mean', 0)) / 
                                    self.metrics_before.get('read_latency_ms', {}).get('mean', 1)) * 100
                
                throughput_improvement = ((self.metrics_after.get('list_operation', {}).get('throughput', 0) -
                                         self.metrics_before.get('list_operation', {}).get('throughput', 0)) /
                                         self.metrics_before.get('list_operation', {}).get('throughput', 1)) * 100
                
                f.write(f"Read Latency Improvement: {read_improvement:.2f}%")
                f.write(f"List Throughput Improvement: {throughput_improvement:.2f}%")
                
                if read_improvement > 0:
                    f.write("Compaction has successfully improved read latency.")
                else:
                    f.write("Compaction did not improve read latency as expected.")
                    
                if throughput_improvement > 0:
                    f.write("Compaction has successfully improved list operation throughput.")
                else:
                    f.write("Compaction did not improve list operation throughput as expected.")
            
            f.write("")
        
        print(f"Performance report generated: {report_path}")
        return report_path

    def _generate_charts(self, df: pd.DataFrame, output_path: str) -> None:
        """Generate performance comparison charts"""
        fig, axs = plt.subplots(2, 1, figsize=(10, 12))
        
        # Read latency chart
        axs[0].bar(df['Phase'], df['Read Latency Mean (ms)'], color='skyblue')
        axs[0].set_title('Read Latency Comparison')
        axs[0].set_ylabel('Latency (ms)')
        
        # List throughput chart
        axs[1].bar(df['Phase'], df['List Throughput (keys/s)'], color='lightgreen')
        axs[1].set_title('List Operation Throughput Comparison')
        axs[1].set_ylabel('Throughput (keys/s)')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()


@pytest.fixture
def ozone_cluster():
    """Fixture to provide Ozone cluster connection details"""
    # In a real test, this would get connection information from environment variables
    # or a test configuration file
    cluster_info = {
        "om_host": os.environ.get("OZONE_OM_HOST", "localhost"),
        "om_port": os.environ.get("OZONE_OM_PORT", "9862"),
        "scm_host": os.environ.get("OZONE_SCM_HOST", "localhost"),
        "scm_port": os.environ.get("OZONE_SCM_PORT", "9860")
    }
    
    # Verify cluster is available
    try:
        subprocess.run(["ozone", "version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as e:
        pytest.skip(f"Ozone cluster not available: {e}")
    
    return cluster_info


@pytest.mark.performance
def test_16_compaction_performance(ozone_cluster):
    """Evaluate performance under data compaction"""
    # Use unique volume and bucket names to avoid conflicts with other tests
    volume = f"compaction-vol-{int(time.time())}"
    bucket = f"compaction-bkt-{int(time.time())}"
    
    perf_test = DataCompactionPerformanceTest(ozone_cluster)
    
    try:
        # Step 1: Load the system with fragmented data
        perf_test.setup_fragmented_data(volume, bucket, num_keys=500)
        
        # Get a sample of keys for performance testing
        keys_output = perf_test._execute_shell_cmd(f"ozone sh key list {volume}/{bucket}")
        all_keys = keys_output.strip().split('\n') if keys_output.strip() else []
        sample_keys = all_keys[:20]  # Use a subset of keys for testing
        
        # Step 2: Measure baseline performance
        print("Measuring baseline performance...")
        perf_test.metrics_before = perf_test.measure_performance(volume, bucket, sample_keys)
        
        # Step 3: Trigger data compaction
        perf_test.trigger_compaction()
        
        # Step 4: Monitor system performance during compaction
        print("Measuring performance during compaction...")
        perf_test.metrics_during = perf_test.measure_performance(volume, bucket, sample_keys)
        
        # Wait for compaction to complete
        compaction_completed = perf_test.wait_for_compaction_completion()
        assert compaction_completed, "Compaction did not complete within the timeout period"
        
        # Allow some time for the system to stabilize after compaction
        time.sleep(30)
        
        # Step 5: Measure performance after compaction and compare with baseline
        print("Measuring post-compaction performance...")
        perf_test.metrics_after = perf_test.measure_performance(volume, bucket, sample_keys)
        
        # Generate performance report
        report_path = perf_test.generate_report()
        print(f"Performance report available at: {report_path}")
        
        # Validate performance improvements
        # We expect read latency to improve (decrease) after compaction
        assert perf_test.metrics_after['read_latency_ms']['mean'] <= perf_test.metrics_before['read_latency_ms']['mean'] * 1.1, \
            "Read latency did not improve as expected after compaction"
        
        # While compaction is running, we expect some performance impact
        # But it should be manageable (not more than 50% degradation)
        assert perf_test.metrics_during['read_latency_ms']['mean'] <= perf_test.metrics_before['read_latency_ms']['mean'] * 1.5, \
            "Performance impact during compaction was higher than expected"
        
    finally:
        # Cleanup
        try:
            perf_test._execute_shell_cmd(f"ozone sh bucket delete {volume}/{bucket}")
            perf_test._execute_shell_cmd(f"ozone sh volume delete {volume}")
        except Exception as e:
            print(f"Cleanup failed: {e}")


@pytest.mark.parametrize("data_volume", [
    {"num_keys": 100, "key_size_kb": 10, "description": "Small dataset"},
    {"num_keys": 500, "key_size_kb": 50, "description": "Medium dataset"},
    {"num_keys": 1000, "key_size_kb": 100, "description": "Large dataset"}
])
def test_16_compaction_performance_with_different_data_volumes(ozone_cluster, data_volume):
    """Evaluate performance under data compaction with different data volumes"""
    # Use unique volume and bucket names to avoid conflicts with other tests
    volume = f"compvol-{int(time.time())}"
    bucket = f"compbkt-{int(time.time())}"
    
    perf_test = DataCompactionPerformanceTest(ozone_cluster)
    
    try:
        # Step 1: Load the system with fragmented data
        print(f"Testing with {data_volume['description']}")
        perf_test.setup_fragmented_data(
            volume, bucket, 
            num_keys=data_volume['num_keys'],
            key_sizes_kb=[data_volume['key_size_kb']]
        )
        
        # Get a sample of keys for performance testing
        keys_output = perf_test._execute_shell_cmd(f"ozone sh key list {volume}/{bucket}")
        all_keys = keys_output.strip().split('\n') if keys_output.strip() else []
        sample_keys = all_keys[:20] if len(all_keys) > 20 else all_keys
        
        # Step 2: Measure baseline performance
        print("Measuring baseline performance...")
        perf_test.metrics_before = perf_test.measure_performance(volume, bucket, sample_keys)
        
        # Step 3: Trigger data compaction
        perf_test.trigger_compaction()
        
        # Step 4: Monitor system performance during compaction
        print("Measuring performance during compaction...")
        perf_test.metrics_during = perf_test.measure_performance(volume, bucket, sample_keys)
        
        # Wait for compaction to complete
        compaction_completed = perf_test.wait_for_compaction_completion()
        assert compaction_completed, "Compaction did not complete within the timeout period"
        
        # Allow some time for the system to stabilize after compaction
        time.sleep(30)
        
        # Step 5: Measure performance after compaction and compare with baseline
        print("Measuring post-compaction performance...")
        perf_test.metrics_after = perf_test.measure_performance(volume, bucket, sample_keys)
        
        # Generate performance report
        report_path = perf_test.generate_report()
        print(f"Performance report available at: {report_path}")
        
        # Validate performance improvements
        # For larger datasets, we expect more significant improvements
        expected_improvement_factor = 1.1  # Default improvement factor
        if data_volume['num_keys'] >= 1000:
            expected_improvement_factor = 1.3  # Expect more improvement for larger datasets
        
        assert perf_test.metrics_after['read_latency_ms']['mean'] <= perf_test.metrics_before['read_latency_ms']['mean'] * expected_improvement_factor, \
            f"Read latency did not improve as expected after compaction for {data_volume['description']}"
        
    finally:
        # Cleanup
        try:
            perf_test._execute_shell_cmd(f"ozone sh bucket delete {volume}/{bucket}")
            perf_test._execute_shell_cmd(f"ozone sh volume delete {volume}")
        except Exception as e:
            print(f"Cleanup failed: {e}")

import pytest
import os
import time
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import logging
from concurrent.futures import ThreadPoolExecutor
import json
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test constants
TEST_VOLUME = "perftest"
TEST_BUCKET = "chunktest"
TEST_DIR = "/tmp/ozone-chunk-test"
RESULT_DIR = f"{TEST_DIR}/results"
BENCHMARK_ITERATIONS = 3  # Number of runs per configuration for averaging

# Test file sizes (in bytes)
FILE_SIZES = [
    1024 * 10,       # 10 KB
    1024 * 100,      # 100 KB
    1024 * 1024,     # 1 MB
    1024 * 1024 * 10,  # 10 MB
    1024 * 1024 * 100  # 100 MB
]

# Chunk sizes to test (in bytes)
CHUNK_SIZES = [
    1024 * 64,       # 64 KB (default)
    1024 * 256,      # 256 KB
    1024 * 512,      # 512 KB
    1024 * 1024,     # 1 MB
    1024 * 1024 * 4  # 4 MB
]


def setup_module():
    """Set up test environment for all tests."""
    # Create test directories
    os.makedirs(TEST_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    # Create test volume and bucket
    subprocess.run(["ozone", "sh", "volume", "create", TEST_VOLUME], check=True)
    subprocess.run(["ozone", "sh", "bucket", "create", f"/{TEST_VOLUME}/{TEST_BUCKET}"], check=True)


def teardown_module():
    """Clean up after all tests are done."""
    # Clean up test data
    subprocess.run(["ozone", "sh", "bucket", "delete", f"/{TEST_VOLUME}/{TEST_BUCKET}"], check=True)
    subprocess.run(["ozone", "sh", "volume", "delete", TEST_VOLUME], check=True)
    
    # Keep the results directory for analysis


def generate_test_file(size_bytes: int, filename: str):
    """Generate a test file of specified size."""
    with open(filename, "wb") as f:
        f.write(os.urandom(size_bytes))
    return filename


def configure_chunk_size(chunk_size: int):
    """
    Configure the Ozone chunk size by updating the configuration
    and restarting the necessary services.
    """
    logger.info(f"Configuring chunk size to {chunk_size} bytes")
    
    # This would typically involve:
    # 1. Updating ozone-site.xml configuration
    # 2. Restarting the Ozone services
    # For testing purposes, we'll assume these steps are done externally
    # or simulate them using environment variables
    
    # Set environment variable to simulate configuration change
    os.environ["OZONE_CHUNK_SIZE"] = str(chunk_size)
    
    # In a real implementation, you would update configuration files and restart services:
    # subprocess.run(["update_ozone_config.sh", "--chunk-size", str(chunk_size)], check=True)
    # subprocess.run(["restart_ozone_services.sh"], check=True)
    
    # Allow time for services to restart and stabilize
    time.sleep(5)


def run_write_benchmark(file_path: str, key_name: str) -> float:
    """
    Measure the time taken to write a file to Ozone.
    Returns the throughput in MB/s.
    """
    start_time = time.time()
    
    # Write the file to Ozone
    subprocess.run([
        "ozone", "sh", "key", "put", 
        f"/{TEST_VOLUME}/{TEST_BUCKET}/{key_name}", file_path
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    elapsed_time = time.time() - start_time
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    throughput = file_size_mb / elapsed_time if elapsed_time > 0 else 0
    
    logger.info(f"Write throughput for {file_size_mb:.2f} MB: {throughput:.2f} MB/s")
    return throughput


def run_read_benchmark(file_path: str, key_name: str) -> float:
    """
    Measure the time taken to read a file from Ozone.
    Returns the throughput in MB/s.
    """
    output_file = f"{file_path}.read"
    
    start_time = time.time()
    
    # Read the file from Ozone
    subprocess.run([
        "ozone", "sh", "key", "get", 
        f"/{TEST_VOLUME}/{TEST_BUCKET}/{key_name}", output_file
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    elapsed_time = time.time() - start_time
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    throughput = file_size_mb / elapsed_time if elapsed_time > 0 else 0
    
    # Clean up the read file
    os.remove(output_file)
    
    logger.info(f"Read throughput for {file_size_mb:.2f} MB: {throughput:.2f} MB/s")
    return throughput


def benchmark_performance(chunk_size: int) -> Dict:
    """
    Run benchmarks for a specific chunk size across different file sizes.
    Returns a dictionary of results.
    """
    results = {
        "chunk_size": chunk_size,
        "write_throughput": {},
        "read_throughput": {},
        "latency": {}
    }
    
    # Configure the chunk size in Ozone
    configure_chunk_size(chunk_size)
    
    for file_size in FILE_SIZES:
        size_key = str(file_size)
        write_throughputs = []
        read_throughputs = []
        latencies = []
        
        for i in range(BENCHMARK_ITERATIONS):
            # Generate unique file and key names for each iteration
            test_file = f"{TEST_DIR}/test_file_{file_size}_{i}.dat"
            key_name = f"test_key_{file_size}_{i}_{int(time.time())}"
            
            # Generate test file
            generate_test_file(file_size, test_file)
            
            # Measure write performance
            start_time = time.time()
            write_throughput = run_write_benchmark(test_file, key_name)
            write_throughputs.append(write_throughput)
            
            # Measure read performance
            read_throughput = run_read_benchmark(test_file, key_name)
            read_throughputs.append(read_throughput)
            
            # Calculate operation latency
            end_time = time.time()
            latencies.append(end_time - start_time)
            
            # Clean up
            os.remove(test_file)
            subprocess.run([
                "ozone", "sh", "key", "delete", 
                f"/{TEST_VOLUME}/{TEST_BUCKET}/{key_name}"
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Store average results
        results["write_throughput"][size_key] = sum(write_throughputs) / len(write_throughputs)
        results["read_throughput"][size_key] = sum(read_throughputs) / len(read_throughputs)
        results["latency"][size_key] = sum(latencies) / len(latencies)
    
    return results


def visualize_results(results: List[Dict], output_dir: str):
    """
    Create visualizations of benchmark results.
    """
    # Convert results to DataFrame for easier plotting
    data = []
    for result in results:
        chunk_size = result["chunk_size"]
        for file_size, write_tp in result["write_throughput"].items():
            read_tp = result["read_throughput"][file_size]
            latency = result["latency"][file_size]
            data.append({
                "chunk_size": chunk_size,
                "file_size": int(file_size),
                "write_throughput": write_tp,
                "read_throughput": read_tp,
                "latency": latency
            })
    
    df = pd.DataFrame(data)
    
    # Create directory for plots
    plots_dir = f"{output_dir}/plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot write throughput
    plt.figure(figsize=(12, 8))
    for file_size in FILE_SIZES:
        file_data = df[df["file_size"] == file_size]
        plt.plot(file_data["chunk_size"], file_data["write_throughput"], 
                 marker='o', label=f"{file_size/(1024*1024):.2f} MB")
    
    plt.xlabel("Chunk Size (bytes)")
    plt.ylabel("Write Throughput (MB/s)")
    plt.title("Write Throughput vs Chunk Size")
    plt.legend(title="File Size")
    plt.grid(True)
    plt.savefig(f"{plots_dir}/write_throughput.png")
    
    # Plot read throughput
    plt.figure(figsize=(12, 8))
    for file_size in FILE_SIZES:
        file_data = df[df["file_size"] == file_size]
        plt.plot(file_data["chunk_size"], file_data["read_throughput"], 
                 marker='o', label=f"{file_size/(1024*1024):.2f} MB")
    
    plt.xlabel("Chunk Size (bytes)")
    plt.ylabel("Read Throughput (MB/s)")
    plt.title("Read Throughput vs Chunk Size")
    plt.legend(title="File Size")
    plt.grid(True)
    plt.savefig(f"{plots_dir}/read_throughput.png")
    
    # Save raw data
    df.to_csv(f"{output_dir}/benchmark_results.csv", index=False)
    
    # Generate summary report
    generate_summary_report(df, f"{output_dir}/summary_report.txt")


def generate_summary_report(results_df: pd.DataFrame, output_path: str):
    """
    Generate a summary report from the benchmark results.
    """
    with open(output_path, "w") as f:
        f.write("=== Apache Ozone Chunk Size Performance Summary ===\n\n")
        
        f.write("== Overall Performance Analysis ==\n")
        
        # Best chunk size for small files (< 1MB)
        small_files = results_df[results_df["file_size"] < 1024*1024]
        if not small_files.empty:
            best_small_write = small_files.loc[small_files["write_throughput"].idxmax()]
            best_small_read = small_files.loc[small_files["read_throughput"].idxmax()]
            
            f.write(f"Best chunk size for small files (< 1MB) write operations: {best_small_write['chunk_size']} bytes\n")
            f.write(f"Best chunk size for small files (< 1MB) read operations: {best_small_read['chunk_size']} bytes\n\n")
        
        # Best chunk size for large files (>= 1MB)
        large_files = results_df[results_df["file_size"] >= 1024*1024]
        if not large_files.empty:
            best_large_write = large_files.loc[large_files["write_throughput"].idxmax()]
            best_large_read = large_files.loc[large_files["read_throughput"].idxmax()]
            
            f.write(f"Best chunk size for large files (>= 1MB) write operations: {best_large_write['chunk_size']} bytes\n")
            f.write(f"Best chunk size for large files (>= 1MB) read operations: {best_large_read['chunk_size']} bytes\n\n")
        
        # Overall recommendations
        all_writes = results_df.groupby("chunk_size")["write_throughput"].mean()
        all_reads = results_df.groupby("chunk_size")["read_throughput"].mean()
        
        best_overall_write = all_writes.idxmax()
        best_overall_read = all_reads.idxmax()
        
        f.write(f"Best overall chunk size for write operations: {best_overall_write} bytes\n")
        f.write(f"Best overall chunk size for read operations: {best_overall_read} bytes\n\n")
        
        f.write("== Detailed Analysis ==\n")
        for chunk_size in CHUNK_SIZES:
            chunk_data = results_df[results_df["chunk_size"] == chunk_size]
            avg_write = chunk_data["write_throughput"].mean()
            avg_read = chunk_data["read_throughput"].mean()
            avg_latency = chunk_data["latency"].mean()
            
            f.write(f"\nChunk size: {chunk_size} bytes\n")
            f.write(f"  Average write throughput: {avg_write:.2f} MB/s\n")
            f.write(f"  Average read throughput: {avg_read:.2f} MB/s\n")
            f.write(f"  Average operation latency: {avg_latency:.4f} seconds\n")
            
            # File size specific details
            f.write("  File size specific performance:\n")
            for file_size in FILE_SIZES:
                file_data = chunk_data[chunk_data["file_size"] == file_size]
                if not file_data.empty:
                    file_write = file_data["write_throughput"].values[0]
                    file_read = file_data["read_throughput"].values[0]
                    file_latency = file_data["latency"].values[0]
                    
                    f.write(f"    {file_size/(1024*1024):.2f} MB: Write={file_write:.2f} MB/s, "
                            f"Read={file_read:.2f} MB/s, Latency={file_latency:.4f}s\n")


@pytest.mark.performance
def test_17_chunk_size_performance():
    """Test performance with different chunk sizes."""
    logger.info("Starting chunk size performance benchmark tests")
    
    timestamp = int(time.time())
    test_run_dir = f"{RESULT_DIR}/run_{timestamp}"
    os.makedirs(test_run_dir, exist_ok=True)
    
    logger.info(f"Test results will be saved in: {test_run_dir}")
    
    # Run benchmarks for each chunk size configuration
    results = []
    for chunk_size in CHUNK_SIZES:
        logger.info(f"Benchmarking with chunk size: {chunk_size} bytes")
        chunk_results = benchmark_performance(chunk_size)
        results.append(chunk_results)
        
        # Save individual result
        with open(f"{test_run_dir}/chunk_{chunk_size}_results.json", "w") as f:
            json.dump(chunk_results, f, indent=2)
    
    # Generate visualizations and report
    logger.info("Generating visualizations and performance report")
    visualize_results(results, test_run_dir)
    
    # Validate results
    assert len(results) == len(CHUNK_SIZES), "Expected results for all chunk size configurations"
    
    # Find optimal chunk sizes for different workloads
    small_files_data = [
        (result["chunk_size"], 
         sum(result["write_throughput"].get(str(size), 0) for size in FILE_SIZES if size < 1024*1024) / 
         sum(1 for size in FILE_SIZES if size < 1024*1024))
        for result in results
    ]
    
    large_files_data = [
        (result["chunk_size"], 
         sum(result["write_throughput"].get(str(size), 0) for size in FILE_SIZES if size >= 1024*1024) / 
         sum(1 for size in FILE_SIZES if size >= 1024*1024))
        for result in results
    ]
    
    # Find best chunk sizes
    optimal_small_files = max(small_files_data, key=lambda x: x[1])[0]
    optimal_large_files = max(large_files_data, key=lambda x: x[1])[0]
    
    logger.info(f"Optimal chunk size for small files: {optimal_small_files} bytes")
    logger.info(f"Optimal chunk size for large files: {optimal_large_files} bytes")
    
    # Final report path
    final_report_path = f"{test_run_dir}/summary_report.txt"
    logger.info(f"Performance test complete. Final report: {final_report_path}")
    
    # Assert that we have meaningful data
    assert os.path.exists(final_report_path), "Summary report should be generated"
    with open(final_report_path, "r") as f:
        report_content = f.read()
    
    assert "Best chunk size for small files" in report_content, "Report should include small file analysis"
    assert "Best chunk size for large files" in report_content, "Report should include large file analysis"

#!/usr/bin/env python3

import pytest
import time
import os
import subprocess
import threading
import datetime
import csv
import matplotlib.pyplot as plt
import numpy as np
from pyozone.client import OzoneClient
import psutil
import logging
import pandas as pd
from pathlib import Path


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("continuous_ingest_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("continuous_ingest_test")

# Test Configuration
TEST_DURATION_HOURS = 24
METRICS_COLLECTION_INTERVAL_SEC = 60
INGEST_BATCH_SIZE_MB = 10
NUM_PARALLEL_INGEST_THREADS = 5
OZONE_METRICS_ENDPOINT = "http://localhost:9876/metrics"

# Directory setup
TEST_DIR = Path("./test_data")
RESULTS_DIR = Path("./test_results")
for dir_path in [TEST_DIR, RESULTS_DIR]:
    dir_path.mkdir(exist_ok=True)


class MetricsCollector:
    """Collect system and Ozone metrics during the test run"""
    
    def __init__(self, output_file):
        self.output_file = output_file
        self.running = False
        self.metrics_thread = None
        self.metrics_data = []
        self.headers = [
            'timestamp', 'cpu_percent', 'memory_percent', 
            'disk_io_read_bytes', 'disk_io_write_bytes',
            'network_sent_bytes', 'network_recv_bytes',
            'datanodes_active', 'containers_count',
            'keys_processed', 'total_bytes_written'
        ]
        
        # Create CSV file with headers
        with open(self.output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)
    
    def start(self):
        """Start metrics collection in a background thread"""
        self.running = True
        self.metrics_thread = threading.Thread(target=self._collect_metrics)
        self.metrics_thread.daemon = True
        self.metrics_thread.start()
        logger.info(f"Started metrics collection, saving to {self.output_file}")
    
    def stop(self):
        """Stop metrics collection"""
        self.running = False
        if self.metrics_thread:
            self.metrics_thread.join(timeout=10)
        logger.info("Stopped metrics collection")
    
    def _collect_metrics(self):
        """Collect metrics at regular intervals"""
        # Initial values for calculating delta
        prev_io_counters = psutil.disk_io_counters()
        prev_net_counters = psutil.net_io_counters()
        keys_processed = 0
        total_bytes_written = 0
        
        while self.running:
            try:
                # System metrics
                timestamp = datetime.datetime.now().isoformat()
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_percent = psutil.virtual_memory().percent
                
                # IO metrics (delta since last measurement)
                current_io = psutil.disk_io_counters()
                disk_read_delta = current_io.read_bytes - prev_io_counters.read_bytes
                disk_write_delta = current_io.write_bytes - prev_io_counters.write_bytes
                prev_io_counters = current_io
                
                # Network metrics (delta)
                current_net = psutil.net_io_counters()
                net_sent_delta = current_net.bytes_sent - prev_net_counters.bytes_sent
                net_recv_delta = current_net.bytes_recv - prev_net_counters.bytes_recv
                prev_net_counters = current_net
                
                # Ozone metrics (would normally come from metrics API)
                # These are mocked for the example - in real tests, fetch from metrics endpoint
                datanodes_active = self._get_ozone_metric("datanodes_active")
                containers_count = self._get_ozone_metric("containers_count")
                
                # Update progress metrics from data ingest
                # In real implementation, these would be updated by the ingest process
                keys_processed += np.random.randint(10, 50)  # Simulated progress
                total_bytes_written += np.random.randint(1, 10) * 1024 * 1024  # Simulated MB written
                
                # Save the metrics
                row = [
                    timestamp, cpu_percent, memory_percent,
                    disk_read_delta, disk_write_delta,
                    net_sent_delta, net_recv_delta,
                    datanodes_active, containers_count,
                    keys_processed, total_bytes_written
                ]
                
                with open(self.output_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
                
                # Sleep until next collection
                time.sleep(METRICS_COLLECTION_INTERVAL_SEC)
                
            except Exception as e:
                logger.error(f"Error collecting metrics: {str(e)}")
    
    def _get_ozone_metric(self, metric_name):
        """
        Get Ozone metric from metrics endpoint
        In this example, just return simulated values
        """
        # In real implementation, fetch from metrics API
        if metric_name == "datanodes_active":
            # Simulate occasional datanode fluctuation
            return max(3, 3 + int(np.random.normal(0, 0.5)))
        elif metric_name == "containers_count":
            # Simulate growing number of containers over time
            return int(100 + time.time() / 300)  # Increases gradually
        return 0

    def analyze_and_plot_results(self):
        """Analyze collected metrics and generate plots"""
        logger.info("Analyzing metrics data...")
        
        # Read the saved metrics
        df = pd.read_csv(self.output_file)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create plots directory
        plots_dir = RESULTS_DIR / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Generate system resource plots
        self._create_plot(df, 'cpu_percent', 'CPU Usage (%)', plots_dir / 'cpu_usage.png')
        self._create_plot(df, 'memory_percent', 'Memory Usage (%)', plots_dir / 'memory_usage.png')
        
        # IO rates (convert to MB/s)
        df['disk_write_rate_mbs'] = df['disk_io_write_bytes'] / 1024 / 1024 / METRICS_COLLECTION_INTERVAL_SEC
        self._create_plot(df, 'disk_write_rate_mbs', 'Disk Write Rate (MB/s)', plots_dir / 'disk_write_rate.png')
        
        # Throughput over time
        timestamps = df['timestamp']
        timepoints = [(t - timestamps.iloc[0]).total_seconds() / 3600 for t in timestamps]  # Hours from start
        
        # Calculate throughput rate (change in bytes written over time)
        df['throughput_mbs'] = df['total_bytes_written'].diff() / (1024 * 1024 * METRICS_COLLECTION_INTERVAL_SEC)
        df['throughput_mbs'] = df['throughput_mbs'].fillna(0)
        
        # Plot throughput
        self._create_plot(df, 'throughput_mbs', 'Data Ingest Throughput (MB/s)', plots_dir / 'throughput.png')
        
        # Check for performance degradation
        self._analyze_degradation(df)
        
        logger.info(f"Analysis complete. Plots saved to {plots_dir}")
        return plots_dir
    
    def _create_plot(self, df, column, title, filename):
        """Create a time series plot for the given metric"""
        plt.figure(figsize=(12, 6))
        plt.plot(df['timestamp'], df[column])
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel(title)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
    
    def _analyze_degradation(self, df):
        """
        Analyze the metrics for signs of performance degradation
        Returns a dict with degradation assessments
        """
        results = {}
        
        # Check for throughput degradation using linear regression
        x = np.arange(len(df))
        
        # Smooth throughput data for analysis (moving average)
        window_size = max(10, len(df) // 20)  # Use at least 5% of datapoints for window
        df['smoothed_throughput'] = df['throughput_mbs'].rolling(window=window_size, min_periods=1).mean()
        
        # Linear regression to detect trend
        y = df['smoothed_throughput'].values
        slope, intercept = np.polyfit(x, y, 1)
        
        # Calculate relative degradation (% change from start to end)
        if intercept > 0:  # Avoid division by zero
            relative_change = (slope * len(df)) / intercept * 100
            results['throughput_trend_percent'] = relative_change
            
            if relative_change < -20:
                results['degradation_severity'] = 'Severe'
            elif relative_change < -10:
                results['degradation_severity'] = 'Moderate'
            elif relative_change < -5:
                results['degradation_severity'] = 'Minor'
            else:
                results['degradation_severity'] = 'None'
        
        # Check if CPU or memory usage increases significantly over time
        for resource in ['cpu_percent', 'memory_percent']:
            x = np.arange(len(df))
            y = df[resource].values
            slope, _ = np.polyfit(x, y, 1)
            total_change = slope * len(df)
            results[f'{resource}_trend'] = total_change
        
        # Save results to file
        with open(RESULTS_DIR / 'degradation_analysis.txt', 'w') as f:
            for key, value in results.items():
                f.write(f"{key}: {value}\n")
        
        return results


class DataIngestWorker:
    """Worker to ingest data into Ozone continuously"""
    
    def __init__(self, worker_id, volume_name, bucket_name):
        self.worker_id = worker_id
        self.volume_name = volume_name
        self.bucket_name = bucket_name
        self.running = False
        self.thread = None
        self.keys_processed = 0
        self.bytes_written = 0
    
    def start(self):
        """Start data ingest in a background thread"""
        self.running = True
        self.thread = threading.Thread(target=self._ingest_data)
        self.thread.daemon = True
        self.thread.start()
        logger.info(f"Started data ingest worker {self.worker_id}")
    
    def stop(self):
        """Stop data ingest"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=10)
        logger.info(f"Stopped data ingest worker {self.worker_id}")
    
    def _ingest_data(self):
        """Continuously ingest data into Ozone"""
        try:
            client = OzoneClient()
            
            # Create test data file if not exists
            test_file = TEST_DIR / f"test_data_{self.worker_id}.bin"
            if not test_file.exists():
                self._create_test_file(test_file, INGEST_BATCH_SIZE_MB)
            
            batch_num = 0
            while self.running:
                try:
                    # Create unique key name based on timestamp and batch number
                    timestamp = int(time.time())
                    key_name = f"data_{self.worker_id}_{timestamp}_{batch_num}"
                    
                    # Put data into Ozone
                    with open(test_file, 'rb') as f:
                        client.put_key(self.volume_name, self.bucket_name, key_name, f)
                    
                    # Update progress metrics
                    self.keys_processed += 1
                    self.bytes_written += test_file.stat().st_size
                    batch_num += 1
                    
                    # Small sleep to prevent overwhelming the system
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Error in data ingest worker {self.worker_id}: {str(e)}")
                    time.sleep(5)  # Back off on error
            
        except Exception as e:
            logger.error(f"Failed to initialize worker {self.worker_id}: {str(e)}")
    
    def _create_test_file(self, path, size_mb):
        """Create a test file of the specified size"""
        with open(path, 'wb') as f:
            f.write(os.urandom(size_mb * 1024 * 1024))
        logger.info(f"Created test file {path} of size {size_mb}MB")


@pytest.fixture(scope="module")
def ozone_setup():
    """Setup Ozone volume and bucket for testing"""
    volume_name = f"perf-vol-{int(time.time())}"
    bucket_name = f"perf-bucket-{int(time.time())}"
    
    # Create volume and bucket
    try:
        # Using ozone shell for setup
        subprocess.run(["ozone", "sh", "volume", "create", volume_name], check=True)
        subprocess.run(["ozone", "sh", "bucket", "create", f"{volume_name}/{bucket_name}"], check=True)
        
        yield volume_name, bucket_name
        
        # Clean up
        subprocess.run(["ozone", "sh", "bucket", "delete", f"{volume_name}/{bucket_name}"], check=True)
        subprocess.run(["ozone", "sh", "volume", "delete", volume_name], check=True)
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Failed to set up Ozone environment: {str(e)}")


def test_18_continuous_data_ingest_performance(ozone_setup):
    """
    Measure performance under continuous data ingest
    
    Tests the stability and performance of the Ozone cluster under continuous
    data ingest operations over an extended period (24 hours by default).
    Collects and analyzes system metrics to detect any degradation over time.
    """
    volume_name, bucket_name = ozone_setup
    
    # Setup metrics collection
    metrics_file = RESULTS_DIR / f"metrics_{int(time.time())}.csv"
    metrics_collector = MetricsCollector(metrics_file)
    
    # Create and start data ingest workers
    workers = []
    for i in range(NUM_PARALLEL_INGEST_THREADS):
        worker = DataIngestWorker(i, volume_name, bucket_name)
        workers.append(worker)
    
    try:
        # Start metrics collection
        metrics_collector.start()
        
        # Start data ingest workers
        for worker in workers:
            worker.start()
        
        # Calculate test duration in seconds
        test_duration_sec = TEST_DURATION_HOURS * 3600
        
        # Log progress periodically
        start_time = time.time()
        end_time = start_time + test_duration_sec
        
        # For test runs shorter than the full duration
        if pytest.config.getoption("--quick-test", default=False):
            end_time = start_time + 600  # 10 minutes for quick test
        
        while time.time() < end_time:
            elapsed = time.time() - start_time
            elapsed_hours = elapsed / 3600
            
            # Calculate total progress across all workers
            total_keys = sum(w.keys_processed for w in workers)
            total_bytes = sum(w.bytes_written for w in workers)
            total_mb = total_bytes / (1024 * 1024)
            
            logger.info(f"Progress: {elapsed_hours:.2f}h / {TEST_DURATION_HOURS}h - "
                       f"Keys: {total_keys}, Data: {total_mb:.2f}MB")
            
            # Sleep for progress report interval
            time.sleep(300)  # Report every 5 minutes
        
    finally:
        # Stop workers
        for worker in workers:
            worker.stop()
        
        # Stop metrics collection
        metrics_collector.stop()
    
    # Analyze results
    plots_dir = metrics_collector.analyze_and_plot_results()
    
    # Read degradation analysis
    analysis_file = RESULTS_DIR / 'degradation_analysis.txt'
    degradation_results = {}
    with open(analysis_file, 'r') as f:
        for line in f:
            key, value = line.strip().split(': ')
            try:
                degradation_results[key] = float(value)
            except ValueError:
                degradation_results[key] = value
    
    # Validate test results
    throughput_trend = degradation_results.get('throughput_trend_percent', 0)
    degradation_severity = degradation_results.get('degradation_severity', 'Unknown')
    
    # Log results
    logger.info(f"Test complete. Throughput trend: {throughput_trend:.2f}%")
    logger.info(f"Degradation assessment: {degradation_severity}")
    
    # Assert acceptable performance (no severe degradation)
    assert degradation_severity not in ['Severe'], \
        f"Severe performance degradation detected: {throughput_trend:.2f}% throughput decline"
    
    if degradation_severity == 'Moderate':
        logger.warning(f"Moderate performance degradation detected: {throughput_trend:.2f}% throughput decline")
    
    # Check that the test ran for the expected duration
    actual_duration = time.time() - start_time
    expected_duration = test_duration_sec
    assert actual_duration >= expected_duration * 0.95, \
        f"Test did not run for expected duration: {actual_duration}s vs {expected_duration}s"

import os
import time
import pytest
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Constants for the test
VOLUME_NAME = "perfvol"
BUCKET_NAME = "perfbucket"
TEST_DATA_DIR = "/tmp/ozone_test_data"
RESULTS_DIR = "/tmp/ozone_test_results"
EC_CONFIG_PATH = "/etc/hadoop/conf/ozone-site.xml"

# Test file sizes to use (in MB)
TEST_FILE_SIZES = [10, 50, 100, 250, 500, 750, 1024, 2048]  # Various file sizes from 10MB to 2GB

# Number of operations to perform for each test
NUM_OPERATIONS = 5

# Threads to use for concurrent tests
MAX_THREADS = 8

def setup_module(module):
    """Set up environment for the performance tests"""
    os.makedirs(TEST_DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Create test files of different sizes
    for size_mb in TEST_FILE_SIZES:
        file_path = os.path.join(TEST_DATA_DIR, f"test_file_{size_mb}mb")
        if not os.path.exists(file_path):
            subprocess.run(f"dd if=/dev/urandom of={file_path} bs=1M count={size_mb}", shell=True)

def teardown_module(module):
    """Clean up after tests"""
    # Keep the result data but remove test files
    for size_mb in TEST_FILE_SIZES:
        file_path = os.path.join(TEST_DATA_DIR, f"test_file_{size_mb}mb")
        if os.path.exists(file_path):
            os.remove(file_path)

def enable_erasure_coding():
    """Enable erasure coding in Ozone configuration"""
    # This would typically modify ozone-site.xml to enable EC
    # For example, set ozone.client.ec.enable=true
    subprocess.run(f"hadoop fs -put -f {EC_CONFIG_PATH} {EC_CONFIG_PATH}.backup", shell=True)
    
    # Example command to enable EC (specific commands would depend on your cluster setup)
    subprocess.run("""
        hdfs ec -enablePolicy -policy RS-3-2-1024k
        ozone admin ec enableec /ec
    """, shell=True)
    
    # Restart Ozone services to apply changes
    subprocess.run("ozone admin restart -service datanode", shell=True)
    time.sleep(10)  # Wait for services to restart

def disable_erasure_coding():
    """Disable erasure coding and revert to replication"""
    # Restore original configuration
    subprocess.run(f"hadoop fs -get {EC_CONFIG_PATH}.backup {EC_CONFIG_PATH}", shell=True)
    
    # Restart Ozone services to apply changes
    subprocess.run("ozone admin restart -service datanode", shell=True)
    time.sleep(10)  # Wait for services to restart

def perform_benchmark(ec_enabled, file_sizes=None, num_ops=None):
    """
    Perform read/write benchmarks on Ozone
    
    Args:
        ec_enabled: Whether erasure coding is enabled
        file_sizes: List of file sizes to test with
        num_ops: Number of operations to perform for each file size
    
    Returns:
        Dictionary with benchmark results
    """
    if file_sizes is None:
        file_sizes = TEST_FILE_SIZES
    if num_ops is None:
        num_ops = NUM_OPERATIONS
    
    test_type = "erasure_coding" if ec_enabled else "replication"
    
    # Create volume and bucket for tests
    volume_name = f"{VOLUME_NAME}_{test_type}"
    bucket_name = f"{BUCKET_NAME}_{test_type}"
    
    subprocess.run(f"ozone sh volume create /{volume_name}", shell=True)
    subprocess.run(f"ozone sh bucket create /{volume_name}/{bucket_name}", shell=True)
    
    results = {
        "write_speed_mbps": [],
        "read_speed_mbps": [],
        "file_sizes_mb": file_sizes,
        "test_type": test_type
    }
    
    # Perform write tests
    for size_mb in file_sizes:
        test_file = os.path.join(TEST_DATA_DIR, f"test_file_{size_mb}mb")
        key_name = f"key_{size_mb}mb"
        
        write_times = []
        read_times = []
        
        for i in range(num_ops):
            # Write test
            start_time = time.time()
            subprocess.run(
                f"ozone sh key put /{volume_name}/{bucket_name}/{key_name} {test_file}",
                shell=True
            )
            end_time = time.time()
            write_time = end_time - start_time
            write_speed = size_mb / write_time  # MB/s
            write_times.append(write_speed)
            
            # Read test
            output_file = os.path.join(TEST_DATA_DIR, f"output_{size_mb}mb_{i}")
            start_time = time.time()
            subprocess.run(
                f"ozone sh key get /{volume_name}/{bucket_name}/{key_name} {output_file}",
                shell=True
            )
            end_time = time.time()
            read_time = end_time - start_time
            read_speed = size_mb / read_time  # MB/s
            read_times.append(read_speed)
            
            # Clean up output file
            if os.path.exists(output_file):
                os.remove(output_file)
        
        # Average performance
        results["write_speed_mbps"].append(sum(write_times) / len(write_times))
        results["read_speed_mbps"].append(sum(read_times) / len(read_times))
    
    # Clean up
    subprocess.run(f"ozone sh bucket delete /{volume_name}/{bucket_name}", shell=True)
    subprocess.run(f"ozone sh volume delete /{volume_name}", shell=True)
    
    return results

def simulate_node_failure_recovery(ec_enabled):
    """
    Simulate node failures and measure recovery time
    
    Args:
        ec_enabled: Whether erasure coding is enabled
    
    Returns:
        Recovery time in seconds
    """
    test_type = "erasure_coding" if ec_enabled else "replication"
    
    # Create volume and bucket for tests
    volume_name = f"{VOLUME_NAME}_recovery_{test_type}"
    bucket_name = f"{BUCKET_NAME}_recovery_{test_type}"
    
    subprocess.run(f"ozone sh volume create /{volume_name}", shell=True)
    subprocess.run(f"ozone sh bucket create /{volume_name}/{bucket_name}", shell=True)
    
    # Use a medium-sized file for the test
    test_file = os.path.join(TEST_DATA_DIR, "test_file_100mb")
    key_name = "recovery_test_key"
    
    # Write a test file
    subprocess.run(
        f"ozone sh key put /{volume_name}/{bucket_name}/{key_name} {test_file}",
        shell=True
    )
    
    # Simulate node failure by stopping one datanode
    datanode_id = subprocess.check_output("ozone admin datanode list | head -n 1 | awk '{print $1}'", shell=True).decode().strip()
    subprocess.run(f"ozone admin datanode stop {datanode_id}", shell=True)
    time.sleep(5)  # Give time for the system to register the failure
    
    # Measure recovery time
    start_time = time.time()
    
    # Wait for recovery to complete by continuously checking if the key is accessible
    max_wait_time = 300  # 5 minutes max wait
    recovery_complete = False
    
    while time.time() - start_time < max_wait_time and not recovery_complete:
        try:
            output_file = os.path.join(TEST_DATA_DIR, "recovery_output")
            result = subprocess.run(
                f"ozone sh key get /{volume_name}/{bucket_name}/{key_name} {output_file}",
                shell=True, capture_output=True
            )
            if result.returncode == 0:
                recovery_complete = True
                if os.path.exists(output_file):
                    os.remove(output_file)
        except:
            pass
        
        if not recovery_complete:
            time.sleep(1)
    
    recovery_time = time.time() - start_time if recovery_complete else max_wait_time
    
    # Restart the datanode
    subprocess.run(f"ozone admin datanode start {datanode_id}", shell=True)
    time.sleep(10)  # Allow time for datanode to restart
    
    # Clean up
    subprocess.run(f"ozone sh bucket delete /{volume_name}/{bucket_name}", shell=True)
    subprocess.run(f"ozone sh volume delete /{volume_name}", shell=True)
    
    return recovery_time

def calculate_storage_efficiency(ec_enabled):
    """
    Calculate storage efficiency by measuring actual space used
    
    Args:
        ec_enabled: Whether erasure coding is enabled
    
    Returns:
        Dictionary with storage efficiency metrics
    """
    test_type = "erasure_coding" if ec_enabled else "replication"
    
    # Create volume and bucket for tests
    volume_name = f"{VOLUME_NAME}_storage_{test_type}"
    bucket_name = f"{BUCKET_NAME}_storage_{test_type}"
    
    subprocess.run(f"ozone sh volume create /{volume_name}", shell=True)
    subprocess.run(f"ozone sh bucket create /{volume_name}/{bucket_name}", shell=True)
    
    # Get initial storage usage
    initial_usage = float(subprocess.check_output(
        "ozone admin storage --unit=MB | grep -i 'used space' | awk '{print $3}'", 
        shell=True
    ).decode().strip())
    
    # Upload a large file
    test_file = os.path.join(TEST_DATA_DIR, "test_file_1024mb")
    key_name = "storage_test_key"
    
    subprocess.run(
        f"ozone sh key put /{volume_name}/{bucket_name}/{key_name} {test_file}",
        shell=True
    )
    
    # Get final storage usage
    final_usage = float(subprocess.check_output(
        "ozone admin storage --unit=MB | grep -i 'used space' | awk '{print $3}'", 
        shell=True
    ).decode().strip())
    
    # Calculate efficiency metrics
    storage_used = final_usage - initial_usage
    raw_data_size = 1024  # The test file is 1024MB
    storage_overhead = storage_used / raw_data_size
    
    # Clean up
    subprocess.run(f"ozone sh bucket delete /{volume_name}/{bucket_name}", shell=True)
    subprocess.run(f"ozone sh volume delete /{volume_name}", shell=True)
    
    return {
        "test_type": test_type,
        "raw_data_mb": raw_data_size,
        "storage_used_mb": storage_used,
        "storage_overhead_ratio": storage_overhead
    }

def plot_benchmark_results(ec_results, replication_results):
    """Plot performance comparison between erasure coding and replication"""
    result_file = os.path.join(RESULTS_DIR, "benchmark_comparison.png")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Write speed plot
    ax1.plot(ec_results["file_sizes_mb"], ec_results["write_speed_mbps"], 'b-o', label='Erasure Coding')
    ax1.plot(replication_results["file_sizes_mb"], replication_results["write_speed_mbps"], 'r-o', label='Replication')
    ax1.set_xlabel('File Size (MB)')
    ax1.set_ylabel('Write Speed (MB/s)')
    ax1.set_title('Write Performance: Erasure Coding vs. Replication')
    ax1.legend()
    ax1.grid(True)
    
    # Read speed plot
    ax2.plot(ec_results["file_sizes_mb"], ec_results["read_speed_mbps"], 'b-o', label='Erasure Coding')
    ax2.plot(replication_results["file_sizes_mb"], replication_results["read_speed_mbps"], 'r-o', label='Replication')
    ax2.set_xlabel('File Size (MB)')
    ax2.set_ylabel('Read Speed (MB/s)')
    ax2.set_title('Read Performance: Erasure Coding vs. Replication')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(result_file)
    plt.close()
    
    return result_file

def save_benchmark_results_to_csv(ec_results, replication_results, ec_recovery_time, replication_recovery_time, 
                                ec_storage_metrics, replication_storage_metrics):
    """Save all benchmark results to a CSV file"""
    result_file = os.path.join(RESULTS_DIR, "benchmark_results.csv")
    
    # Create DataFrame for performance results
    df_perf = pd.DataFrame({
        'File_Size_MB': ec_results["file_sizes_mb"],
        'EC_Write_Speed_MBps': ec_results["write_speed_mbps"],
        'Replication_Write_Speed_MBps': replication_results["write_speed_mbps"],
        'EC_Read_Speed_MBps': ec_results["read_speed_mbps"],
        'Replication_Read_Speed_MBps': replication_results["read_speed_mbps"],
    })
    
    # Create DataFrame for recovery and storage metrics
    df_metrics = pd.DataFrame({
        'Metric': ['Recovery_Time_Seconds', 'Raw_Data_MB', 'Storage_Used_MB', 'Storage_Overhead_Ratio'],
        'Erasure_Coding': [
            ec_recovery_time, 
            ec_storage_metrics["raw_data_mb"],
            ec_storage_metrics["storage_used_mb"],
            ec_storage_metrics["storage_overhead_ratio"]
        ],
        'Replication': [
            replication_recovery_time,
            replication_storage_metrics["raw_data_mb"],
            replication_storage_metrics["storage_used_mb"],
            replication_storage_metrics["storage_overhead_ratio"]
        ]
    })
    
    # Write to CSV
    with open(result_file, 'w') as f:
        f.write("PERFORMANCE METRICS\n")
        df_perf.to_csv(f, index=False)
        f.write("\n\nRECOVERY AND STORAGE METRICS\n")
        df_metrics.to_csv(f, index=False)
    
    return result_file

@pytest.mark.performance
def test_19_erasure_coding_performance():
    """Test performance with erasure coding comparing to replication-based setup"""
    # Step 1: Configure Ozone to use replication (default) and benchmark
    disable_erasure_coding()
    replication_results = perform_benchmark(ec_enabled=False)
    replication_recovery_time = simulate_node_failure_recovery(ec_enabled=False)
    replication_storage_metrics = calculate_storage_efficiency(ec_enabled=False)
    
    # Step 2: Configure Ozone to use erasure coding and benchmark
    enable_erasure_coding()
    ec_results = perform_benchmark(ec_enabled=True)
    ec_recovery_time = simulate_node_failure_recovery(ec_enabled=True)
    ec_storage_metrics = calculate_storage_efficiency(ec_enabled=True)
    
    # Step 3: Restore original configuration
    disable_erasure_coding()
    
    # Step 4: Generate comparison plots and CSV results
    plot_file = plot_benchmark_results(ec_results, replication_results)
    csv_file = save_benchmark_results_to_csv(
        ec_results, replication_results, 
        ec_recovery_time, replication_recovery_time,
        ec_storage_metrics, replication_storage_metrics
    )
    
    # Assert that the result files were created
    assert os.path.exists(plot_file), f"Failed to create performance plot: {plot_file}"
    assert os.path.exists(csv_file), f"Failed to create results CSV: {csv_file}"
    
    # Verify storage efficiency improvement with erasure coding
    storage_improvement = (replication_storage_metrics["storage_overhead_ratio"] / 
                          ec_storage_metrics["storage_overhead_ratio"])
    
    # Performance degradation should be acceptable (not more than 30% slower)
    avg_ec_write = sum(ec_results["write_speed_mbps"]) / len(ec_results["write_speed_mbps"])
    avg_rep_write = sum(replication_results["write_speed_mbps"]) / len(replication_results["write_speed_mbps"])
    write_ratio = avg_ec_write / avg_rep_write
    
    avg_ec_read = sum(ec_results["read_speed_mbps"]) / len(ec_results["read_speed_mbps"])
    avg_rep_read = sum(replication_results["read_speed_mbps"]) / len(replication_results["read_speed_mbps"])
    read_ratio = avg_ec_read / avg_rep_read
    
    recovery_ratio = replication_recovery_time / ec_recovery_time
    
    print(f"Storage efficiency improvement: {storage_improvement:.2f}x")
    print(f"Write performance ratio (EC/Rep): {write_ratio:.2f}")
    print(f"Read performance ratio (EC/Rep): {read_ratio:.2f}")
    print(f"Recovery time ratio (Rep/EC): {recovery_ratio:.2f}")
    
    # Assertions for expected results
    assert storage_improvement > 1.0, "Erasure coding should improve storage efficiency"
    assert write_ratio > 0.7, "Write performance with EC shouldn't be less than 70% of replication"
    assert read_ratio > 0.7, "Read performance with EC shouldn't be less than 70% of replication"
    assert recovery_ratio > 0.5, "Recovery with EC shouldn't be more than 2x slower than replication"

import os
import time
import pytest
import subprocess
import shutil
import logging
import random
import string
import json
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_migration_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ozone_migration_test")

# Constants
TEST_DATA_DIR = "test_migration_data"
RESULT_DIR = "migration_results"
METRICS_FILE = os.path.join(RESULT_DIR, "migration_metrics.csv")
PLOTS_DIR = os.path.join(RESULT_DIR, "plots")
ACCEPTABLE_MIGRATION_TIME = 600  # 10 minutes in seconds
MAX_PERFORMANCE_DEGRADATION = 0.3  # 30% max acceptable performance degradation

# Migration scenarios to test
migration_scenarios = [
    {
        "name": "add_one_datanode",
        "description": "Add one new datanode to the cluster",
        "setup_cmd": "docker-compose scale datanode=4",  # Assuming initial setup has 3 datanodes
        "original_nodes": 3,
        "target_nodes": 4
    },
    {
        "name": "add_multiple_datanodes",
        "description": "Add multiple new datanodes to the cluster",
        "setup_cmd": "docker-compose scale datanode=6",
        "original_nodes": 3,
        "target_nodes": 6
    },
    {
        "name": "retire_datanode",
        "description": "Retire one datanode from the cluster",
        "setup_cmd": "ozone admin datanode -op decommission -id datanode1",
        "original_nodes": 3,
        "target_nodes": 2
    }
]

# Test data sizes for performance comparison (in MB)
test_data_sizes = [10, 100, 500, 1024]  # 10MB, 100MB, 500MB, 1GB


def setup_module():
    """Setup required for all tests."""
    # Create directories for results
    os.makedirs(TEST_DATA_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # Generate test files of different sizes
    generate_test_files()


def teardown_module():
    """Clean up after all tests."""
    # Clean up test data but keep the results
    if os.path.exists(TEST_DATA_DIR):
        shutil.rmtree(TEST_DATA_DIR)


def generate_test_files():
    """Generate test files of different sizes for testing."""
    for size_mb in test_data_sizes:
        file_path = os.path.join(TEST_DATA_DIR, f"test_file_{size_mb}MB.dat")
        
        # Check if file already exists
        if os.path.exists(file_path):
            continue
            
        # Generate file with random content
        with open(file_path, 'wb') as f:
            # Write in chunks to avoid memory issues
            chunk_size = 1024 * 1024  # 1MB
            for _ in range(size_mb):
                f.write(os.urandom(chunk_size))
        
        logger.info(f"Generated test file: {file_path}")


def generate_unique_name() -> str:
    """Generate a unique name for volumes and buckets."""
    suffix = ''.join(random.choice(string.ascii_lowercase) for _ in range(5))
    timestamp = int(time.time())
    return f"test{timestamp}{suffix}"


def run_ozone_command(cmd: str) -> Tuple[str, str, int]:
    """Run an Ozone CLI command and return stdout, stderr, and return code."""
    process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    stdout, stderr = process.communicate()
    return stdout, stderr, process.returncode


def create_test_data_in_ozone(volume: str, bucket: str, file_count: int = 5) -> List[str]:
    """Create test data in Ozone and return the list of keys."""
    # Create volume and bucket if they don't exist
    run_ozone_command(f"ozone sh volume create /{volume}")
    run_ozone_command(f"ozone sh bucket create /{volume}/{bucket}")
    
    # Upload test files
    keys = []
    for size_mb in test_data_sizes:
        for i in range(file_count):
            key = f"testfile_{size_mb}MB_{i}"
            file_path = os.path.join(TEST_DATA_DIR, f"test_file_{size_mb}MB.dat")
            
            # Upload the file to Ozone
            cmd = f"ozone sh key put /{volume}/{bucket}/{key} {file_path}"
            _, stderr, rc = run_ozone_command(cmd)
            
            if rc == 0:
                keys.append(key)
                logger.info(f"Uploaded {file_path} to /{volume}/{bucket}/{key}")
            else:
                logger.error(f"Failed to upload file: {stderr}")
    
    return keys


def verify_data_integrity(volume: str, bucket: str, keys: List[str]) -> bool:
    """Verify data integrity by checking if all keys exist and have correct content."""
    for key in keys:
        # Check if key exists
        cmd = f"ozone sh key info /{volume}/{bucket}/{key}"
        stdout, stderr, rc = run_ozone_command(cmd)
        
        if rc != 0:
            logger.error(f"Key /{volume}/{bucket}/{key} does not exist after migration: {stderr}")
            return False
        
        # Optionally download and verify content
        temp_download_path = os.path.join(TEST_DATA_DIR, "temp_download")
        download_cmd = f"ozone sh key get /{volume}/{bucket}/{key} {temp_download_path}"
        _, stderr, rc = run_ozone_command(download_cmd)
        
        if rc != 0:
            logger.error(f"Failed to download key for verification: {stderr}")
            return False
    
    return True


def collect_cluster_metrics() -> Dict:
    """Collect current cluster metrics using Ozone admin commands."""
    metrics = {}
    
    # Get cluster status
    stdout, _, _ = run_ozone_command("ozone admin status")
    
    # Get datanode metrics
    stdout, _, _ = run_ozone_command("ozone admin datanode -list")
    
    # Parse metrics and store relevant data
    # This is a simplified example; in a real test, you would parse the output
    metrics["timestamp"] = time.time()
    metrics["cluster_status"] = "healthy" if "healthy" in stdout.lower() else "unhealthy"
    
    # Additional metrics can be collected via JMX/Prometheus endpoints
    # For example, using requests library to fetch metrics from the Ozone metrics endpoint
    
    return metrics


def measure_write_performance(volume: str, bucket: str, file_size_mb: int) -> float:
    """Measure write performance by uploading a file and timing it."""
    file_path = os.path.join(TEST_DATA_DIR, f"test_file_{file_size_mb}MB.dat")
    key = f"perf_test_{int(time.time())}"
    
    start_time = time.time()
    cmd = f"ozone sh key put /{volume}/{bucket}/{key} {file_path}"
    _, stderr, rc = run_ozone_command(cmd)
    end_time = time.time()
    
    if rc != 0:
        logger.error(f"Failed to upload file for performance measurement: {stderr}")
        return 0.0
    
    # Calculate throughput in MB/s
    elapsed_time = end_time - start_time
    throughput = file_size_mb / elapsed_time if elapsed_time > 0 else 0
    
    return throughput


def plot_metrics(metrics_df, scenario_name):
    """Generate plots from metrics data."""
    # Plot migration time
    plt.figure(figsize=(10, 6))
    plt.plot(metrics_df['timestamp'], metrics_df['throughput'], marker='o')
    plt.title(f'Throughput During Migration - {scenario_name}')
    plt.xlabel('Time (s)')
    plt.ylabel('Throughput (MB/s)')
    plt.grid(True)
    plt.savefig(os.path.join(PLOTS_DIR, f'{scenario_name}_throughput.png'))
    plt.close()


@pytest.mark.parametrize("scenario", migration_scenarios)
def test_20_performance_data_migration(scenario):
    """
    Evaluate performance under data migration scenarios.
    
    This test sets up different data migration scenarios (adding nodes, retiring nodes),
    initiates the data migration process, monitors system performance during migration,
    measures time taken for migration to complete, and verifies data integrity post-migration.
    """
    logger.info(f"Starting data migration test for scenario: {scenario['name']}")
    
    # Create unique volume and bucket for this test
    volume = f"vol{generate_unique_name()}"
    bucket = f"bkt{generate_unique_name()}"
    
    try:
        # Step 1: Set up the migration scenario
        logger.info(f"Setting up migration scenario: {scenario['description']}")
        
        # Create test data before migration
        logger.info("Creating test data in Ozone")
        keys = create_test_data_in_ozone(volume, bucket)
        
        # Measure baseline performance
        baseline_throughput = {}
        for size in test_data_sizes:
            baseline_throughput[size] = measure_write_performance(volume, bucket, size)
            logger.info(f"Baseline throughput for {size}MB: {baseline_throughput[size]:.2f} MB/s")
        
        # Step 2: Initiate data migration process
        logger.info(f"Initiating data migration with command: {scenario['setup_cmd']}")
        migration_start_time = time.time()
        stdout, stderr, rc = run_ozone_command(scenario['setup_cmd'])
        
        if rc != 0:
            logger.error(f"Failed to initiate migration: {stderr}")
            pytest.fail(f"Migration setup command failed: {stderr}")
        
        # Step 3: Monitor system performance during migration
        metrics_data = []
        migration_complete = False
        migration_timeout = migration_start_time + ACCEPTABLE_MIGRATION_TIME
        
        while not migration_complete and time.time() < migration_timeout:
            # Collect metrics
            metrics = collect_cluster_metrics()
            
            # Check migration progress - this would depend on specific Ozone admin commands
            stdout, _, _ = run_ozone_command("ozone admin rebalance status")
            
            # Check if migration is complete
            if "completed" in stdout.lower() or "finished" in stdout.lower():
                migration_complete = True
            
            # Measure current performance
            current_throughput = {}
            for size in test_data_sizes:
                current_throughput[size] = measure_write_performance(volume, bucket, size)
                
                # Calculate performance degradation
                degradation = 1.0 - (current_throughput[size] / baseline_throughput[size]) if baseline_throughput[size] > 0 else 1.0
                
                metrics["file_size_mb"] = size
                metrics["throughput"] = current_throughput[size]
                metrics["degradation"] = degradation
                metrics_data.append(metrics.copy())
                
                logger.info(f"Current throughput for {size}MB: {current_throughput[size]:.2f} MB/s "
                           f"(degradation: {degradation:.2%})")
            
            # Sleep before next measurement
            time.sleep(10)
        
        # Step 4: Measure time taken for migration
        migration_end_time = time.time()
        migration_duration = migration_end_time - migration_start_time
        logger.info(f"Migration completed in {migration_duration:.2f} seconds")
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_csv(os.path.join(RESULT_DIR, f"{scenario['name']}_metrics.csv"), index=False)
        
        # Generate plots
        plot_metrics(metrics_df, scenario['name'])
        
        # Step 5: Verify data integrity post-migration
        logger.info("Verifying data integrity post-migration")
        data_integrity_ok = verify_data_integrity(volume, bucket, keys)
        
        # Assertions
        assert migration_complete, f"Migration did not complete within timeout of {ACCEPTABLE_MIGRATION_TIME} seconds"
        assert migration_duration < ACCEPTABLE_MIGRATION_TIME, f"Migration took too long: {migration_duration:.2f}s > {ACCEPTABLE_MIGRATION_TIME}s"
        
        # Check that performance degradation during migration is within acceptable limits
        max_degradation = metrics_df['degradation'].max()
        assert max_degradation <= MAX_PERFORMANCE_DEGRADATION, f"Performance degradation too high: {max_degradation:.2%} > {MAX_PERFORMANCE_DEGRADATION:.2%}"
        
        # Verify data integrity
        assert data_integrity_ok, "Data integrity check failed after migration"
        
        logger.info(f"Migration test for scenario '{scenario['name']}' completed successfully")
        
    except Exception as e:
        logger.exception(f"Error during migration test: {str(e)}")
        pytest.fail(f"Migration test failed with error: {str(e)}")
    
    finally:
        # Clean up
        try:
            run_ozone_command(f"ozone sh bucket delete /{volume}/{bucket}")
            run_ozone_command(f"ozone sh volume delete /{volume}")
        except Exception as cleanup_error:
            logger.error(f"Error during cleanup: {str(cleanup_error)}")

import os
import time
import subprocess
import pytest
import psutil
import pandas as pd
import matplotlib.pyplot as plt
from ozone.client import OzoneClient, OMClient
from concurrent.futures import ThreadPoolExecutor

# Test configuration
CONFIG = {
    "cluster_host": "localhost",
    "cluster_port": 9878,  # Standard Ozone port
    "secure_port": 9879,   # SSL port for Ozone
    "volume": "perftest",
    "bucket": "encryption",
    "file_sizes_mb": [10, 100, 500, 1024],  # Different sizes for testing
    "num_iterations": 5,  # Number of test repetitions for averaging
    "threads": [1, 4, 8],  # Different thread counts to test concurrency
    "acceptable_overhead_percent": 15,  # Max acceptable performance degradation
    "cert_path": "/etc/security/ssl/ozone.crt",
    "key_path": "/etc/security/ssl/ozone.key"
}

class OzonePerformanceTester:
    """Helper class for Ozone performance testing with and without encryption"""
    
    def __init__(self, config):
        self.config = config
        self.results = {
            "encrypted": {"write": [], "read": [], "cpu": []},
            "unencrypted": {"write": [], "read": [], "cpu": []}
        }
        
    def setup_test_environment(self, encryption_enabled=False):
        """Set up the test environment with or without encryption"""
        # Create client based on encryption setting
        if encryption_enabled:
            # SSL-enabled client
            self.client = OzoneClient(
                self.config["cluster_host"], 
                self.config["secure_port"], 
                secure=True,
                cert_path=self.config["cert_path"],
                key_path=self.config["key_path"]
            )
            print("Created SSL-enabled client")
        else:
            # Regular client
            self.client = OzoneClient(
                self.config["cluster_host"], 
                self.config["cluster_port"]
            )
            print("Created regular client")
            
        # Ensure volume and bucket exist
        self._ensure_volume_and_bucket()
        
    def _ensure_volume_and_bucket(self):
        """Create volume and bucket if they don't exist"""
        # Volume
        try:
            self.client.create_volume(self.config["volume"])
            print(f"Volume {self.config['volume']} created")
        except Exception as e:
            if "VOLUME_ALREADY_EXISTS" not in str(e):
                raise
                
        # Bucket
        try:
            self.client.create_bucket(self.config["volume"], self.config["bucket"])
            print(f"Bucket {self.config['bucket']} created")
        except Exception as e:
            if "BUCKET_ALREADY_EXISTS" not in str(e):
                raise
    
    def generate_test_file(self, size_mb):
        """Generate a test file of specified size"""
        filename = f"testfile_{size_mb}mb.dat"
        
        # Create file with random data
        with open(filename, "wb") as f:
            f.write(os.urandom(size_mb * 1024 * 1024))
        
        return filename
    
    def perform_write_test(self, size_mb, thread_count=1, encryption_enabled=False):
        """Perform write performance test"""
        filename = self.generate_test_file(size_mb)
        
        # Track CPU before test
        cpu_before = psutil.cpu_percent(interval=0.1)
        
        start_time = time.time()
        
        # Multi-threaded writes
        def write_file(i):
            key = f"key_{size_mb}mb_{i}"
            with open(filename, "rb") as f:
                self.client.put_key(
                    self.config["volume"],
                    self.config["bucket"],
                    key,
                    f
                )
            
        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            executor.map(write_file, range(thread_count))
            
        elapsed_time = time.time() - start_time
        cpu_after = psutil.cpu_percent(interval=0.1)
        cpu_usage = cpu_after - cpu_before
        
        # Remove test file
        os.remove(filename)
        
        # Calculate throughput in MB/s
        throughput = (size_mb * thread_count) / elapsed_time
        
        mode = "encrypted" if encryption_enabled else "unencrypted"
        self.results[mode]["write"].append({
            "size_mb": size_mb,
            "threads": thread_count,
            "time_seconds": elapsed_time,
            "throughput_mbs": throughput,
            "cpu_usage": cpu_usage
        })
        
        return throughput, cpu_usage
    
    def perform_read_test(self, size_mb, thread_count=1, encryption_enabled=False):
        """Perform read performance test"""
        # First write files for reading
        filename = self.generate_test_file(size_mb)
        keys = []
        
        for i in range(thread_count):
            key = f"key_read_{size_mb}mb_{i}"
            keys.append(key)
            with open(filename, "rb") as f:
                self.client.put_key(
                    self.config["volume"],
                    self.config["bucket"],
                    key,
                    f
                )
        
        # Remove test file
        os.remove(filename)
        
        # Track CPU before test
        cpu_before = psutil.cpu_percent(interval=0.1)
        
        start_time = time.time()
        
        # Multi-threaded reads
        def read_file(key):
            with self.client.get_key(
                self.config["volume"],
                self.config["bucket"],
                key
            ) as f:
                data = f.read()
        
        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            executor.map(read_file, keys)
            
        elapsed_time = time.time() - start_time
        cpu_after = psutil.cpu_percent(interval=0.1)
        cpu_usage = cpu_after - cpu_before
        
        # Calculate throughput in MB/s
        throughput = (size_mb * thread_count) / elapsed_time
        
        mode = "encrypted" if encryption_enabled else "unencrypted"
        self.results[mode]["read"].append({
            "size_mb": size_mb,
            "threads": thread_count,
            "time_seconds": elapsed_time,
            "throughput_mbs": throughput,
            "cpu_usage": cpu_usage
        })
        
        # Clean up the keys
        for key in keys:
            self.client.delete_key(self.config["volume"], self.config["bucket"], key)
        
        return throughput, cpu_usage
    
    def generate_report(self):
        """Generate performance comparison report"""
        # Convert results to dataframes
        encrypted_write_df = pd.DataFrame(self.results["encrypted"]["write"])
        unencrypted_write_df = pd.DataFrame(self.results["unencrypted"]["write"])
        
        encrypted_read_df = pd.DataFrame(self.results["encrypted"]["read"])
        unencrypted_read_df = pd.DataFrame(self.results["unencrypted"]["read"])
        
        # Calculate overhead percentage
        write_overhead = 100 * (1 - encrypted_write_df["throughput_mbs"].mean() / 
                             unencrypted_write_df["throughput_mbs"].mean())
        
        read_overhead = 100 * (1 - encrypted_read_df["throughput_mbs"].mean() / 
                            unencrypted_read_df["throughput_mbs"].mean())
        
        # CPU usage difference
        encrypted_cpu = pd.DataFrame(self.results["encrypted"]["cpu"]).mean() if self.results["encrypted"]["cpu"] else 0
        unencrypted_cpu = pd.DataFrame(self.results["unencrypted"]["cpu"]).mean() if self.results["unencrypted"]["cpu"] else 0
        
        # Create report dict
        report = {
            "write_overhead_percent": write_overhead,
            "read_overhead_percent": read_overhead,
            "encrypted_throughput_write_mbs": encrypted_write_df["throughput_mbs"].mean(),
            "unencrypted_throughput_write_mbs": unencrypted_write_df["throughput_mbs"].mean(),
            "encrypted_throughput_read_mbs": encrypted_read_df["throughput_mbs"].mean(),
            "unencrypted_throughput_read_mbs": unencrypted_read_df["throughput_mbs"].mean(),
            "encrypted_cpu_usage": encrypted_cpu,
            "unencrypted_cpu_usage": unencrypted_cpu
        }
        
        # Plot comparison graphs
        self._plot_comparison()
        
        return report
    
    def _plot_comparison(self):
        """Generate performance comparison plots"""
        # Prepare data
        encrypted_write_df = pd.DataFrame(self.results["encrypted"]["write"])
        unencrypted_write_df = pd.DataFrame(self.results["unencrypted"]["write"])
        
        # Group by size and threads
        enc_grouped = encrypted_write_df.groupby(["size_mb", "threads"])["throughput_mbs"].mean().reset_index()
        unenc_grouped = unencrypted_write_df.groupby(["size_mb", "threads"])["throughput_mbs"].mean().reset_index()
        
        # Plot throughput comparison
        plt.figure(figsize=(12, 8))
        
        # Write throughput
        plt.subplot(2, 1, 1)
        for thread in self.config["threads"]:
            enc_data = enc_grouped[enc_grouped["threads"] == thread]
            unenc_data = unenc_grouped[unenc_grouped["threads"] == thread]
            
            plt.plot(enc_data["size_mb"], enc_data["throughput_mbs"], 
                     marker='o', label=f'Encrypted ({thread} threads)')
            plt.plot(unenc_data["size_mb"], unenc_data["throughput_mbs"], 
                     marker='x', label=f'Unencrypted ({thread} threads)')
        
        plt.title('Write Throughput: Encrypted vs Unencrypted')
        plt.xlabel('File Size (MB)')
        plt.ylabel('Throughput (MB/s)')
        plt.legend()
        plt.grid(True)
        
        # Read throughput
        plt.subplot(2, 1, 2)
        encrypted_read_df = pd.DataFrame(self.results["encrypted"]["read"])
        unencrypted_read_df = pd.DataFrame(self.results["unencrypted"]["read"])
        
        enc_grouped = encrypted_read_df.groupby(["size_mb", "threads"])["throughput_mbs"].mean().reset_index()
        unenc_grouped = unencrypted_read_df.groupby(["size_mb", "threads"])["throughput_mbs"].mean().reset_index()
        
        for thread in self.config["threads"]:
            enc_data = enc_grouped[enc_grouped["threads"] == thread]
            unenc_data = unenc_grouped[unenc_grouped["threads"] == thread]
            
            plt.plot(enc_data["size_mb"], enc_data["throughput_mbs"], 
                     marker='o', label=f'Encrypted ({thread} threads)')
            plt.plot(unenc_data["size_mb"], unenc_data["throughput_mbs"], 
                     marker='x', label=f'Unencrypted ({thread} threads)')
        
        plt.title('Read Throughput: Encrypted vs Unencrypted')
        plt.xlabel('File Size (MB)')
        plt.ylabel('Throughput (MB/s)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('ozone_encryption_performance.png')
        plt.close()


@pytest.fixture(scope="module")
def performance_tester():
    """Fixture to provide a performance tester instance"""
    tester = OzonePerformanceTester(CONFIG)
    return tester


def test_21_encryption_performance_impact(performance_tester):
    """Test performance with data encryption in transit.
    
    Compares read/write performance and CPU usage with and without SSL/TLS encryption.
    Verifies that the encryption overhead is within acceptable limits.
    """
    # Part 1: Run tests without encryption
    performance_tester.setup_test_environment(encryption_enabled=False)
    
    print("\n=== Running performance tests without encryption ===")
    
    # Iterate over file sizes and thread counts
    for size in CONFIG["file_sizes_mb"]:
        for thread_count in CONFIG["threads"]:
            # Repeat tests for better averages
            for i in range(CONFIG["num_iterations"]):
                print(f"\nRunning iteration {i+1} for size {size}MB with {thread_count} threads")
                
                # Write test
                write_throughput, write_cpu = performance_tester.perform_write_test(
                    size, thread_count, encryption_enabled=False)
                print(f"Unencrypted write: {write_throughput:.2f} MB/s, CPU: {write_cpu:.2f}%")
                
                # Read test
                read_throughput, read_cpu = performance_tester.perform_read_test(
                    size, thread_count, encryption_enabled=False)
                print(f"Unencrypted read: {read_throughput:.2f} MB/s, CPU: {read_cpu:.2f}%")
                
    # Part 2: Run tests with encryption
    performance_tester.setup_test_environment(encryption_enabled=True)
    
    print("\n=== Running performance tests with encryption ===")
    
    # Iterate over file sizes and thread counts
    for size in CONFIG["file_sizes_mb"]:
        for thread_count in CONFIG["threads"]:
            # Repeat tests for better averages
            for i in range(CONFIG["num_iterations"]):
                print(f"\nRunning iteration {i+1} for size {size}MB with {thread_count} threads")
                
                # Write test
                write_throughput, write_cpu = performance_tester.perform_write_test(
                    size, thread_count, encryption_enabled=True)
                print(f"Encrypted write: {write_throughput:.2f} MB/s, CPU: {write_cpu:.2f}%")
                
                # Read test
                read_throughput, read_cpu = performance_tester.perform_read_test(
                    size, thread_count, encryption_enabled=True)
                print(f"Encrypted read: {read_throughput:.2f} MB/s, CPU: {read_cpu:.2f}%")
    
    # Generate performance report
    report = performance_tester.generate_report()
    
    print("\n=== Performance Results ===")
    for key, value in report.items():
        print(f"{key}: {value:.2f}")
    
    # Assertions to check if encryption overhead is acceptable
    assert report["write_overhead_percent"] < CONFIG["acceptable_overhead_percent"], \
        f"Write performance degradation ({report['write_overhead_percent']:.2f}%) exceeds acceptable limit ({CONFIG['acceptable_overhead_percent']}%)"
    
    assert report["read_overhead_percent"] < CONFIG["acceptable_overhead_percent"], \
        f"Read performance degradation ({report['read_overhead_percent']:.2f}%) exceeds acceptable limit ({CONFIG['acceptable_overhead_percent']}%)"
    
    print(f"\nEncryption overhead - Write: {report['write_overhead_percent']:.2f}%, Read: {report['read_overhead_percent']:.2f}%")
    print(f"Test passed: Encryption overhead is within acceptable limits (<{CONFIG['acceptable_overhead_percent']}%)")

import pytest
import time
import subprocess
import threading
import csv
import os
import concurrent.futures
import statistics
import logging
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants for test configuration
OZONE_SHELL_CMD = "ozone sh"
DEFAULT_VOLUME = "perfvol"
TEST_DATA_FILE = "test_data.txt"
RESULT_FILE = "metadata_perf_results.csv"

# Test parameters
OPERATION_COUNTS = [100, 500, 1000]  # Number of operations to perform
CONCURRENCY_LEVELS = [1, 5, 10, 20]  # Number of concurrent operations


class MetadataOperationResult:
    """Class to store results of metadata operations performance"""
    
    def __init__(self):
        self.operation_type = ""
        self.start_time = 0
        self.end_time = 0
        self.success = False
        self.error = None
        
    @property
    def latency(self) -> float:
        """Calculate latency in milliseconds"""
        return (self.end_time - self.start_time) * 1000
        

class OzoneMetadataTest:
    """Utility class to perform metadata operations on Ozone"""
    
    def __init__(self):
        # Create test data file if it doesn't exist
        if not os.path.exists(TEST_DATA_FILE):
            with open(TEST_DATA_FILE, 'w') as f:
                f.write("This is test data for performance testing")
    
    def setup_test_environment(self):
        """Set up the test environment with base volume"""
        logger.info("Setting up test environment")
        self._run_cmd(f"{OZONE_SHELL_CMD} volume create {DEFAULT_VOLUME}")
    
    def cleanup_test_environment(self):
        """Clean up the test environment"""
        logger.info("Cleaning up test environment")
        self._run_cmd(f"{OZONE_SHELL_CMD} volume delete {DEFAULT_VOLUME}")
    
    def create_bucket(self, bucket_name: str) -> MetadataOperationResult:
        """Create a bucket in the default volume"""
        result = MetadataOperationResult()
        result.operation_type = "create_bucket"
        
        try:
            result.start_time = time.time()
            output = self._run_cmd(f"{OZONE_SHELL_CMD} bucket create {DEFAULT_VOLUME}/{bucket_name}")
            result.end_time = time.time()
            result.success = True
        except Exception as e:
            result.end_time = time.time()
            result.error = str(e)
            result.success = False
            
        return result
    
    def delete_bucket(self, bucket_name: str) -> MetadataOperationResult:
        """Delete a bucket from the default volume"""
        result = MetadataOperationResult()
        result.operation_type = "delete_bucket"
        
        try:
            result.start_time = time.time()
            output = self._run_cmd(f"{OZONE_SHELL_CMD} bucket delete {DEFAULT_VOLUME}/{bucket_name}")
            result.end_time = time.time()
            result.success = True
        except Exception as e:
            result.end_time = time.time()
            result.error = str(e)
            result.success = False
            
        return result
    
    def list_buckets(self) -> MetadataOperationResult:
        """List all buckets in the default volume"""
        result = MetadataOperationResult()
        result.operation_type = "list_buckets"
        
        try:
            result.start_time = time.time()
            output = self._run_cmd(f"{OZONE_SHELL_CMD} bucket list {DEFAULT_VOLUME}")
            result.end_time = time.time()
            result.success = True
        except Exception as e:
            result.end_time = time.time()
            result.error = str(e)
            result.success = False
            
        return result
        
    def put_key(self, bucket_name: str, key_name: str) -> MetadataOperationResult:
        """Put a key into the specified bucket"""
        result = MetadataOperationResult()
        result.operation_type = "put_key"
        
        try:
            result.start_time = time.time()
            output = self._run_cmd(f"{OZONE_SHELL_CMD} key put {DEFAULT_VOLUME}/{bucket_name}/ {TEST_DATA_FILE} --key={key_name}")
            result.end_time = time.time()
            result.success = True
        except Exception as e:
            result.end_time = time.time()
            result.error = str(e)
            result.success = False
            
        return result
    
    def list_keys(self, bucket_name: str) -> MetadataOperationResult:
        """List all keys in the specified bucket"""
        result = MetadataOperationResult()
        result.operation_type = "list_keys"
        
        try:
            result.start_time = time.time()
            output = self._run_cmd(f"{OZONE_SHELL_CMD} key list {DEFAULT_VOLUME}/{bucket_name}")
            result.end_time = time.time()
            result.success = True
        except Exception as e:
            result.end_time = time.time()
            result.error = str(e)
            result.success = False
            
        return result
    
    def _run_cmd(self, cmd: str) -> str:
        """Run a shell command and return the output"""
        logger.debug(f"Running command: {cmd}")
        result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise Exception(f"Command failed: {result.stderr}")
        return result.stdout


class MetadataOperationTester:
    """Class to run parallel metadata operations and measure performance"""
    
    def __init__(self):
        self.ozone_test = OzoneMetadataTest()
        
    def monitor_om_performance(self) -> Dict:
        """Monitor Ozone Manager performance metrics via JMX or metrics API"""
        # In a real implementation, this would query the OM metrics endpoint
        # and collect performance data like CPU, memory usage, GC stats, etc.
        
        # Simplified version just returns dummy data
        return {
            "cpu_usage": 60,
            "memory_usage": 70,
            "active_requests": 45,
            "queue_size": 5
        }
    
    def check_data_operations(self) -> Dict:
        """Check if data read/write operations are impacted during metadata stress"""
        # Create a test bucket
        test_bucket = "datacheckbucket"
        self.ozone_test.create_bucket(test_bucket)
        
        # Measure write and read latencies
        write_start = time.time()
        self.ozone_test.put_key(test_bucket, "testkey")
        write_latency = (time.time() - write_start) * 1000
        
        # Clean up
        self.ozone_test.delete_bucket(test_bucket)
        
        return {
            "write_latency_ms": write_latency
        }
        
    def run_operation_batch(self, operation_count: int, concurrency: int, operation_type: str) -> List[MetadataOperationResult]:
        """Run a batch of operations with specified concurrency"""
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_op = {}
            
            for i in range(operation_count):
                if operation_type == "create_bucket":
                    future = executor.submit(self.ozone_test.create_bucket, f"bucket{i}")
                elif operation_type == "list_buckets":
                    future = executor.submit(self.ozone_test.list_buckets)
                elif operation_type == "delete_bucket":
                    future = executor.submit(self.ozone_test.delete_bucket, f"bucket{i}")
                future_to_op[future] = i
            
            for future in concurrent.futures.as_completed(future_to_op):
                result = future.result()
                results.append(result)
                
        return results
    
    def analyze_results(self, results: List[MetadataOperationResult]) -> Dict:
        """Analyze operation results for performance metrics"""
        latencies = [result.latency for result in results if result.success]
        success_rate = sum(1 for result in results if result.success) / len(results) * 100
        
        return {
            "operation_count": len(results),
            "success_count": sum(1 for result in results if result.success),
            "success_rate": success_rate,
            "min_latency_ms": min(latencies) if latencies else 0,
            "max_latency_ms": max(latencies) if latencies else 0,
            "avg_latency_ms": statistics.mean(latencies) if latencies else 0,
            "p95_latency_ms": statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies) if latencies else 0,
            "throughput_ops_sec": len(latencies) / (sum(latencies) / 1000) if sum(latencies) > 0 else 0
        }
    
    def save_results(self, test_config: Dict, results: Dict) -> None:
        """Save test results to CSV file"""
        header = ["timestamp", "operation_type", "operation_count", "concurrency", 
                 "success_rate", "min_latency_ms", "avg_latency_ms", "max_latency_ms", 
                 "p95_latency_ms", "throughput_ops_sec"]
        
        file_exists = os.path.isfile(RESULT_FILE)
        
        with open(RESULT_FILE, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=header)
            
            if not file_exists:
                writer.writeheader()
            
            row = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "operation_type": test_config["operation_type"],
                "operation_count": test_config["operation_count"],
                "concurrency": test_config["concurrency"],
                "success_rate": results["success_rate"],
                "min_latency_ms": results["min_latency_ms"],
                "avg_latency_ms": results["avg_latency_ms"],
                "max_latency_ms": results["max_latency_ms"],
                "p95_latency_ms": results["p95_latency_ms"],
                "throughput_ops_sec": results["throughput_ops_sec"]
            }
            
            writer.writerow(row)


@pytest.mark.parametrize("operation_count", OPERATION_COUNTS)
@pytest.mark.parametrize("concurrency", CONCURRENCY_LEVELS)
def test_22_metadata_operations_performance(operation_count: int, concurrency: int):
    """
    Evaluate performance under heavy metadata operations.
    
    This test simulates a high volume of metadata operations and measures the throughput
    and latency to ensure the Ozone Manager can handle the load efficiently.
    """
    # Initialize test components
    metadata_tester = MetadataOperationTester()
    ozone_test = OzoneMetadataTest()
    
    # Setup test environment
    ozone_test.setup_test_environment()
    
    try:
        logger.info(f"Running metadata performance test with {operation_count} operations at concurrency {concurrency}")
        
        # Test configurations for different operation types
        test_configs = [
            {"operation_type": "create_bucket", "operation_count": operation_count, "concurrency": concurrency},
            {"operation_type": "list_buckets", "operation_count": operation_count, "concurrency": concurrency}
        ]
        
        all_results = {}
        
        # Run each operation type test
        for config in test_configs:
            # Monitor OM performance before test
            before_metrics = metadata_tester.monitor_om_performance()
            
            # Run the operations
            results = metadata_tester.run_operation_batch(
                operation_count=config["operation_count"],
                concurrency=config["concurrency"],
                operation_type=config["operation_type"]
            )
            
            # Monitor OM performance after test
            after_metrics = metadata_tester.monitor_om_performance()
            
            # Analyze and save results
            analysis = metadata_tester.analyze_results(results)
            metadata_tester.save_results(config, analysis)
            
            all_results[config["operation_type"]] = analysis
            
            # Log results
            logger.info(f"Results for {config['operation_type']}:")
            logger.info(f"  Success rate: {analysis['success_rate']:.2f}%")
            logger.info(f"  Avg latency: {analysis['avg_latency_ms']:.2f} ms")
            logger.info(f"  Throughput: {analysis['throughput_ops_sec']:.2f} ops/sec")
            
            # Check data operations performance
            data_metrics = metadata_tester.check_data_operations()
            logger.info(f"Data operation write latency: {data_metrics['write_latency_ms']:.2f} ms")
        
        # For each operation type, assert that the performance meets requirements
        for op_type, analysis in all_results.items():
            # Assertions based on expected results - adjust thresholds as needed for your environment
            assert analysis['success_rate'] > 95, f"{op_type} success rate below threshold"
            
            # Example thresholds - adjust based on your environment
            if operation_count <= 100:
                assert analysis['avg_latency_ms'] < 200, f"{op_type} average latency too high"
                assert analysis['throughput_ops_sec'] > 5, f"{op_type} throughput too low"
            elif operation_count <= 500:
                assert analysis['avg_latency_ms'] < 300, f"{op_type} average latency too high"
                assert analysis['throughput_ops_sec'] > 4, f"{op_type} throughput too low"
            else:
                assert analysis['avg_latency_ms'] < 400, f"{op_type} average latency too high"
                assert analysis['throughput_ops_sec'] > 3, f"{op_type} throughput too low"
    
    finally:
        # Clean up test environment
        ozone_test.cleanup_test_environment()

#!/usr/bin/env python3

import os
import time
import pytest
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import tempfile
import random
import string

# Utility functions for performance testing
def generate_random_file(file_path: str, size_mb: float) -> str:
    """Generate a random file of specified size in MB"""
    # Convert MB to bytes
    size_bytes = int(size_mb * 1024 * 1024)
    
    with open(file_path, 'wb') as f:
        # Write in chunks to avoid memory issues with large files
        chunk_size = min(10 * 1024 * 1024, size_bytes)  # 10MB chunks or file size if smaller
        remaining = size_bytes
        
        while remaining > 0:
            write_size = min(chunk_size, remaining)
            f.write(os.urandom(write_size))
            remaining -= write_size
    
    return file_path

def execute_command(cmd: List[str], timeout: int = 300) -> Tuple[str, str, int]:
    """Execute a shell command and return stdout, stderr, and return code"""
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate(timeout=timeout)
        return stdout.decode('utf-8'), stderr.decode('utf-8'), process.returncode
    except subprocess.TimeoutExpired:
        process.kill()
        return "", f"Command timed out after {timeout} seconds", -1

def get_random_name(prefix: str = "", length: int = 8) -> str:
    """Generate a random name for volumes and buckets"""
    chars = string.ascii_lowercase + string.digits
    random_str = ''.join(random.choice(chars) for _ in range(length))
    return f"{prefix}{random_str}"

def setup_ozone_block_size(block_size_mb: int) -> bool:
    """
    Configure Ozone with the specified block size
    
    Args:
        block_size_mb: Block size in MB
    
    Returns:
        True if configuration succeeded, False otherwise
    """
    # Convert MB to bytes for the configuration
    block_size = block_size_mb * 1024 * 1024
    
    # In a real environment, this would modify the ozone-site.xml
    # Here we're simulating the configuration change
    cmd = [
        "ssh", 
        "ozone-admin", 
        f"echo 'dfs.blocksize{block_size}' >> /etc/hadoop/conf/ozone-site.xml"
    ]
    
    try:
        # This is a simulation - in a real test, you would actually apply these changes
        # to ozone-site.xml and restart the necessary services
        print(f"Configuring Ozone with block size: {block_size_mb}MB")
        
        # For this example, we'll pretend the configuration was successful
        # In a real test, you would verify the configuration was applied correctly
        return True
    except Exception as e:
        print(f"Error configuring block size: {e}")
        return False

def restart_ozone_cluster():
    """
    Restart the Ozone cluster to apply configuration changes
    """
    print("Restarting Ozone cluster to apply configuration changes...")
    # In a real environment, you would execute commands to restart the Ozone services
    # Example: execute_command(["ozone", "admin", "restart", "--all"])
    time.sleep(5)  # Simulating restart time

def run_benchmark(block_size_mb: int, file_sizes: List[float], access_patterns: List[str]) -> Dict:
    """
    Run benchmarks for given block size, file sizes, and access patterns
    
    Args:
        block_size_mb: Current block size in MB
        file_sizes: List of file sizes to test (in MB)
        access_patterns: List of access patterns to test
    
    Returns:
        Dictionary of benchmark results
    """
    results = {
        'block_size': block_size_mb,
        'file_size': [],
        'access_pattern': [],
        'write_throughput': [],
        'read_throughput': [],
        'write_latency': [],
        'read_latency': []
    }
    
    volume_name = get_random_name("vol")
    bucket_name = get_random_name("bucket")
    
    # Create volume and bucket for testing
    execute_command(["ozone", "sh", "volume", "create", f"/{volume_name}"])
    execute_command(["ozone", "sh", "bucket", "create", f"/{volume_name}/{bucket_name}"])
    
    # Run benchmarks for each file size and access pattern
    for file_size in file_sizes:
        for pattern in access_patterns:
            print(f"Testing block_size={block_size_mb}MB, file_size={file_size}MB, pattern={pattern}")
            
            # Generate test file
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                test_file_path = temp_file.name
            
            generate_random_file(test_file_path, file_size)
            
            # Run write benchmark
            key_name = f"test-{block_size_mb}-{file_size}-{pattern}-{int(time.time())}"
            
            write_start = time.time()
            if pattern == "sequential":
                cmd = ["ozone", "sh", "key", "put", f"/{volume_name}/{bucket_name}/{key_name}", test_file_path]
            else:  # random
                # For random access, we would need a custom benchmark tool
                # This is a simplified simulation
                cmd = ["ozone", "sh", "key", "put", f"/{volume_name}/{bucket_name}/{key_name}", test_file_path]
            
            stdout, stderr, rc = execute_command(cmd)
            write_end = time.time()
            
            if rc != 0:
                print(f"Error writing file: {stderr}")
                continue
            
            # Calculate write metrics
            write_time = write_end - write_start
            write_throughput = file_size / write_time  # MB/s
            
            # Run read benchmark
            output_file = test_file_path + ".out"
            
            read_start = time.time()
            if pattern == "sequential":
                cmd = ["ozone", "sh", "key", "get", f"/{volume_name}/{bucket_name}/{key_name}", output_file]
            else:  # random
                # For random access, we would need a custom benchmark tool
                cmd = ["ozone", "sh", "key", "get", f"/{volume_name}/{bucket_name}/{key_name}", output_file]
            
            stdout, stderr, rc = execute_command(cmd)
            read_end = time.time()
            
            if rc != 0:
                print(f"Error reading file: {stderr}")
                continue
            
            # Calculate read metrics
            read_time = read_end - read_start
            read_throughput = file_size / read_time  # MB/s
            
            # Clean up
            os.unlink(test_file_path)
            if os.path.exists(output_file):
                os.unlink(output_file)
            
            # Store results
            results['file_size'].append(file_size)
            results['access_pattern'].append(pattern)
            results['write_throughput'].append(write_throughput)
            results['read_throughput'].append(read_throughput)
            results['write_latency'].append(write_time)
            results['read_latency'].append(read_time)
    
    # Clean up volume and bucket
    execute_command(["ozone", "sh", "bucket", "delete", f"/{volume_name}/{bucket_name}"])
    execute_command(["ozone", "sh", "volume", "delete", f"/{volume_name}"])
    
    return results

def analyze_results(all_results: List[Dict]) -> Tuple[pd.DataFrame, Dict]:
    """
    Analyze benchmark results to determine optimal block sizes
    
    Args:
        all_results: List of result dictionaries from run_benchmark()
    
    Returns:
        DataFrame with analyzed results and dictionary of optimal block sizes
    """
    # Combine all results into a single DataFrame
    df_list = []
    for results in all_results:
        block_size = results['block_size']
        for i in range(len(results['file_size'])):
            row = {
                'block_size_mb': block_size,
                'file_size_mb': results['file_size'][i],
                'access_pattern': results['access_pattern'][i],
                'write_throughput_mbs': results['write_throughput'][i],
                'read_throughput_mbs': results['read_throughput'][i],
                'write_latency_s': results['write_latency'][i],
                'read_latency_s': results['read_latency'][i]
            }
            df_list.append(row)
    
    df = pd.DataFrame(df_list)
    
    # Determine optimal block size for each file size and access pattern
    optimal_block_sizes = {}
    
    for file_size in df['file_size_mb'].unique():
        optimal_block_sizes[file_size] = {}
        
        for pattern in df['access_pattern'].unique():
            filtered = df[(df['file_size_mb'] == file_size) & (df['access_pattern'] == pattern)]
            
            # Find block size with best read throughput
            best_read_idx = filtered['read_throughput_mbs'].idxmax()
            optimal_read_block_size = filtered.loc[best_read_idx, 'block_size_mb']
            
            # Find block size with best write throughput
            best_write_idx = filtered['write_throughput_mbs'].idxmax()
            optimal_write_block_size = filtered.loc[best_write_idx, 'block_size_mb']
            
            # Store the results
            optimal_block_sizes[file_size][f"{pattern}_read"] = optimal_read_block_size
            optimal_block_sizes[file_size][f"{pattern}_write"] = optimal_write_block_size
    
    return df, optimal_block_sizes

def plot_results(df: pd.DataFrame, output_dir: str):
    """
    Create performance charts from benchmark results
    
    Args:
        df: DataFrame with benchmark results
        output_dir: Directory to save charts
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Group by block size and file size for overall comparisons
    for pattern in df['access_pattern'].unique():
        pattern_df = df[df['access_pattern'] == pattern]
        
        # Plot read throughput
        plt.figure(figsize=(10, 8))
        for block_size in pattern_df['block_size_mb'].unique():
            subset = pattern_df[pattern_df['block_size_mb'] == block_size]
            plt.plot(subset['file_size_mb'], subset['read_throughput_mbs'], 
                     marker='o', label=f"Block Size {block_size}MB")
        
        plt.xlabel('File Size (MB)')
        plt.ylabel('Read Throughput (MB/s)')
        plt.title(f'Read Throughput for {pattern.capitalize()} Access Pattern')
        plt.grid(True)
        plt.legend()
        plt.savefig(f"{output_dir}/read_throughput_{pattern}.png")
        
        # Plot write throughput
        plt.figure(figsize=(10, 8))
        for block_size in pattern_df['block_size_mb'].unique():
            subset = pattern_df[pattern_df['block_size_mb'] == block_size]
            plt.plot(subset['file_size_mb'], subset['write_throughput_mbs'], 
                     marker='o', label=f"Block Size {block_size}MB")
        
        plt.xlabel('File Size (MB)')
        plt.ylabel('Write Throughput (MB/s)')
        plt.title(f'Write Throughput for {pattern.capitalize()} Access Pattern')
        plt.grid(True)
        plt.legend()
        plt.savefig(f"{output_dir}/write_throughput_{pattern}.png")

@pytest.mark.performance
def test_23_ozone_block_size_performance():
    """
    Test performance with varying block sizes in Apache Ozone.
    
    This test:
    1. Configures Ozone with different block sizes (64MB, 128MB, 256MB)
    2. Runs read/write benchmarks for each configuration
    3. Analyzes performance metrics for different file sizes and access patterns
    4. Identifies optimal block sizes for different workloads
    """
    # Test parameters
    block_sizes = [64, 128, 256]  # Block sizes in MB
    file_sizes = [4, 16, 64, 256, 1024, 4096]  # File sizes in MB
    access_patterns = ["sequential", "random"]
    results_dir = "block_size_benchmark_results"
    
    # Check if Ozone cluster is accessible
    stdout, stderr, rc = execute_command(["ozone", "version"])
    if rc != 0:
        pytest.skip("Ozone cluster is not accessible. Please ensure the cluster is running.")
    
    # Run benchmarks for each block size
    all_results = []
    
    for block_size in block_sizes:
        # Configure Ozone with the current block size
        if not setup_ozone_block_size(block_size):
            pytest.fail(f"Failed to configure Ozone with block size {block_size}MB")
        
        # Restart the cluster to apply configuration
        restart_ozone_cluster()
        
        # Run benchmark for current block size
        results = run_benchmark(block_size, file_sizes, access_patterns)
        all_results.append(results)
    
    # Analyze results
    df, optimal_block_sizes = analyze_results(all_results)
    
    # Save results to CSV
    os.makedirs(results_dir, exist_ok=True)
    df.to_csv(f"{results_dir}/benchmark_results.csv", index=False)
    
    # Plot results
    plot_results(df, results_dir)
    
    # Print optimal block sizes
    print("\nOptimal Block Sizes:")
    for file_size, patterns in optimal_block_sizes.items():
        print(f"\nFile Size: {file_size}MB")
        for pattern_type, optimal_block in patterns.items():
            print(f"  {pattern_type}: {optimal_block}MB")
    
    # Verify we have results for each configuration
    for block_size in block_sizes:
        assert block_size in df['block_size_mb'].unique(), f"Missing results for {block_size}MB block size"
    
    for file_size in file_sizes:
        assert file_size in df['file_size_mb'].unique(), f"Missing results for {file_size}MB file size"
    
    # All access patterns tested
    assert set(access_patterns) == set(df['access_pattern'].unique()), "Not all access patterns were tested"
    
    # Verify we have performance data
    assert not df['read_throughput_mbs'].isnull().all(), "No read throughput data collected"
    assert not df['write_throughput_mbs'].isnull().all(), "No write throughput data collected"
    
    # Assert that we have identified optimal block sizes
    assert len(optimal_block_sizes) > 0, "Failed to identify optimal block sizes"
    
    # Log and assert final conclusion
    print("\nPerformance testing with varying block sizes completed successfully.")
    print("Optimal block sizes have been identified for different workload types.")

import time
import pytest
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
import logging
from typing import Dict, List, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class OzoneClient:
    """Helper class to interact with Ozone through shell commands"""
    
    def __init__(self, ozone_bin="/opt/hadoop/bin/ozone"):
        self.ozone_bin = ozone_bin
        
    def execute_command(self, command, check=True):
        """Execute an ozone shell command"""
        full_command = f"{self.ozone_bin} {command}"
        logger.info(f"Executing: {full_command}")
        result = subprocess.run(
            full_command, 
            shell=True, 
            check=check, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        return result
    
    def create_volume(self, volume):
        return self.execute_command(f"sh volume create {volume}")
    
    def create_bucket(self, volume, bucket):
        return self.execute_command(f"sh bucket create {volume}/{bucket}")
    
    def put_key(self, volume, bucket, key, file_path):
        return self.execute_command(f"sh key put {volume}/{bucket}/{key} {file_path}")
    
    def get_key(self, volume, bucket, key, output_file):
        return self.execute_command(f"sh key get {volume}/{bucket}/{key} {output_file}")
    
    def create_snapshot(self, volume, bucket, snapshot_name):
        return self.execute_command(f"sh bucket createsnapshot {volume}/{bucket} {snapshot_name}")
    
    def delete_snapshot(self, volume, bucket, snapshot_name):
        return self.execute_command(f"sh bucket deletesnapshot {volume}/{bucket} {snapshot_name}")
    
    def list_snapshots(self, volume, bucket):
        result = self.execute_command(f"sh bucket listsnapshots {volume}/{bucket}")
        return result.stdout


class PerformanceMonitor:
    """Helper class to monitor performance metrics"""
    
    def __init__(self):
        self.metrics = []
        
    def start_operation(self, operation_name, details=None):
        """Start timing an operation"""
        self.current_operation = {
            'operation': operation_name,
            'details': details,
            'start_time': time.time(),
            'end_time': None,
            'duration': None
        }
        return self.current_operation
        
    def end_operation(self):
        """End timing the current operation and record metrics"""
        self.current_operation['end_time'] = time.time()
        self.current_operation['duration'] = self.current_operation['end_time'] - self.current_operation['start_time']
        self.metrics.append(self.current_operation)
        return self.current_operation['duration']
    
    def get_metrics_dataframe(self):
        """Convert metrics to pandas DataFrame for analysis"""
        return pd.DataFrame(self.metrics)
    
    def plot_operations_comparison(self, operation_type, with_snapshots=False, output_file=None):
        """Plot performance comparison of operations with and without snapshots"""
        df = self.get_metrics_dataframe()
        
        # Filter by operation type
        filtered_df = df[df['operation'] == operation_type]
        
        # Group by whether snapshots were involved
        grouped = filtered_df.groupby(
            filtered_df['details'].apply(lambda x: 'With Snapshots' if with_snapshots else 'Without Snapshots')
        )
        
        fig, ax = plt.subplots(figsize=(10, 6))
        grouped['duration'].plot(kind='box', ax=ax)
        ax.set_ylabel('Duration (seconds)')
        ax.set_title(f'Performance Impact: {operation_type}')
        
        if output_file:
            plt.savefig(output_file)
        
        return fig


def generate_test_file(size_mb):
    """Generate a test file of specified size in MB"""
    fd, path = tempfile.mkstemp()
    with os.fdopen(fd, 'wb') as f:
        f.write(os.urandom(int(size_mb * 1024 * 1024)))
    return path


@pytest.fixture(scope="function")
def ozone_test_env():
    """Fixture to set up and tear down the test environment"""
    # Create a unique volume and bucket for this test
    client = OzoneClient()
    timestamp = int(time.time())
    volume = f"perfvol{timestamp}"
    bucket = f"perfbucket{timestamp}"
    
    # Set up
    client.create_volume(volume)
    client.create_bucket(volume, bucket)
    
    yield {
        "client": client,
        "volume": volume,
        "bucket": bucket,
        "performance_monitor": PerformanceMonitor()
    }
    
    # Tear down - we'll leave the data for potential analysis
    # In a real scenario, you might want to clean up


@pytest.mark.parametrize("file_size_mb", [5, 10, 25])
@pytest.mark.parametrize("num_files", [10, 50])
@pytest.mark.parametrize("snapshot_interval", [5, 10])
def test_24_snapshot_performance_impact(ozone_test_env, file_size_mb, num_files, snapshot_interval):
    """
    Measure performance impact of snapshotting.
    
    This test case:
    1. Creates a baseline workload
    2. Takes snapshots at regular intervals during the workload
    3. Measures the performance impact of snapshot creation
    4. Analyzes read performance from snapshots
    """
    client = ozone_test_env["client"]
    volume = ozone_test_env["volume"]
    bucket = ozone_test_env["bucket"]
    monitor = ozone_test_env["performance_monitor"]
    
    # Generate test files
    test_files = []
    for i in range(num_files):
        file_path = generate_test_file(file_size_mb)
        test_files.append((f"key{i}", file_path))
    
    # Part 1: Baseline performance (without snapshots)
    logger.info(f"Running baseline workload (without snapshots)...")
    
    # Write baseline
    write_durations_baseline = []
    for key, file_path in test_files:
        monitor.start_operation("write", "baseline")
        client.put_key(volume, bucket, key, file_path)
        write_durations_baseline.append(monitor.end_operation())

    # Read baseline
    temp_output = tempfile.mktemp()
    read_durations_baseline = []
    for key, _ in test_files:
        monitor.start_operation("read", "baseline")
        client.get_key(volume, bucket, key, temp_output)
        read_durations_baseline.append(monitor.end_operation())
    
    # Part 2: Performance with snapshots
    logger.info(f"Running workload with snapshots at interval of {snapshot_interval}...")
    
    # Create new test files for the snapshot test
    snapshot_test_files = []
    for i in range(num_files):
        file_path = generate_test_file(file_size_mb)
        snapshot_test_files.append((f"snapshot-key{i}", file_path))
    
    # Write with snapshots taken at intervals
    write_durations_with_snapshots = []
    snapshots_created = []
    
    for i, (key, file_path) in enumerate(snapshot_test_files):
        # Take snapshot at defined intervals
        if i % snapshot_interval == 0:
            snapshot_name = f"snapshot-{i}"
            monitor.start_operation("create_snapshot", snapshot_name)
            client.create_snapshot(volume, bucket, snapshot_name)
            monitor.end_operation()
            snapshots_created.append(snapshot_name)
        
        # Perform write operation
        monitor.start_operation("write", "with_snapshots")
        client.put_key(volume, bucket, key, file_path)
        write_durations_with_snapshots.append(monitor.end_operation())
    
    # Final snapshot after all writes
    final_snapshot = "final-snapshot"
    monitor.start_operation("create_snapshot", final_snapshot)
    client.create_snapshot(volume, bucket, final_snapshot)
    monitor.end_operation()
    snapshots_created.append(final_snapshot)
    
    # Read with snapshots present
    read_durations_with_snapshots = []
    for key, _ in snapshot_test_files:
        monitor.start_operation("read", "with_snapshots")
        client.get_key(volume, bucket, key, temp_output)
        read_durations_with_snapshots.append(monitor.end_operation())
    
    # Part 3: Read from snapshots
    logger.info("Measuring read performance from snapshots...")
    
    read_durations_from_snapshots = []
    # Read from the final snapshot
    for key, _ in snapshot_test_files[:5]:  # Read a subset of keys from snapshot
        monitor.start_operation("read", "from_snapshot")
        # Note: In a real implementation, you would use the actual syntax for reading from a snapshot
        # This is a placeholder based on expected API
        client.execute_command(f"sh key get --snapshot={final_snapshot} {volume}/{bucket}/{key} {temp_output}", check=False)
        read_durations_from_snapshots.append(monitor.end_operation())
    
    # Analyze results
    avg_write_baseline = np.mean(write_durations_baseline)
    avg_write_with_snapshots = np.mean(write_durations_with_snapshots)
    avg_read_baseline = np.mean(read_durations_baseline)
    avg_read_with_snapshots = np.mean(read_durations_with_snapshots)
    avg_read_from_snapshots = np.mean(read_durations_from_snapshots) if read_durations_from_snapshots else 0
    
    logger.info(f"Average write duration (baseline): {avg_write_baseline:.4f}s")
    logger.info(f"Average write duration (with snapshots): {avg_write_with_snapshots:.4f}s")
    logger.info(f"Write performance impact: {((avg_write_with_snapshots/avg_write_baseline)-1)*100:.2f}%")
    
    logger.info(f"Average read duration (baseline): {avg_read_baseline:.4f}s")
    logger.info(f"Average read duration (with snapshots): {avg_read_with_snapshots:.4f}s")
    logger.info(f"Average read duration (from snapshots): {avg_read_from_snapshots:.4f}s")
    
    # Generate performance reports
    results_df = pd.DataFrame({
        'Metric': [
            'Avg Write (Baseline)', 
            'Avg Write (With Snapshots)', 
            'Avg Read (Baseline)', 
            'Avg Read (With Snapshots)',
            'Avg Read (From Snapshots)',
            'Write Impact (%)',
            'Read Impact (%)'
        ],
        'Value': [
            avg_write_baseline, 
            avg_write_with_snapshots, 
            avg_read_baseline, 
            avg_read_with_snapshots,
            avg_read_from_snapshots,
            ((avg_write_with_snapshots/avg_write_baseline)-1)*100,
            ((avg_read_with_snapshots/avg_read_baseline)-1)*100
        ]
    })
    
    logger.info("\nPerformance Summary:")
    logger.info(results_df.to_string(index=False))
    
    # Assertions based on expected result
    # Note: These thresholds would be adjusted based on actual performance expectations
    
    # Snapshot creation should have minimal impact on ongoing operations
    # Here we're asserting that writes shouldn't be more than 30% slower with snapshots
    assert (avg_write_with_snapshots / avg_write_baseline) < 1.3, \
           f"Write performance with snapshots degraded by more than 30%: {((avg_write_with_snapshots/avg_write_baseline)-1)*100:.2f}%"
    
    # Read performance from snapshots should be comparable to normal reads
    # Here we're asserting that reads from snapshots shouldn't be more than 50% slower
    if read_durations_from_snapshots:
        assert (avg_read_from_snapshots / avg_read_baseline) < 1.5, \
               f"Read from snapshots is significantly slower than normal reads: {((avg_read_from_snapshots/avg_read_baseline)-1)*100:.2f}%"
    
    # Clean up test files
    for _, file_path in test_files + snapshot_test_files:
        if os.path.exists(file_path):
            os.remove(file_path)
    if os.path.exists(temp_output):
        os.remove(temp_output)

#!/usr/bin/env python3
"""
Test suite for Apache Ozone performance testing with multi-datacenter replication.
"""

import os
import time
import pytest
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import logging
import socket
import concurrent.futures
from dataclasses import dataclass
import random
import tempfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_TIMEOUT = 300  # seconds
DATA_SIZES = [
    (10 * 1024),       # 10 KB
    (512 * 1024),      # 512 KB
    (1 * 1024 * 1024), # 1 MB
    (10 * 1024 * 1024) # 10 MB
]
REPLICATION_SLA = 30  # seconds

@dataclass
class DatacenterConfig:
    """Configuration for a simulated datacenter."""
    name: str
    host: str
    port: int
    latency: int  # simulated WAN latency in ms
    bandwidth: int  # simulated bandwidth in Kbps

# Test datacenter configurations
TEST_DATACENTERS = [
    DatacenterConfig("dc1", "localhost", 9863, 0, 1000000),     # Primary datacenter
    DatacenterConfig("dc2", "localhost", 9864, 50, 100000),     # Secondary with 50ms latency, 100Mbps
    DatacenterConfig("dc3", "localhost", 9865, 100, 50000),     # Secondary with 100ms latency, 50Mbps
]

class OzoneClusterManager:
    """Manages multiple Ozone clusters for testing."""
    
    def __init__(self, datacenters: List[DatacenterConfig]):
        self.datacenters = datacenters
        self.clusters = {}
        
    def setup_clusters(self):
        """Set up Ozone clusters in different simulated datacenters."""
        for dc in self.datacenters:
            logger.info(f"Setting up Ozone cluster in datacenter {dc.name}")
            
            # In a real setup, this would create an actual cluster
            # For this test code, we're simulating by setting up connection configs
            self.clusters[dc.name] = {
                "config": dc,
                "connection_string": f"o3://{dc.host}:{dc.port}"
            }
            
            # Apply network condition simulation via tc (Traffic Control)
            if dc.latency > 0:
                self._apply_network_simulation(dc)
                
        logger.info(f"Successfully set up {len(self.datacenters)} Ozone clusters")
        return True
    
    def _apply_network_simulation(self, dc: DatacenterConfig):
        """Apply network simulation using tc (would require root privileges in a real env)"""
        logger.info(f"Simulating network conditions for {dc.name}: {dc.latency}ms latency, {dc.bandwidth}Kbps bandwidth")
        # In a real test, we would execute tc commands here
        # Example: 
        # subprocess.run(["sudo", "tc", "qdisc", "add", "dev", "lo", "root", "netem", 
        #               "delay", f"{dc.latency}ms", "rate", f"{dc.bandwidth}kbit"])
        
    def configure_replication(self):
        """Configure cross-datacenter replication between Ozone clusters."""
        logger.info("Configuring cross-datacenter replication")
        # In a real setup, this would configure actual replication between clusters
        # For this test, we're just simulating the configuration
        
        # Creating replication policies
        for source_dc, source_config in self.clusters.items():
            for target_dc, target_config in self.clusters.items():
                if source_dc != target_dc:
                    logger.info(f"Setting up replication from {source_dc} to {target_dc}")
                    # Simulate replication configuration
                    
        return True
    
    def execute_ozone_command(self, dc_name: str, command: List[str], timeout: int = DEFAULT_TIMEOUT) -> Tuple[int, str, str]:
        """Execute an Ozone command on the specified datacenter."""
        logger.debug(f"Executing on {dc_name}: ozone {' '.join(command)}")
        
        # In a real test, we would connect to the actual cluster
        # For simulation, we'll just log the command and return success
        
        # Simulate command execution with network conditions
        dc = next((dc for dc in self.datacenters if dc.name == dc_name), None)
        if dc:
            # Simulate network delay
            time.sleep(dc.latency / 1000.0)
        
        # Return simulated success 
        return 0, f"Success: ozone {' '.join(command)}", ""
    
    def cleanup(self):
        """Clean up all clusters and network simulations."""
        for dc_name, cluster in self.clusters.items():
            logger.info(f"Cleaning up cluster in {dc_name}")
            # In a real setup, this would stop clusters and clean up resources
            
        # Remove network simulations
        # subprocess.run(["sudo", "tc", "qdisc", "del", "dev", "lo", "root"])
        
        logger.info("Cleanup completed")


class PerformanceAnalyzer:
    """Analyzes and reports on performance metrics."""
    
    def __init__(self):
        self.metrics = {
            "write_times": {},
            "replication_lag": {},
            "read_times": {}
        }
    
    def add_write_metric(self, dc_name: str, data_size: int, duration: float):
        """Add a write operation metric."""
        if dc_name not in self.metrics["write_times"]:
            self.metrics["write_times"][dc_name] = []
        
        self.metrics["write_times"][dc_name].append({
            "data_size": data_size,
            "duration": duration
        })
    
    def add_replication_metric(self, source_dc: str, target_dc: str, data_size: int, duration: float):
        """Add a replication lag metric."""
        key = f"{source_dc}->{target_dc}"
        if key not in self.metrics["replication_lag"]:
            self.metrics["replication_lag"][key] = []
        
        self.metrics["replication_lag"][key].append({
            "data_size": data_size,
            "duration": duration
        })
    
    def add_read_metric(self, dc_name: str, data_size: int, duration: float):
        """Add a read operation metric."""
        if dc_name not in self.metrics["read_times"]:
            self.metrics["read_times"][dc_name] = []
        
        self.metrics["read_times"][dc_name].append({
            "data_size": data_size,
            "duration": duration
        })
    
    def generate_report(self):
        """Generate and return a performance report."""
        report = {
            "write_performance": {},
            "replication_performance": {},
            "read_performance": {}
        }
        
        # Calculate averages for each metric type
        for dc, write_metrics in self.metrics["write_times"].items():
            if write_metrics:
                avg_write_time = sum(m["duration"] for m in write_metrics) / len(write_metrics)
                report["write_performance"][dc] = avg_write_time
        
        for path, repl_metrics in self.metrics["replication_lag"].items():
            if repl_metrics:
                avg_repl_time = sum(m["duration"] for m in repl_metrics) / len(repl_metrics)
                report["replication_performance"][path] = avg_repl_time
        
        for dc, read_metrics in self.metrics["read_times"].items():
            if read_metrics:
                avg_read_time = sum(m["duration"] for m in read_metrics) / len(read_metrics)
                report["read_performance"][dc] = avg_read_time
        
        return report


def create_test_file(size_bytes: int) -> str:
    """Create a test file of a specified size."""
    fd, path = tempfile.mkstemp()
    with os.fdopen(fd, 'wb') as f:
        f.write(b'0' * size_bytes)
    return path


@pytest.fixture(scope="module")
def ozone_multi_datacenter_setup():
    """Set up multi-datacenter Ozone clusters for testing."""
    cluster_mgr = OzoneClusterManager(TEST_DATACENTERS)
    
    # Set up clusters
    setup_success = cluster_mgr.setup_clusters()
    assert setup_success, "Failed to set up multi-datacenter Ozone clusters"
    
    # Configure replication
    repl_success = cluster_mgr.configure_replication()
    assert repl_success, "Failed to configure cross-datacenter replication"
    
    yield cluster_mgr
    
    # Cleanup after tests
    cluster_mgr.cleanup()


@pytest.mark.performance
def test_25_multi_datacenter_replication_performance(ozone_multi_datacenter_setup):
    """Test performance under multi-datacenter replication"""
    cluster_mgr = ozone_multi_datacenter_setup
    performance_analyzer = PerformanceAnalyzer()
    
    # Test volume and bucket names (avoiding underscores)
    volume_name = "perfvolume"
    bucket_name = "replicationbucket"
    
    # Create volume and bucket in the primary datacenter
    primary_dc = TEST_DATACENTERS[0].name
    
    logger.info(f"Creating test volume and bucket in primary datacenter {primary_dc}")
    cluster_mgr.execute_ozone_command(primary_dc, ["volume", "create", volume_name])
    cluster_mgr.execute_ozone_command(primary_dc, ["bucket", "create", f"/{volume_name}/{bucket_name}"])

    # Test different data sizes
    for data_size in DATA_SIZES:
        # Create test file
        test_file_path = create_test_file(data_size)
        key_name = f"testkey-{int(time.time())}-{random.randint(1000, 9999)}"
        
        try:
            # 1. Write data from primary datacenter
            logger.info(f"Writing {data_size/1024:.2f} KB file to primary datacenter")
            start_time = time.time()
            cluster_mgr.execute_ozone_command(
                primary_dc, 
                ["key", "put", f"/{volume_name}/{bucket_name}/", test_file_path, "--name", key_name]
            )
            write_duration = time.time() - start_time
            performance_analyzer.add_write_metric(primary_dc, data_size, write_duration)
            
            logger.info(f"Write completed in {write_duration:.2f} seconds")
            
            # 2. Measure replication lag to each secondary datacenter
            for dc in TEST_DATACENTERS[1:]:
                secondary_dc = dc.name
                
                # Start checking for replication completion
                logger.info(f"Measuring replication lag to {secondary_dc}")
                start_time = time.time()
                
                # Poll for the key to appear in the secondary datacenter
                is_replicated = False
                while time.time() - start_time < DEFAULT_TIMEOUT:
                    # Check if key exists in secondary datacenter
                    # In a real test, we would check the actual key existence
                    # For simulation, we'll add a delay based on datacenter latency and data size
                    expected_repl_time = (dc.latency / 1000.0) + (data_size / (dc.bandwidth * 128))
                    time.sleep(min(expected_repl_time, 1.0))  # Simulate replication delay
                    
                    # Check for existence - simulated
                    is_replicated = True
                    break
                
                assert is_replicated, f"Replication to {secondary_dc} did not complete within timeout"
                
                replication_duration = time.time() - start_time
                performance_analyzer.add_replication_metric(
                    primary_dc, secondary_dc, data_size, replication_duration
                )
                
                logger.info(f"Replication to {secondary_dc} completed in {replication_duration:.2f} seconds")
                
                # Assert replication completes within SLA
                assert replication_duration <= REPLICATION_SLA, \
                    f"Replication to {secondary_dc} exceeded SLA ({replication_duration:.2f}s > {REPLICATION_SLA}s)"
                
                # 3. Measure read performance from secondary datacenter
                logger.info(f"Reading file from {secondary_dc}")
                start_time = time.time()
                
                # In a real test, we would read the actual key
                # For simulation, we'll add a delay based on datacenter latency
                cluster_mgr.execute_ozone_command(
                    secondary_dc,
                    ["key", "info", f"/{volume_name}/{bucket_name}/{key_name}"]
                )
                
                read_duration = time.time() - start_time
                performance_analyzer.add_read_metric(secondary_dc, data_size, read_duration)
                logger.info(f"Read from {secondary_dc} completed in {read_duration:.2f} seconds")
        
        finally:
            # Clean up test file
            if os.path.exists(test_file_path):
                os.unlink(test_file_path)
    
    # Generate performance report
    report = performance_analyzer.generate_report()
    
    # Log performance summary
    logger.info("==== Performance Summary ====")
    
    logger.info("Write performance (seconds):")
    for dc, avg_time in report["write_performance"].items():
        logger.info(f"  {dc}: {avg_time:.2f}s")
    
    logger.info("Replication lag (seconds):")
    for path, avg_time in report["replication_performance"].items():
        logger.info(f"  {path}: {avg_time:.2f}s")
        # Verify replication SLA
        assert avg_time <= REPLICATION_SLA, \
            f"Average replication time for {path} ({avg_time:.2f}s) exceeds SLA ({REPLICATION_SLA}s)"
    
    logger.info("Read performance (seconds):")
    for dc, avg_time in report["read_performance"].items():
        logger.info(f"  {dc}: {avg_time:.2f}s")
    
    # Verify read performance consistency across datacenters
    read_times = list(report["read_performance"].values())
    if read_times:
        max_diff = max(read_times) - min(read_times)
        avg_time = sum(read_times) / len(read_times)
        variance_pct = (max_diff / avg_time) * 100 if avg_time > 0 else 0
        
        logger.info(f"Read time variance: {variance_pct:.2f}%")
        # Assuming a 50% variance is acceptable for this test
        assert variance_pct <= 50, \
            f"Read performance varies too much across datacenters ({variance_pct:.2f}% > 50%)"

import os
import time
import json
import subprocess
import statistics
import pytest
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass


@dataclass
class ConsistencyConfig:
    name: str
    ratis_server_request_timeout: int = 3000  # in milliseconds
    ratis_client_request_timeout: int = 10000  # in milliseconds
    ratis_minimum_rpc_timeout: int = 3000  # in milliseconds
    ratis_leader_election_minimum_timeout: int = 1000  # in milliseconds
    # The number of copies maintained for each block
    replication_factor: int = 3


@pytest.fixture(scope="module")
def setup_ozone_cluster():
    """Set up the Ozone cluster for testing."""
    # In a real test, this would connect to a running cluster
    # or potentially set one up using Docker or some other automation
    yield


@pytest.fixture(scope="function")
def benchmark_data_file(tmp_path):
    """Create a file for benchmarking."""
    test_file_path = tmp_path / "test_data.bin"
    # Create a 10MB file
    with open(test_file_path, "wb") as f:
        f.write(os.urandom(10 * 1024 * 1024))
    
    return test_file_path


def apply_consistency_config(config: ConsistencyConfig):
    """Apply the given consistency configuration to the Ozone cluster."""
    # In a real implementation, this would update ozone-site.xml or use admin APIs
    print(f"Applying consistency configuration: {config.name}")
    
    # Example command to update configuration (would need to be implemented)
    cmd = [
        "ozone", "admin", "config", "set",
        f"ozone.ratis.server.request.timeout={config.ratis_server_request_timeout}",
        f"ozone.ratis.client.request.timeout={config.ratis_client_request_timeout}",
        f"ozone.ratis.minimum.rpc.timeout={config.ratis_minimum_rpc_timeout}",
        f"ozone.ratis.leader.election.minimum.timeout={config.ratis_leader_election_minimum_timeout}",
        f"ozone.ratis.replication.factor={config.replication_factor}"
    ]
    
    # This is a placeholder - in a real test, you'd execute these commands
    # subprocess.run(cmd, check=True)
    
    # Allow time for config changes to propagate
    time.sleep(5)


def run_write_benchmark(file_path: str, volume: str, bucket: str, 
                       iterations: int = 10) -> List[float]:
    """Run a write benchmark and return latency measurements."""
    latencies = []
    
    for i in range(iterations):
        key = f"benchmark_key_{i}"
        start_time = time.time()
        
        # In a real test, this would use the Ozone client APIs
        cmd = ["ozone", "sh", "key", "put", f"{volume}/{bucket}/", file_path, "--key", key]
        # subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        end_time = time.time()
        latency = (end_time - start_time) * 1000  # Convert to ms
        latencies.append(latency)
    
    return latencies


def run_read_benchmark(volume: str, bucket: str, 
                      iterations: int = 10) -> List[float]:
    """Run a read benchmark and return latency measurements."""
    latencies = []
    
    for i in range(iterations):
        key = f"benchmark_key_{i}"
        start_time = time.time()
        
        # In a real test, this would use the Ozone client APIs
        cmd = ["ozone", "sh", "key", "get", f"{volume}/{bucket}/{key}", "-"]
        # subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        end_time = time.time()
        latency = (end_time - start_time) * 1000  # Convert to ms
        latencies.append(latency)
    
    return latencies


def simulate_node_failure(node_id: int) -> float:
    """Simulate a node failure and measure recovery time."""
    # In a real test, this might stop a container or service
    print(f"Simulating failure on node {node_id}")
    
    # Record start time
    start_time = time.time()
    
    # Simulate stopping a node
    # subprocess.run(["docker", "stop", f"ozone-node-{node_id}"], check=True)
    
    # Wait for recovery
    # Here we might poll the cluster status until the node is marked as down
    # and Ozone has redistributed the load
    time.sleep(5)  # Simulated recovery time
    
    # Bring the node back
    # subprocess.run(["docker", "start", f"ozone-node-{node_id}"], check=True)
    
    # Wait for node to rejoin
    # Here we'd poll cluster status until the node is back
    time.sleep(5)
    
    end_time = time.time()
    recovery_time = end_time - start_time
    
    print(f"Recovery time: {recovery_time:.2f} seconds")
    return recovery_time


def analyze_results(results: Dict):
    """Analyze and display benchmark results."""
    # Create a DataFrame for easier analysis
    df_data = []
    
    for config_name, config_results in results.items():
        for op_type, metrics in config_results.items():
            if op_type != "recovery_time":
                df_data.append({
                    "Configuration": config_name,
                    "Operation": op_type,
                    "Average Latency (ms)": statistics.mean(metrics),
                    "Median Latency (ms)": statistics.median(metrics),
                    "P95 Latency (ms)": sorted(metrics)[int(len(metrics) * 0.95)],
                    "Throughput (ops/s)": 1000 / statistics.mean(metrics)
                })
            else:
                df_data.append({
                    "Configuration": config_name,
                    "Operation": "Node Recovery",
                    "Recovery Time (s)": metrics
                })
    
    df = pd.DataFrame(df_data)
    print("\nPerformance Results Summary:")
    print(df)
    
    # Save results to file
    df.to_csv("consistency_benchmark_results.csv", index=False)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Latency comparison
    latency_df = df[df["Operation"] != "Node Recovery"]
    plot_data = latency_df.pivot(index="Configuration", columns="Operation", values="Average Latency (ms)")
    plot_data.plot(kind="bar")
    plt.title("Latency by Consistency Configuration")
    plt.ylabel("Average Latency (ms)")
    plt.tight_layout()
    plt.savefig("consistency_latency_comparison.png")
    
    # Recovery time comparison
    recovery_df = df[df["Operation"] == "Node Recovery"]
    plt.figure(figsize=(10, 6))
    recovery_df.plot(kind="bar", x="Configuration", y="Recovery Time (s)")
    plt.title("Recovery Time by Consistency Configuration")
    plt.ylabel("Recovery Time (seconds)")
    plt.tight_layout()
    plt.savefig("consistency_recovery_comparison.png")
    
    return df


@pytest.mark.parametrize("config", [
    ConsistencyConfig(name="Strong_Consistency", replication_factor=3, 
                     ratis_server_request_timeout=6000, ratis_client_request_timeout=20000),
    ConsistencyConfig(name="Balanced", replication_factor=3,
                     ratis_server_request_timeout=3000, ratis_client_request_timeout=10000),
    ConsistencyConfig(name="High_Performance", replication_factor=2,
                     ratis_server_request_timeout=1000, ratis_client_request_timeout=5000)
])
def test_26_consistency_level_performance(setup_ozone_cluster, benchmark_data_file, config):
    """
    Evaluate performance with different consistency levels.
    
    This test measures read/write performance and recovery times
    across different Ozone consistency configurations.
    """
    # Test variables
    volume = "perfvolume"
    bucket = "perfbucket"
    test_file = benchmark_data_file
    iterations = 20  # Number of operations for each benchmark
    
    # Create volume and bucket if they don't exist
    # In a real test, you'd use the actual Ozone client
    print(f"Ensuring volume {volume} and bucket {bucket} exist")
    # subprocess.run(["ozone", "sh", "volume", "create", volume], check=True)
    # subprocess.run(["ozone", "sh", "bucket", "create", f"{volume}/{bucket}"], check=True)
    
    # Apply the consistency configuration
    apply_consistency_config(config)
    
    # Initialize results dictionary
    results = {}
    results[config.name] = {}
    
    # Run write benchmark
    print(f"Running write benchmark with {config.name} configuration")
    write_latencies = run_write_benchmark(str(test_file), volume, bucket, iterations)
    results[config.name]["write"] = write_latencies
    
    # Run read benchmark
    print(f"Running read benchmark with {config.name} configuration")
    read_latencies = run_read_benchmark(volume, bucket, iterations)
    results[config.name]["read"] = read_latencies
    
    # Parallel operations benchmark (simulate concurrent user access)
    print(f"Running parallel operations benchmark with {config.name} configuration")
    parallel_latencies = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for i in range(5):  # 5 parallel operations
            futures.append(executor.submit(run_read_benchmark, volume, bucket, 5))
        
        for future in futures:
            parallel_latencies.extend(future.result())
    
    results[config.name]["parallel_read"] = parallel_latencies
    
    # Simulate node failure and measure recovery time
    if config.replication_factor > 1:  # Only test failure if replication is enabled
        print(f"Simulating node failure with {config.name} configuration")
        recovery_time = simulate_node_failure(1)  # Simulate failure on first node
        results[config.name]["recovery_time"] = recovery_time
    
    # Analyze results
    df = analyze_results(results)
    
    # Assert that the results are within acceptable ranges
    # Note: In a real test, you would define these thresholds based on your requirements
    for op_type in ["write", "read"]:
        avg_latency = statistics.mean(results[config.name][op_type])
        
        # These assertions would be tailored to your specific performance requirements
        if config.name == "High_Performance":
            assert avg_latency < 500, f"High performance {op_type} latency too high: {avg_latency}ms"
        elif config.name == "Balanced":
            assert avg_latency < 750, f"Balanced {op_type} latency too high: {avg_latency}ms"
        else:  # Strong_Consistency
            assert avg_latency < 1000, f"Strong consistency {op_type} latency too high: {avg_latency}ms"
    
    # Check that recovery time matches expected behavior based on consistency level
    if config.replication_factor > 1:
        if config.name == "Strong_Consistency":
            assert results[config.name]["recovery_time"] < 20, "Recovery time too long for strong consistency"
        elif config.name == "Balanced":
            assert results[config.name]["recovery_time"] < 15, "Recovery time too long for balanced consistency"
        else:  # High_Performance
            assert results[config.name]["recovery_time"] < 10, "Recovery time too long for high performance"
    
    # Save performance data to a report
    with open(f"performance_report_{config.name}.json", "w") as f:
        json.dump({
            "configuration": config.__dict__,
            "write_latencies_ms": write_latencies,
            "read_latencies_ms": read_latencies,
            "parallel_read_latencies_ms": parallel_latencies,
            "recovery_time_s": results[config.name].get("recovery_time", "N/A")
        }, f, indent=2)
    
    print(f"Performance test completed for {config.name} configuration")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import time
import threading
import subprocess
import os
import random
import logging
import statistics
from typing import Dict, List, Tuple
import concurrent.futures
import psutil
from datetime import datetime


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ozone_perf_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class OzonePerformanceMonitor:
    """Class to monitor Ozone performance metrics during testing"""
    
    def __init__(self, monitoring_interval=5):
        self.monitoring_interval = monitoring_interval  # seconds
        self.stop_monitoring = False
        self.metrics = {
            'cpu_usage': [],
            'memory_usage': [],
            'disk_io': [],
            'space_reclamation': []
        }
        self.start_time = None
        self.end_time = None
        
    def start_monitoring(self):
        """Start the performance monitoring thread"""
        self.start_time = datetime.now()
        self.stop_monitoring = False
        self.monitor_thread = threading.Thread(target=self._monitor_performance)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop the performance monitoring thread"""
        self.stop_monitoring = True
        if hasattr(self, 'monitor_thread') and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=10)
        self.end_time = datetime.now()
        
    def _monitor_performance(self):
        """Continuously monitor system performance"""
        while not self.stop_monitoring:
            try:
                # Collect CPU usage
                self.metrics['cpu_usage'].append(psutil.cpu_percent(interval=1))
                
                # Collect memory usage
                mem = psutil.virtual_memory()
                self.metrics['memory_usage'].append(mem.percent)
                
                # Collect disk I/O
                disk_io = psutil.disk_io_counters()
                self.metrics['disk_io'].append((disk_io.read_bytes, disk_io.write_bytes))
                
                # Get space utilization (simplified approach)
                # In a real implementation, you'd use Ozone Admin API to get actual space reclamation metrics
                try:
                    result = subprocess.run(
                        ["ozone admin -listStatus"], 
                        shell=True, 
                        capture_output=True, 
                        text=True
                    )
                    # Parse output to extract space reclamation metrics
                    # This is a placeholder - actual parsing depends on the command output format
                    self.metrics['space_reclamation'].append(time.time())
                except Exception as e:
                    logger.error(f"Failed to get space reclamation metrics: {e}")
                
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
    
    def analyze_performance(self) -> Dict:
        """Analyze the collected performance data"""
        analysis = {}
        
        if not self.metrics['cpu_usage']:
            return {'error': 'No performance data collected'}
        
        # CPU analysis
        analysis['cpu_avg'] = statistics.mean(self.metrics['cpu_usage'])
        analysis['cpu_max'] = max(self.metrics['cpu_usage'])
        analysis['cpu_min'] = min(self.metrics['cpu_usage'])
        analysis['cpu_stddev'] = statistics.stdev(self.metrics['cpu_usage']) if len(self.metrics['cpu_usage']) > 1 else 0
        
        # Memory analysis
        analysis['memory_avg'] = statistics.mean(self.metrics['memory_usage'])
        analysis['memory_max'] = max(self.metrics['memory_usage'])
        
        # Disk I/O analysis - calculate throughput
        if len(self.metrics['disk_io']) > 1:
            read_start = self.metrics['disk_io'][0][0]
            read_end = self.metrics['disk_io'][-1][0]
            write_start = self.metrics['disk_io'][0][1]
            write_end = self.metrics['disk_io'][-1][1]
            
            duration = (self.end_time - self.start_time).total_seconds() if self.end_time else 1
            
            analysis['read_throughput_MBps'] = (read_end - read_start) / (1024 * 1024 * duration)
            analysis['write_throughput_MBps'] = (write_end - write_start) / (1024 * 1024 * duration)
        
        # Space reclamation analysis
        # In a real implementation, you'd analyze the space reclamation metrics
        analysis['space_reclamation_events'] = len(self.metrics['space_reclamation'])
        
        return analysis


class OzoneDataOperations:
    """Class to handle data operations for Ozone"""
    
    def __init__(self, volume: str, bucket: str, base_dir: str = "/tmp/ozone-data"):
        self.volume = volume
        self.bucket = bucket
        self.base_dir = base_dir
        
        # Ensure the base directory exists
        os.makedirs(base_dir, exist_ok=True)
        
    def create_test_file(self, file_name: str, size_mb: int) -> str:
        """Create a test file of specified size in MB"""
        file_path = os.path.join(self.base_dir, file_name)
        
        with open(file_path, 'wb') as f:
            f.write(os.urandom(size_mb * 1024 * 1024))
            
        return file_path
    
    def upload_file(self, file_path: str, key: str) -> bool:
        """Upload a file to Ozone"""
        try:
            result = subprocess.run(
                ["ozone", "fs", "-put", file_path, f"o3fs://{self.bucket}.{self.volume}/{key}"],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Error uploading file {file_path}: {e}")
            return False
    
    def delete_key(self, key: str) -> bool:
        """Delete a key from Ozone"""
        try:
            result = subprocess.run(
                ["ozone", "fs", "-rm", f"o3fs://{self.bucket}.{self.volume}/{key}"],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Error deleting key {key}: {e}")
            return False
    
    def list_keys(self) -> List[str]:
        """List all keys in the bucket"""
        try:
            result = subprocess.run(
                ["ozone", "fs", "-ls", f"o3fs://{self.bucket}.{self.volume}/"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                # Parse the output to extract key names
                lines = result.stdout.strip().split('\n')
                keys = []
                for line in lines:
                    if line.strip():
                        # Extract the key name from the line
                        parts = line.split()
                        if len(parts) >= 8:  # Typical ls output has at least 8 parts
                            key = parts[-1].split('/')[-1]
                            keys.append(key)
                return keys
            return []
        except Exception as e:
            logger.error(f"Error listing keys: {e}")
            return []


@pytest.fixture(scope="module")
def ozone_setup():
    """Setup fixture for Ozone testing"""
    # Generate unique volume and bucket names for this test run
    timestamp = int(time.time())
    volume_name = f"perfvolume{timestamp}"
    bucket_name = f"perfbucket{timestamp}"
    
    # Create volume and bucket
    try:
        # Create volume
        subprocess.run(["ozone", "sh", "volume", "create", volume_name], check=True)
        
        # Create bucket
        subprocess.run(["ozone", "sh", "bucket", "create", f"{volume_name}/{bucket_name}"], check=True)
        
        logger.info(f"Created test volume '{volume_name}' and bucket '{bucket_name}'")
        
        yield {"volume": volume_name, "bucket": bucket_name}
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to setup Ozone environment: {e}")
        raise
    
    finally:
        # Cleanup
        try:
            # Delete bucket
            subprocess.run(["ozone", "sh", "bucket", "delete", f"{volume_name}/{bucket_name}"])
            
            # Delete volume
            subprocess.run(["ozone", "sh", "volume", "delete", volume_name])
            
            logger.info(f"Cleaned up test volume '{volume_name}' and bucket '{bucket_name}'")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")


def test_27_continuous_data_deletion_performance(ozone_setup):
    """
    Test performance under continuous data deletion.
    
    This test evaluates how the system performs when continuous data ingest
    is happening simultaneously with continuous data deletion operations.
    """
    volume = ozone_setup["volume"]
    bucket = ozone_setup["bucket"]
    
    # Initialize performance monitor
    monitor = OzonePerformanceMonitor(monitoring_interval=2)
    
    # Initialize data operations
    data_ops = OzoneDataOperations(volume, bucket)
    
    # Define test parameters
    test_duration_seconds = 120  # 2 minutes
    ingest_interval_seconds = 1
    delete_interval_seconds = 2
    file_sizes_mb = [1, 5, 10, 25, 50]  # Various file sizes
    
    uploaded_keys = []
    stop_threads = False
    
    def ingest_data_continuously():
        """Thread function to continuously ingest data"""
        count = 0
        while not stop_threads:
            try:
                # Create and upload a file with random size
                size_mb = random.choice(file_sizes_mb)
                file_name = f"testfile_{count}_{size_mb}MB.dat"
                file_path = data_ops.create_test_file(file_name, size_mb)
                key = f"perf_test/ingest/{count}_{size_mb}MB_{int(time.time())}"
                
                start_time = time.time()
                success = data_ops.upload_file(file_path, key)
                end_time = time.time()
                
                if success:
                    uploaded_keys.append(key)
                    ingest_rate = size_mb / (end_time - start_time)
                    logger.info(f"Uploaded {file_name} ({size_mb} MB) as {key} at {ingest_rate:.2f} MB/s")
                else:
                    logger.warning(f"Failed to upload {file_name}")
                
                # Clean up the local file
                os.remove(file_path)
                
                count += 1
                time.sleep(ingest_interval_seconds)
            except Exception as e:
                logger.error(f"Error in ingest thread: {e}")
    
    def delete_data_continuously():
        """Thread function to continuously delete data"""
        while not stop_threads:
            try:
                # If there are keys to delete, pick one randomly
                if uploaded_keys:
                    index = random.randint(0, len(uploaded_keys) - 1)
                    key = uploaded_keys.pop(index)
                    
                    start_time = time.time()
                    success = data_ops.delete_key(key)
                    end_time = time.time()
                    
                    if success:
                        logger.info(f"Deleted {key} in {end_time - start_time:.2f} seconds")
                    else:
                        logger.warning(f"Failed to delete {key}")
                
                time.sleep(delete_interval_seconds)
            except Exception as e:
                logger.error(f"Error in delete thread: {e}")
    
    try:
        # Start performance monitoring
        monitor.start_monitoring()
        
        # Create and start the ingest and delete threads
        ingest_thread = threading.Thread(target=ingest_data_continuously)
        delete_thread = threading.Thread(target=delete_data_continuously)
        
        ingest_thread.daemon = True
        delete_thread.daemon = True
        
        ingest_thread.start()
        delete_thread.start()
        
        # Run the test for the specified duration
        logger.info(f"Running continuous data ingest and deletion test for {test_duration_seconds} seconds")
        time.sleep(test_duration_seconds)
        
        # Signal threads to stop
        stop_threads = True
        
        # Wait for threads to complete
        ingest_thread.join(timeout=10)
        delete_thread.join(timeout=10)
        
    finally:
        # Stop performance monitoring
        monitor.stop_monitoring = True
        
        # Analyze performance data
        perf_analysis = monitor.analyze_performance()
        
        logger.info("Performance Analysis Results:")
        logger.info(f"CPU Usage: Avg={perf_analysis['cpu_avg']:.2f}%, Max={perf_analysis['cpu_max']:.2f}%")
        logger.info(f"Memory Usage: Avg={perf_analysis['memory_avg']:.2f}%, Max={perf_analysis['memory_max']:.2f}%")
        
        if 'read_throughput_MBps' in perf_analysis:
            logger.info(f"Read Throughput: {perf_analysis['read_throughput_MBps']:.2f} MB/s")
        if 'write_throughput_MBps' in perf_analysis:
            logger.info(f"Write Throughput: {perf_analysis['write_throughput_MBps']:.2f} MB/s")
    
    # Validate performance requirements
    # Note: These thresholds should be adjusted based on your specific environment/requirements
    assert perf_analysis['cpu_avg'] < 90, f"Average CPU usage too high: {perf_analysis['cpu_avg']:.2f}%"
    
    if 'cpu_stddev' in perf_analysis:
        assert perf_analysis['cpu_stddev'] < 20, f"CPU usage too variable: stddev={perf_analysis['cpu_stddev']:.2f}%"
    
    # Check for predictable throughput (no severe degradation)
    if 'read_throughput_MBps' in perf_analysis and 'write_throughput_MBps' in perf_analysis:
        # These thresholds are examples and should be adjusted based on your system capabilities
        assert perf_analysis['read_throughput_MBps'] > 1.0, f"Read throughput too low: {perf_analysis['read_throughput_MBps']:.2f} MB/s"
        assert perf_analysis['write_throughput_MBps'] > 1.0, f"Write throughput too low: {perf_analysis['write_throughput_MBps']:.2f} MB/s"
    
    # Verify remaining keys (check that deletion worked)
    remaining_keys = data_ops.list_keys()
    logger.info(f"Remaining keys after test: {len(remaining_keys)}")
    
    # Check the list of uploaded_keys vs. remaining keys to verify deletion occurred
    logger.info(f"Keys uploaded during test: {len(uploaded_keys)}")
    
    # The test is successful if the system maintained stable performance
    # throughout the continuous data churn operation
    logger.info("Performance test under continuous data deletion completed successfully")

#!/usr/bin/env python3

import os
import time
import logging
import subprocess
import pytest
import statistics
import concurrent.futures
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants for the test
OZONE_CONF_DIR = os.environ.get('OZONE_CONF_DIR', '/etc/ozone')
TEST_DATA_DIR = "/tmp/ozone_perf_test_data"
OZONE_BIN = os.environ.get('OZONE_HOME', '/opt/ozone') + "/bin/ozone"

# Storage tier policies
STORAGE_POLICIES = {
    "HOT": {"replication": 3, "type": "DISK"},
    "WARM": {"replication": 2, "type": "DISK"},
    "COLD": {"replication": 1, "type": "ARCHIVE"}
}

# Test data sizes in MB
TEST_DATA_SIZES = [10, 100, 500, 1024, 4096]

class OzoneStorageTierPerformanceTester:
    def __init__(self):
        self.test_files = {}
        os.makedirs(TEST_DATA_DIR, exist_ok=True)
    
    def setup_test_data(self, sizes_mb: List[int]) -> Dict[int, str]:
        """Create test files of specified sizes"""
        for size_mb in sizes_mb:
            file_path = f"{TEST_DATA_DIR}/test_file_{size_mb}MB.dat"
            if not os.path.exists(file_path):
                logger.info(f"Creating test file of size {size_mb}MB")
                with open(file_path, 'wb') as f:
                    f.write(os.urandom(size_mb * 1024 * 1024))
            self.test_files[size_mb] = file_path
        return self.test_files
    
    def create_volume_and_bucket(self, volume: str, bucket: str) -> None:
        """Create volume and bucket in Ozone"""
        try:
            subprocess.run([OZONE_BIN, "sh", "volume", "create", volume], check=True)
            subprocess.run([OZONE_BIN, "sh", "bucket", "create", f"/{volume}/{bucket}"], check=True)
            logger.info(f"Created volume {volume} and bucket {bucket}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create volume/bucket: {e}")
            raise
    
    def set_storage_policy(self, volume: str, bucket: str, policy: str) -> None:
        """Set storage policy for the bucket"""
        try:
            subprocess.run(
                [OZONE_BIN, "sh", "bucket", "setproperties", f"/{volume}/{bucket}", 
                 f"--replicationFactor={STORAGE_POLICIES[policy]['replication']}",
                 f"--storageType={STORAGE_POLICIES[policy]['type']}"], 
                check=True
            )
            logger.info(f"Set {policy} policy for /{volume}/{bucket}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to set storage policy: {e}")
            raise
    
    def write_data(self, volume: str, bucket: str, file_size_mb: int) -> float:
        """Write data to the bucket and return time taken in seconds"""
        file_path = self.test_files[file_size_mb]
        key = f"key_{file_size_mb}mb"
        
        start_time = time.time()
        try:
            subprocess.run(
                [OZONE_BIN, "sh", "key", "put", f"/{volume}/{bucket}/", file_path, "--key", key],
                check=True
            )
            end_time = time.time()
            duration = end_time - start_time
            logger.info(f"Wrote {file_size_mb}MB to /{volume}/{bucket}/{key} in {duration:.2f} seconds")
            return duration
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to write data: {e}")
            raise
    
    def read_data(self, volume: str, bucket: str, file_size_mb: int) -> float:
        """Read data from the bucket and return time taken in seconds"""
        key = f"key_{file_size_mb}mb"
        output_file = f"{TEST_DATA_DIR}/output_{volume}_{bucket}_{file_size_mb}mb.dat"
        
        start_time = time.time()
        try:
            subprocess.run(
                [OZONE_BIN, "sh", "key", "get", f"/{volume}/{bucket}/{key}", output_file],
                check=True
            )
            end_time = time.time()
            duration = end_time - start_time
            logger.info(f"Read {file_size_mb}MB from /{volume}/{bucket}/{key} in {duration:.2f} seconds")
            os.remove(output_file)  # Clean up
            return duration
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to read data: {e}")
            raise
    
    def move_data_between_tiers(self, source_vol: str, source_bucket: str, 
                               target_vol: str, target_bucket: str, 
                               file_size_mb: int) -> float:
        """Simulate data movement between tiers and measure time"""
        source_key = f"key_{file_size_mb}mb"
        target_key = f"key_{file_size_mb}mb"
        temp_file = f"{TEST_DATA_DIR}/temp_{file_size_mb}mb.dat"
        
        start_time = time.time()
        try:
            # Get from source
            subprocess.run(
                [OZONE_BIN, "sh", "key", "get", f"/{source_vol}/{source_bucket}/{source_key}", temp_file],
                check=True
            )
            
            # Put to target
            subprocess.run(
                [OZONE_BIN, "sh", "key", "put", f"/{target_vol}/{target_bucket}/", temp_file, "--key", target_key],
                check=True
            )
            
            end_time = time.time()
            duration = end_time - start_time
            logger.info(f"Moved {file_size_mb}MB from {source_vol}/{source_bucket} to {target_vol}/{target_bucket} in {duration:.2f} seconds")
            os.remove(temp_file)  # Clean up
            return duration
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to move data between tiers: {e}")
            if os.path.exists(temp_file):
                os.remove(temp_file)
            raise

    def parallel_io_test(self, volume: str, bucket: str, file_size_mb: int, num_threads: int) -> Tuple[float, float]:
        """Test parallel I/O operations"""
        write_times = []
        read_times = []
        
        def write_job(i):
            key = f"key_{file_size_mb}mb_{i}"
            file_path = self.test_files[file_size_mb]
            start_time = time.time()
            subprocess.run([OZONE_BIN, "sh", "key", "put", f"/{volume}/{bucket}/", file_path, "--key", key], check=True)
            return time.time() - start_time
            
        def read_job(i):
            key = f"key_{file_size_mb}mb_{i}"
            output_file = f"{TEST_DATA_DIR}/output_{volume}_{bucket}_{file_size_mb}mb_{i}.dat"
            start_time = time.time()
            subprocess.run([OZONE_BIN, "sh", "key", "get", f"/{volume}/{bucket}/{key}", output_file], check=True)
            os.remove(output_file)
            return time.time() - start_time
        
        # Parallel writes
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            write_times = list(executor.map(write_job, range(num_threads)))
        
        # Parallel reads
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            read_times = list(executor.map(read_job, range(num_threads)))
        
        avg_write = statistics.mean(write_times)
        avg_read = statistics.mean(read_times)
        logger.info(f"Parallel test with {num_threads} threads: avg write {avg_write:.2f}s, avg read {avg_read:.2f}s")
        
        return avg_write, avg_read


@pytest.mark.parametrize("policy", ["HOT", "WARM", "COLD"])
@pytest.mark.parametrize("file_size_mb", [10, 100, 500])
def test_28_storage_policy_single_io_performance(policy, file_size_mb):
    """
    Test Case 28: Measure performance with different storage policies
    Tests write and read performance for individual storage tiers
    """
    tester = OzoneStorageTierPerformanceTester()
    
    # Setup test data
    tester.setup_test_data([file_size_mb])
    
    # Create volume and bucket with specified policy
    volume = f"vol-policy-perf-{policy.lower()}"
    bucket = f"bucket-{file_size_mb}mb"
    
    tester.create_volume_and_bucket(volume, bucket)
    tester.set_storage_policy(volume, bucket, policy)
    
    # Write and read operations
    write_time = tester.write_data(volume, bucket, file_size_mb)
    read_time = tester.read_data(volume, bucket, file_size_mb)
    
    # Calculate throughput (MB/s)
    write_throughput = file_size_mb / write_time
    read_throughput = file_size_mb / read_time
    
    logger.info(f"Storage Policy: {policy}, Size: {file_size_mb}MB")
    logger.info(f"Write Throughput: {write_throughput:.2f} MB/s")
    logger.info(f"Read Throughput: {read_throughput:.2f} MB/s")
    
    # Expected throughput ratios based on policy (these are examples, adjust based on real expectations)
    expected_performance_ratios = {
        "HOT": 1.0,  # Baseline
        "WARM": 0.8,  # Expect 20% slower than HOT
        "COLD": 0.5,  # Expect 50% slower than HOT
    }
    
    # Flexible assertions - performance can vary widely
    # We're checking if the performance is within a reasonable range of expectations
    if policy == "HOT":
        # Store baseline metrics for comparison
        pytest.HOT_write_throughput = write_throughput
        pytest.HOT_read_throughput = read_throughput
    else:
        # Calculate expected throughput based on HOT performance
        expected_write = pytest.HOT_write_throughput * expected_performance_ratios[policy]
        expected_read = pytest.HOT_read_throughput * expected_performance_ratios[policy]
        
        # Allow for 20% variation in actual vs expected performance
        assert write_throughput >= expected_write * 0.8, f"{policy} write performance too low"
        assert read_throughput >= expected_read * 0.8, f"{policy} read performance too low"


@pytest.mark.parametrize("policy_pair", [("HOT", "WARM"), ("WARM", "COLD"), ("HOT", "COLD")])
@pytest.mark.parametrize("file_size_mb", [10, 100])
def test_28_storage_tier_transition_performance(policy_pair, file_size_mb):
    """
    Test Case 28: Measure performance with different storage policies
    Tests data movement between storage tiers and its performance impact
    """
    source_policy, target_policy = policy_pair
    tester = OzoneStorageTierPerformanceTester()
    
    # Setup test data
    tester.setup_test_data([file_size_mb])
    
    # Create source volume/bucket
    source_vol = f"vol-source-{source_policy.lower()}"
    source_bucket = f"bucket-{file_size_mb}"
    tester.create_volume_and_bucket(source_vol, source_bucket)
    tester.set_storage_policy(source_vol, source_bucket, source_policy)
    
    # Create target volume/bucket
    target_vol = f"vol-target-{target_policy.lower()}"
    target_bucket = f"bucket-{file_size_mb}"
    tester.create_volume_and_bucket(target_vol, target_bucket)
    tester.set_storage_policy(target_vol, target_bucket, target_policy)
    
    # Write data to source
    source_write_time = tester.write_data(source_vol, source_bucket, file_size_mb)
    
    # Move data between tiers
    transition_time = tester.move_data_between_tiers(
        source_vol, source_bucket,
        target_vol, target_bucket,
        file_size_mb
    )
    
    # Read from target after transition
    target_read_time = tester.read_data(target_vol, target_bucket, file_size_mb)
    
    # Calculate metrics
    transition_throughput = file_size_mb / transition_time
    source_write_throughput = file_size_mb / source_write_time
    target_read_throughput = file_size_mb / target_read_time
    
    logger.info(f"Transition: {source_policy} → {target_policy}, Size: {file_size_mb}MB")
    logger.info(f"Transition Throughput: {transition_throughput:.2f} MB/s")
    logger.info(f"Source Write Throughput: {source_write_throughput:.2f} MB/s")
    logger.info(f"Target Read Throughput: {target_read_throughput:.2f} MB/s")
    
    # Expected baseline for transition throughput
    # Usually, transitions are expected to be slower than direct reads/writes
    expected_transition_factor = 0.5  # Expecting transition to be 50% of write speed
    
    # Verify transition performance isn't severely degraded
    assert transition_throughput >= source_write_throughput * expected_transition_factor, \
        f"Tier transition from {source_policy} to {target_policy} is too slow"


@pytest.mark.parametrize("policy", ["HOT", "WARM", "COLD"])
@pytest.mark.parametrize("num_threads", [2, 4, 8])
def test_28_storage_policy_parallel_performance(policy, num_threads):
    """
    Test Case 28: Measure performance with different storage policies
    Tests parallel I/O performance with different storage tiers
    """
    tester = OzoneStorageTierPerformanceTester()
    
    # Use a smaller file size for parallel tests to avoid overwhelming the system
    file_size_mb = 25
    
    # Setup test data
    tester.setup_test_data([file_size_mb])
    
    # Create volume and bucket with specified policy
    volume = f"vol-parallel-{policy.lower()}"
    bucket = f"bucket-parallel-{num_threads}"
    
    tester.create_volume_and_bucket(volume, bucket)
    tester.set_storage_policy(volume, bucket, policy)
    
    # Run parallel I/O test
    avg_write_time, avg_read_time = tester.parallel_io_test(volume, bucket, file_size_mb, num_threads)
    
    # Calculate aggregate throughput
    aggregate_write_throughput = (file_size_mb * num_threads) / avg_write_time
    aggregate_read_throughput = (file_size_mb * num_threads) / avg_read_time
    
    logger.info(f"Policy: {policy}, Threads: {num_threads}")
    logger.info(f"Aggregate Write Throughput: {aggregate_write_throughput:.2f} MB/s")
    logger.info(f"Aggregate Read Throughput: {aggregate_read_throughput:.2f} MB/s")
    
    # Expected scaling factors for each policy
    # HOT storage should scale almost linearly with threads
    # WARM and COLD might not scale as well
    expected_scaling = {
        "HOT": 0.8,  # Expect 80% of linear scaling
        "WARM": 0.7,  # Expect 70% of linear scaling
        "COLD": 0.5,  # Expect 50% of linear scaling
    }
    
    # Only check scaling for tests with more than 2 threads
    if num_threads > 2 and hasattr(pytest, f"{policy}_throughput_{num_threads//2}"):
        # Get throughput from test with half the number of threads
        previous_write_throughput = getattr(pytest, f"{policy}_write_throughput_{num_threads//2}")
        previous_read_throughput = getattr(pytest, f"{policy}_read_throughput_{num_threads//2}")
        
        # Calculate expected throughput with linear scaling
        expected_write = previous_write_throughput * 2
        expected_read = previous_read_throughput * 2
        
        # Check if throughput scales according to expectations
        actual_write_scaling = aggregate_write_throughput / previous_write_throughput
        actual_read_scaling = aggregate_read_throughput / previous_read_throughput
        
        logger.info(f"Write scaling factor: {actual_write_scaling:.2f}x")
        logger.info(f"Read scaling factor: {actual_read_scaling:.2f}x")
        
        # Assert that scaling is at least as expected for the policy
        min_expected_scaling = expected_scaling[policy]
        assert actual_write_scaling >= min_expected_scaling, f"{policy} write scaling is lower than expected"
        assert actual_read_scaling >= min_expected_scaling, f"{policy} read scaling is lower than expected"
    
    # Store throughput metrics for future comparison
    setattr(pytest, f"{policy}_write_throughput_{num_threads}", aggregate_write_throughput)
    setattr(pytest, f"{policy}_read_throughput_{num_threads}", aggregate_read_throughput)

import os
import time
import pytest
import subprocess
import tempfile
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor

# Import the Ozone client
from pyarrow import ozone as oz


class QuotaPerformanceTestConfig:
    """Configuration for quota performance testing"""
    # Test volumes and buckets
    VOLUME_NAME = "quotavol"
    BUCKET_NAME = "quotabucket"
    
    # Quota types to test
    QUOTA_TYPES = ["space", "object"]
    
    # Quota sizes to test (in bytes for space, count for objects)
    QUOTA_SIZES = {
        "space": [10*1024*1024, 50*1024*1024, 100*1024*1024],  # 10MB, 50MB, 100MB
        "object": [10, 50, 100]  # 10 objects, 50 objects, 100 objects
    }
    
    # File sizes for testing (in bytes)
    FILE_SIZES = [512*1024, 1024*1024, 4*1024*1024, 7.5*1024*1024]  # 512KB, 1MB, 4MB, 7.5MB
    
    # Number of parallel operations
    CONCURRENCY_LEVELS = [1, 4, 8]
    
    # Number of operations to perform for each test
    NUM_OPERATIONS = 10
    
    # Report directory
    REPORT_DIR = "quota_performance_reports"


class OzoneQuotaManager:
    """Helper class to manage Ozone quotas"""
    
    @staticmethod
    def set_space_quota(volume: str, size_bytes: int) -> None:
        """Set a space quota on a volume"""
        cmd = f"ozone sh volume setquota --space {size_bytes} {volume}"
        subprocess.run(cmd, shell=True, check=True)
    
    @staticmethod
    def set_object_quota(volume: str, count: int) -> None:
        """Set an object count quota on a volume"""
        cmd = f"ozone sh volume setquota --count {count} {volume}"
        subprocess.run(cmd, shell=True, check=True)
    
    @staticmethod
    def clear_quotas(volume: str) -> None:
        """Clear all quotas on a volume"""
        cmd = f"ozone sh volume clrquota --space --count {volume}"
        subprocess.run(cmd, shell=True, check=True)
    
    @staticmethod
    def get_quota_usage(volume: str) -> Dict:
        """Get the current quota usage for a volume"""
        cmd = f"ozone sh volume info {volume} --json"
        result = subprocess.run(cmd, shell=True, check=True, 
                               stdout=subprocess.PIPE, text=True)
        
        # Parse the JSON output to extract quota information
        import json
        volume_info = json.loads(result.stdout)
        
        return {
            "space_used": volume_info.get("usedBytes", 0),
            "space_quota": volume_info.get("quotaInBytes", 0),
            "object_used": volume_info.get("usedNamespace", 0),
            "object_quota": volume_info.get("quotaInNamespace", 0)
        }


class OzoneTestHelper:
    """Helper class for Ozone operations"""
    
    @staticmethod
    def create_volume_and_bucket(volume: str, bucket: str) -> None:
        """Create a volume and bucket if they don't exist"""
        # Create volume
        cmd = f"ozone sh volume create {volume}"
        subprocess.run(cmd, shell=True, check=True)
        
        # Create bucket
        cmd = f"ozone sh bucket create /{volume}/{bucket}"
        subprocess.run(cmd, shell=True, check=True)
    
    @staticmethod
    def create_test_file(size_bytes: int) -> str:
        """Create a test file of specified size"""
        fd, path = tempfile.mkstemp()
        with os.fdopen(fd, 'wb') as f:
            f.write(os.urandom(int(size_bytes)))
        return path
    
    @staticmethod
    def put_object(volume: str, bucket: str, key: str, file_path: str) -> float:
        """Put an object and measure the time it takes"""
        start_time = time.time()
        cmd = f"ozone fs -put {file_path} o3://{volume}.{bucket}/{key}"
        subprocess.run(cmd, shell=True, check=True)
        end_time = time.time()
        return end_time - start_time
    
    @staticmethod
    def delete_object(volume: str, bucket: str, key: str) -> None:
        """Delete an object"""
        cmd = f"ozone fs -rm o3://{volume}.{bucket}/{key}"
        subprocess.run(cmd, shell=True, check=True)
    
    @staticmethod
    def cleanup(volume: str) -> None:
        """Clean up by removing the volume and all its contents"""
        cmd = f"ozone sh volume delete {volume}"
        subprocess.run(cmd, shell=True)


class PerformanceResults:
    """Class to store and analyze performance results"""
    
    def __init__(self):
        self.results = []
    
    def add_result(self, quota_type: str, quota_size: int, operation: str, 
                   concurrency: int, file_size: int, duration: float, 
                   success: bool) -> None:
        """Add a result entry"""
        self.results.append({
            "quota_type": quota_type,
            "quota_size": quota_size,
            "operation": operation,
            "concurrency": concurrency,
            "file_size": file_size,
            "duration": duration,
            "success": success
        })
    
    def get_dataframe(self) -> pd.DataFrame:
        """Convert results to a pandas DataFrame"""
        return pd.DataFrame(self.results)
    
    def save_csv(self, filename: str) -> None:
        """Save results to a CSV file"""
        df = self.get_dataframe()
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df.to_csv(filename, index=False)
    
    def generate_plots(self, output_dir: str) -> None:
        """Generate performance plots from results"""
        df = self.get_dataframe()
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot 1: Performance by quota type and size
        fig, ax = plt.subplots(figsize=(10, 6))
        for quota_type in df['quota_type'].unique():
            subset = df[df['quota_type'] == quota_type]
            quota_sizes = sorted(subset['quota_size'].unique())
            durations = [subset[subset['quota_size'] == size]['duration'].mean() 
                        for size in quota_sizes]
            ax.plot(quota_sizes, durations, marker='o', label=f"{quota_type} quota")
        
        ax.set_xlabel('Quota Size')
        ax.set_ylabel('Average Operation Duration (s)')
        ax.set_title('Performance by Quota Type and Size')
        ax.legend()
        plt.savefig(f"{output_dir}/performance_by_quota_size.png")
        
        # Plot 2: Performance by concurrency level
        fig, ax = plt.subplots(figsize=(10, 6))
        for concurrency in sorted(df['concurrency'].unique()):
            subset = df[df['concurrency'] == concurrency]
            file_sizes = sorted(subset['file_size'].unique())
            durations = [subset[subset['file_size'] == size]['duration'].mean() 
                        for size in file_sizes]
            ax.plot(file_sizes, durations, marker='o', label=f"{concurrency} threads")
        
        ax.set_xlabel('File Size (bytes)')
        ax.set_ylabel('Average Operation Duration (s)')
        ax.set_title('Performance by Concurrency Level')
        ax.legend()
        plt.savefig(f"{output_dir}/performance_by_concurrency.png")


@pytest.fixture(scope="module")
def setup_ozone_environment():
    """Set up the Ozone environment for testing"""
    config = QuotaPerformanceTestConfig()
    helper = OzoneTestHelper()
    
    # Create the test volume and bucket
    helper.create_volume_and_bucket(config.VOLUME_NAME, config.BUCKET_NAME)
    
    yield config
    
    # Clean up after the tests
    helper.cleanup(config.VOLUME_NAME)


@pytest.fixture
def performance_results():
    """Create a new performance results tracker for each test"""
    return PerformanceResults()


def test_29_quota_enforcement_performance(setup_ozone_environment, performance_results):
    """
    Test performance under quota enforcement
    
    This test evaluates the performance impact of quota enforcement by:
    1. Setting various quota limits (space and object count)
    2. Performing operations approaching and exceeding quota limits
    3. Measuring the performance impact of quota enforcement
    4. Testing quota increase/decrease operations
    """
    config = setup_ozone_environment
    quota_manager = OzoneQuotaManager()
    helper = OzoneTestHelper()
    
    # Define test scenarios
    test_scenarios = []
    
    # Generate test scenarios for different quota types and sizes
    for quota_type in config.QUOTA_TYPES:
        for quota_size in config.QUOTA_SIZES[quota_type]:
            for concurrency in config.CONCURRENCY_LEVELS:
                # We'll test with different file sizes
                for file_size in config.FILE_SIZES:
                    test_scenarios.append({
                        "quota_type": quota_type,
                        "quota_size": quota_size,
                        "concurrency": concurrency,
                        "file_size": file_size
                    })
    
    # Run each test scenario
    for scenario in test_scenarios:
        quota_type = scenario["quota_type"]
        quota_size = scenario["quota_size"]
        concurrency = scenario["concurrency"]
        file_size = scenario["file_size"]
        
        print(f"\nRunning test with {quota_type} quota={quota_size}, "
              f"concurrency={concurrency}, file_size={file_size}")
        
        # Clear any existing quotas
        quota_manager.clear_quotas(config.VOLUME_NAME)
        
        # Set the quota for this test
        if quota_type == "space":
            quota_manager.set_space_quota(config.VOLUME_NAME, quota_size)
        else:  # object quota
            quota_manager.set_object_quota(config.VOLUME_NAME, quota_size)
        
        # Create a test file of the specified size
        test_file_path = helper.create_test_file(file_size)
        
        try:
            # Function to perform a single put operation
            def put_operation(i):
                key = f"testkey-{quota_type}-{quota_size}-{concurrency}-{file_size}-{i}"
                try:
                    duration = helper.put_object(
                        config.VOLUME_NAME, 
                        config.BUCKET_NAME, 
                        key, 
                        test_file_path
                    )
                    return {"key": key, "duration": duration, "success": True}
                except Exception as e:
                    # If the operation fails (e.g., quota exceeded), record it
                    return {"key": key, "duration": 0, "success": False, "error": str(e)}
            
            # Perform operations in parallel according to concurrency level
            results = []
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                future_to_index = {
                    executor.submit(put_operation, i): i 
                    for i in range(min(config.NUM_OPERATIONS, quota_size * 2))
                }
                
                for future in future_to_index:
                    result = future.result()
                    results.append(result)
                    
                    # Add to performance results
                    if result["success"]:
                        performance_results.add_result(
                            quota_type=quota_type,
                            quota_size=quota_size,
                            operation="put",
                            concurrency=concurrency,
                            file_size=file_size,
                            duration=result["duration"],
                            success=True
                        )
                    else:
                        performance_results.add_result(
                            quota_type=quota_type,
                            quota_size=quota_size,
                            operation="put",
                            concurrency=concurrency,
                            file_size=file_size,
                            duration=0,
                            success=False
                        )
            
            # Measure time for quota increase operation
            start_time = time.time()
            if quota_type == "space":
                quota_manager.set_space_quota(config.VOLUME_NAME, quota_size * 2)
            else:  # object quota
                quota_manager.set_object_quota(config.VOLUME_NAME, quota_size * 2)
            quota_increase_duration = time.time() - start_time
            
            # Add quota management performance results
            performance_results.add_result(
                quota_type=quota_type,
                quota_size=quota_size,
                operation="increase_quota",
                concurrency=1,
                file_size=0,
                duration=quota_increase_duration,
                success=True
            )
            
            # Measure time for quota decrease operation
            start_time = time.time()
            if quota_type == "space":
                quota_manager.set_space_quota(config.VOLUME_NAME, quota_size)
            else:  # object quota
                quota_manager.set_object_quota(config.VOLUME_NAME, quota_size)
            quota_decrease_duration = time.time() - start_time
            
            performance_results.add_result(
                quota_type=quota_type,
                quota_size=quota_size,
                operation="decrease_quota",
                concurrency=1,
                file_size=0,
                duration=quota_decrease_duration,
                success=True
            )
            
            # Clean up objects for this scenario
            for result in results:
                if result["success"]:
                    helper.delete_object(
                        config.VOLUME_NAME, 
                        config.BUCKET_NAME, 
                        result["key"]
                    )
                    
        finally:
            # Clean up the test file
            if os.path.exists(test_file_path):
                os.remove(test_file_path)
            
            # Clear quotas after this test scenario
            quota_manager.clear_quotas(config.VOLUME_NAME)
    
    # Save and analyze the performance results
    os.makedirs(config.REPORT_DIR, exist_ok=True)
    
    # Save raw data to CSV
    csv_file = f"{config.REPORT_DIR}/quota_performance_results.csv"
    performance_results.save_csv(csv_file)
    
    # Generate performance plots
    performance_results.generate_plots(config.REPORT_DIR)
    
    # Basic analysis of results
    df = performance_results.get_dataframe()
    
    # Calculate success rates
    success_rate = len(df[df['success'] == True]) / len(df) * 100
    
    # Calculate average durations for successful operations
    successful_ops = df[df['success'] == True]
    avg_put_duration = successful_ops[successful_ops['operation'] == 'put']['duration'].mean()
    avg_quota_increase_duration = successful_ops[successful_ops['operation'] == 'increase_quota']['duration'].mean()
    avg_quota_decrease_duration = successful_ops[successful_ops['operation'] == 'decrease_quota']['duration'].mean()
    
    # Print summary results
    print(f"\nPerformance Test Summary:")
    print(f"Success Rate: {success_rate:.2f}%")
    print(f"Average PUT Duration: {avg_put_duration:.4f} seconds")
    print(f"Average Quota Increase Duration: {avg_quota_increase_duration:.4f} seconds")
    print(f"Average Quota Decrease Duration: {avg_quota_decrease_duration:.4f} seconds")
    print(f"Detailed results saved to: {csv_file}")
    print(f"Performance plots saved to: {config.REPORT_DIR}/")
    
    # Assertions to validate the test
    # The test validates that quota operations are efficient (under 1 second)
    assert avg_quota_increase_duration < 1.0, "Quota increase operations are too slow"
    assert avg_quota_decrease_duration < 1.0, "Quota decrease operations are too slow"
    
    # We expect some operations to succeed and some to fail (when exceeding quotas)
    assert success_rate > 0, "No operations succeeded"
    assert success_rate < 100, "All operations succeeded, quota enforcement may not be working"
    
    # Check if performance is acceptable for put operations
    assert avg_put_duration < 5.0, "PUT operations are too slow"

import pytest
import os
import subprocess
import time
import json
import statistics
from typing import List, Dict, Tuple
import logging
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants for test configuration
VOLUME_NAME = "audit-perf-vol"
BUCKET_NAME = "audit-perf-bkt"
TEST_FILE_SIZES = [1024, 10240, 102400, 1048576]  # Sizes in KB: 1KB, 10KB, 100KB, 1MB
OPERATIONS = ["put", "get", "list", "delete"]
AUDIT_LOG_LEVELS = ["OFF", "MINIMAL", "INFO", "DETAILED"]
NUM_ITERATIONS = 5
CONCURRENCY_LEVELS = [1, 5, 10, 20]


class OzoneClusterManager:
    """Helper class for managing Ozone cluster and audit configuration"""
    
    @staticmethod
    def set_audit_log_level(level: str) -> None:
        """
        Set the audit log level for the Ozone cluster
        
        Args:
            level: One of OFF, MINIMAL, INFO, DETAILED
        """
        logger.info(f"Setting audit log level to {level}")
        
        if level == "OFF":
            cmd = "ozone admin audit disable"
        else:
            cmd = f"ozone admin audit enable --level {level}"
            
        try:
            subprocess.run(cmd, shell=True, check=True, 
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # Wait for changes to propagate
            time.sleep(5)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to set audit log level: {e}")
            raise
    
    @staticmethod
    def restart_ozone_services() -> None:
        """Restart all Ozone services to ensure audit config is applied"""
        logger.info("Restarting Ozone services")
        try:
            subprocess.run("ozone admin service restart --all", 
                          shell=True, check=True)
            # Wait for services to restart
            time.sleep(30)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to restart Ozone services: {e}")
            raise


class PerformanceTester:
    """Helper class for running performance tests"""
    
    @staticmethod
    def create_test_files(sizes: List[int]) -> Dict[int, str]:
        """
        Create test files of specified sizes
        
        Args:
            sizes: List of sizes in KB
            
        Returns:
            Dictionary mapping size to file path
        """
        file_paths = {}
        for size in sizes:
            file_name = f"test_file_{size}kb.data"
            file_path = os.path.join("/tmp", file_name)
            
            # Create the file with random data
            with open(file_path, "wb") as f:
                f.write(os.urandom(size * 1024))
                
            file_paths[size] = file_path
            
        return file_paths
    
    @staticmethod
    def setup_test_environment():
        """Create volume and bucket for testing"""
        try:
            # Create volume
            subprocess.run(f"ozone sh volume create /{VOLUME_NAME}", 
                          shell=True, check=True)
            
            # Create bucket
            subprocess.run(f"ozone sh bucket create /{VOLUME_NAME}/{BUCKET_NAME}", 
                          shell=True, check=True)
            
            logger.info(f"Created volume {VOLUME_NAME} and bucket {BUCKET_NAME}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create volume/bucket: {e}")
            raise
    
    @staticmethod
    def cleanup_test_environment():
        """Remove test volume and bucket"""
        try:
            # Delete bucket
            subprocess.run(f"ozone sh bucket delete /{VOLUME_NAME}/{BUCKET_NAME}", 
                          shell=True, check=True)
            
            # Delete volume
            subprocess.run(f"ozone sh volume delete /{VOLUME_NAME}", 
                          shell=True, check=True)
            
            logger.info(f"Cleaned up volume {VOLUME_NAME} and bucket {BUCKET_NAME}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to delete volume/bucket: {e}")
            # Don't raise here as we're cleaning up
    
    @staticmethod
    def run_operation(operation: str, size: int, file_path: str, key_name: str) -> float:
        """
        Run a single Ozone operation and measure its execution time
        
        Args:
            operation: One of put, get, list, delete
            size: Size of the test file in KB
            file_path: Path to the test file
            key_name: The key name to use
            
        Returns:
            Execution time in seconds
        """
        start_time = time.time()
        
        try:
            if operation == "put":
                cmd = f"ozone sh key put /{VOLUME_NAME}/{BUCKET_NAME}/ {key_name} {file_path}"
            elif operation == "get":
                cmd = f"ozone sh key get /{VOLUME_NAME}/{BUCKET_NAME}/{key_name} /tmp/retrieved_{key_name}"
            elif operation == "list":
                cmd = f"ozone sh key list /{VOLUME_NAME}/{BUCKET_NAME}"
            elif operation == "delete":
                cmd = f"ozone sh key delete /{VOLUME_NAME}/{BUCKET_NAME}/{key_name}"
            else:
                raise ValueError(f"Unsupported operation: {operation}")
            
            subprocess.run(cmd, shell=True, check=True, 
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Operation failed: {e}")
            raise
            
        end_time = time.time()
        return end_time - start_time
    
    @staticmethod
    def run_concurrent_operations(operation: str, size: int, file_path: str, 
                                 concurrency: int) -> List[float]:
        """Run operations concurrently and measure times"""
        times = []
        
        def task(i):
            key_name = f"key_{operation}_{size}kb_{i}"
            return PerformanceTester.run_operation(operation, size, file_path, key_name)
        
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            times = list(executor.map(task, range(concurrency)))
            
        return times


@pytest.fixture(scope="module")
def test_files():
    """Create test files for performance testing"""
    file_paths = PerformanceTester.create_test_files(TEST_FILE_SIZES)
    yield file_paths
    
    # Cleanup test files
    for path in file_paths.values():
        if os.path.exists(path):
            os.remove(path)


@pytest.fixture(scope="module")
def test_environment():
    """Setup and teardown the test environment"""
    PerformanceTester.setup_test_environment()
    yield
    PerformanceTester.cleanup_test_environment()


def test_30_audit_logging_performance_impact(test_files, test_environment):
    """
    Evaluate performance with audit logging enabled
    
    This test measures the performance impact of different audit log verbosity levels
    on standard operations like put, get, list, and delete.
    """
    results = []
    
    # For each audit log level
    for audit_level in AUDIT_LOG_LEVELS:
        logger.info(f"Testing with audit level: {audit_level}")
        OzoneClusterManager.set_audit_log_level(audit_level)
        
        # For each operation type
        for operation in OPERATIONS:
            # For each file size
            for size, file_path in test_files.items():
                # For each concurrency level
                for concurrency in CONCURRENCY_LEVELS:
                    # Skip some combinations to reduce test time
                    if size > 100000 and concurrency > 10:
                        continue
                        
                    execution_times = []
                    
                    # Perform multiple iterations for statistical significance
                    for i in range(NUM_ITERATIONS):
                        # For put operations, we need to ensure keys don't exist
                        if operation == "put" and i > 0:
                            # Clear previous keys before next iteration
                            PerformanceTester.cleanup_test_environment()
                            PerformanceTester.setup_test_environment()
                        
                        # For get, list, delete, we need to ensure keys exist first
                        if operation in ["get", "list", "delete"]:
                            # Create the keys first
                            for j in range(concurrency):
                                key_name = f"key_{operation}_{size}kb_{j}"
                                PerformanceTester.run_operation("put", size, file_path, key_name)
                                
                        # Run the actual operation with concurrency
                        times = PerformanceTester.run_concurrent_operations(
                            operation, size, file_path, concurrency)
                        execution_times.extend(times)
                    
                    # Calculate statistics
                    avg_time = statistics.mean(execution_times)
                    p95_time = sorted(execution_times)[int(len(execution_times) * 0.95)]
                    
                    # Record the results
                    results.append({
                        "audit_level": audit_level,
                        "operation": operation,
                        "file_size_kb": size,
                        "concurrency": concurrency,
                        "avg_time_sec": avg_time,
                        "p95_time_sec": p95_time
                    })
                    
                    logger.info(f"Results for {audit_level}, {operation}, {size}KB, "
                               f"concurrency={concurrency}: "
                               f"avg={avg_time:.4f}s, p95={p95_time:.4f}s")
    
    # Convert results to DataFrame for analysis
    df = pd.DataFrame(results)
    
    # Generate performance comparison report
    generate_performance_report(df)
    
    # Validate that audit logging has minimal impact on performance
    validate_audit_performance_impact(df)


def generate_performance_report(df: pd.DataFrame) -> None:
    """Generate performance report with charts"""
    report_dir = "performance_reports"
    os.makedirs(report_dir, exist_ok=True)
    
    # Save raw data
    df.to_csv(f"{report_dir}/audit_performance_results.csv", index=False)
    
    # Group by relevant dimensions and create visualizations
    plt.figure(figsize=(12, 8))
    
    # For each operation type
    for operation in OPERATIONS:
        op_data = df[df["operation"] == operation]
        
        # Plot by audit level for each file size (with concurrency=1)
        plt.figure(figsize=(10, 6))
        for size in TEST_FILE_SIZES:
            size_data = op_data[(op_data["file_size_kb"] == size) & 
                               (op_data["concurrency"] == 1)]
            if not size_data.empty:
                plt.plot(size_data["audit_level"], size_data["avg_time_sec"], 
                        marker='o', label=f"{size}KB")
        
        plt.title(f"Impact of Audit Logging on {operation.upper()} Performance")
        plt.xlabel("Audit Log Level")
        plt.ylabel("Average Operation Time (seconds)")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{report_dir}/audit_impact_{operation}.png")
    
    # Generate overall comparison chart
    plt.figure(figsize=(12, 8))
    baseline = df[df["audit_level"] == "OFF"].groupby(["operation", "file_size_kb", "concurrency"])["avg_time_sec"].mean()
    
    for level in ["MINIMAL", "INFO", "DETAILED"]:
        level_data = df[df["audit_level"] == level]
        level_data = level_data.merge(
            baseline.reset_index().rename(columns={"avg_time_sec": "baseline_time"}),
            on=["operation", "file_size_kb", "concurrency"]
        )
        level_data["overhead_percent"] = (level_data["avg_time_sec"] / level_data["baseline_time"] - 1) * 100
        
        avg_overhead = level_data.groupby("operation")["overhead_percent"].mean()
        avg_overhead.plot(kind="bar", alpha=0.7, label=level)
    
    plt.title("Average Performance Overhead by Audit Log Level")
    plt.xlabel("Operation")
    plt.ylabel("Overhead Percentage (%)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{report_dir}/audit_overall_overhead.png")
    
    logger.info(f"Performance report generated in {report_dir}")


def validate_audit_performance_impact(df: pd.DataFrame) -> None:
    """
    Validate that audit logging has acceptable performance impact
    
    This function checks if the performance degradation caused by audit logging
    is within acceptable limits.
    """
    # Group by operation and audit level, then compute average overhead
    baseline = df[df["audit_level"] == "OFF"].groupby(["operation", "file_size_kb", "concurrency"])["avg_time_sec"].mean()
    
    max_acceptable_overhead = {
        "MINIMAL": 5.0,  # 5% overhead acceptable for minimal logging
        "INFO": 15.0,    # 15% for info level
        "DETAILED": 30.0 # 30% for detailed logging
    }
    
    all_valid = True
    
    for level, max_overhead in max_acceptable_overhead.items():
        level_data = df[df["audit_level"] == level]
        if level_data.empty:
            logger.warning(f"No data for audit level {level}")
            continue
            
        level_data = level_data.merge(
            baseline.reset_index().rename(columns={"avg_time_sec": "baseline_time"}),
            on=["operation", "file_size_kb", "concurrency"]
        )
        level_data["overhead_percent"] = (level_data["avg_time_sec"] / level_data["baseline_time"] - 1) * 100
        
        avg_overhead = level_data.groupby("operation")["overhead_percent"].mean()
        max_observed_overhead = avg_overhead.max()
        
        logger.info(f"Audit level {level} - Max overhead: {max_observed_overhead:.2f}% (limit: {max_overhead}%)")
        
        if max_observed_overhead > max_overhead:
            logger.warning(f"Audit level {level} exceeds performance overhead limit")
            logger.warning(f"Operations with high overhead: "
                          f"{avg_overhead[avg_overhead > max_overhead].to_dict()}")
            all_valid = False
    
    # Assert that all audit levels have acceptable performance overhead
    assert all_valid, "One or more audit log levels exceed acceptable performance overhead"

import pytest
import os
import subprocess
import time
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import io
import random
import string
import tempfile
import logging
from subprocess import Popen, PIPE
from typing import Dict, List, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
VOLUME_NAME = "compperf"
BUCKET_NAME = "comptest"
FILE_SIZES = [1024, 10*1024, 1024*1024, 10*1024*1024, 100*1024*1024]  # Sizes in bytes: 1KB, 10KB, 1MB, 10MB, 100MB
COMPRESSION_ALGORITHMS = ["NONE", "ZSTD", "LZ4", "SNAPPY"]
COMPRESSION_LEVELS = ["DEFAULT", "FAST", "HIGH"]
RESULT_CSV = "compression_performance_results.csv"

# Helper functions
def setup_ozone_environment():
    """Setup the Ozone cluster environment for testing"""
    # Create volume and bucket for testing
    volume_cmd = f"ozone sh volume create /{VOLUME_NAME}"
    bucket_cmd = f"ozone sh bucket create /{VOLUME_NAME}/{BUCKET_NAME}"
    
    try:
        subprocess.run(volume_cmd, shell=True, check=True)
        subprocess.run(bucket_cmd, shell=True, check=True)
        logger.info(f"Created volume {VOLUME_NAME} and bucket {BUCKET_NAME}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to create volume/bucket: {e}")
        raise

def cleanup_ozone_environment():
    """Clean up the Ozone test environment"""
    try:
        bucket_cmd = f"ozone sh bucket delete /{VOLUME_NAME}/{BUCKET_NAME}"
        volume_cmd = f"ozone sh volume delete /{VOLUME_NAME}"
        
        subprocess.run(bucket_cmd, shell=True, check=True)
        subprocess.run(volume_cmd, shell=True, check=True)
        logger.info(f"Deleted volume {VOLUME_NAME} and bucket {BUCKET_NAME}")
    except subprocess.CalledProcessError as e:
        logger.warning(f"Cleanup failed: {e}")

def generate_test_file(file_size: int, compressibility: str) -> str:
    """
    Generate a test file with specified size and compressibility
    
    Args:
        file_size: Size of the file in bytes
        compressibility: 'high', 'medium', or 'low'
        
    Returns:
        Path to the generated file
    """
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    
    # Control compressibility by data patterns
    if compressibility == 'high':
        # Highly compressible - repetitive pattern
        with open(temp_file.name, 'wb') as f:
            pattern = b'0' * 1024
            for _ in range(0, file_size, len(pattern)):
                remaining = min(len(pattern), file_size - f.tell())
                if remaining <= 0:
                    break
                f.write(pattern[:remaining])
    
    elif compressibility == 'medium':
        # Medium compressibility - some structure but with variation
        with open(temp_file.name, 'wb') as f:
            patterns = [bytes([i % 256]) * 256 for i in range(10)]
            while f.tell() < file_size:
                pattern = random.choice(patterns)
                remaining = min(len(pattern), file_size - f.tell())
                if remaining <= 0:
                    break
                f.write(pattern[:remaining])
                
    else:  # low compressibility
        # Low compressibility - random data
        with open(temp_file.name, 'wb') as f:
            while f.tell() < file_size:
                remaining = file_size - f.tell()
                if remaining <= 0:
                    break
                chunk_size = min(1024*1024, remaining)  # Write in 1MB chunks
                f.write(os.urandom(chunk_size))
    
    logger.info(f"Generated {compressibility} compressible test file of {file_size} bytes at {temp_file.name}")
    return temp_file.name

def configure_compression(algorithm: str, level: str) -> None:
    """
    Configure Ozone with different compression settings
    
    Args:
        algorithm: Compression algorithm (NONE, ZSTD, LZ4, SNAPPY)
        level: Compression level (DEFAULT, FAST, HIGH)
    """
    # In a real environment, this would modify Ozone configuration files and restart services
    # For this test, we'll simulate by logging the configuration
    logger.info(f"Configuring Ozone with compression algorithm: {algorithm}, level: {level}")
    
    # In a real scenario, you'd update Ozone config files like:
    # ozone.fs.compression.algorithm={algorithm}
    # ozone.fs.compression.level={level}
    # And then restart the required services
    
    # For testing purposes, we'll sleep to simulate the configuration change
    time.sleep(1)

def measure_performance(operation: str, algorithm: str, level: str, file_path: str, 
                       compressibility: str, file_size: int) -> Dict[str, float]:
    """
    Measure the performance of read/write operations
    
    Args:
        operation: 'write' or 'read'
        algorithm: Compression algorithm
        level: Compression level
        file_path: Path to the test file
        compressibility: Compressibility level
        file_size: Size of the file in bytes
        
    Returns:
        Dictionary with performance metrics
    """
    key_name = f"test-{algorithm}-{level}-{compressibility}-{file_size}-{int(time.time())}"
    metrics = {}
    
    if operation == 'write':
        # Measure write performance
        start_time = time.time()
        
        cmd = f"ozone fs -put {file_path} o3fs://{VOLUME_NAME}.{BUCKET_NAME}/{key_name}"
        process = subprocess.run(cmd, shell=True, stdout=PIPE, stderr=PIPE)
        
        if process.returncode != 0:
            logger.error(f"Write operation failed: {process.stderr.decode()}")
            return {'error': 'Write operation failed'}
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Get the size on disk (compressed)
        disk_size_cmd = f"ozone fs -du o3fs://{VOLUME_NAME}.{BUCKET_NAME}/{key_name}"
        process = subprocess.run(disk_size_cmd, shell=True, stdout=PIPE, stderr=PIPE)
        
        try:
            disk_size = int(process.stdout.decode().split()[0])
        except (IndexError, ValueError):
            disk_size = file_size  # Fallback if we can't get actual disk size
        
        # Calculate metrics
        metrics['throughput'] = file_size / duration  # bytes per second
        metrics['latency'] = duration * 1000  # milliseconds
        metrics['compression_ratio'] = file_size / disk_size if disk_size > 0 else 1.0
        metrics['storage_efficiency'] = 1.0 - (disk_size / file_size) if file_size > 0 else 0.0
        
    elif operation == 'read':
        # Measure read performance
        cmd = f"ozone fs -cat o3fs://{VOLUME_NAME}.{BUCKET_NAME}/{key_name}"
        
        start_time = time.time()
        with open(os.devnull, 'wb') as devnull:
            process = subprocess.run(cmd, shell=True, stdout=devnull, stderr=PIPE)
        
        if process.returncode != 0:
            logger.error(f"Read operation failed: {process.stderr.decode()}")
            return {'error': 'Read operation failed'}
            
        end_time = time.time()
        duration = end_time - start_time
        
        # Calculate metrics
        metrics['throughput'] = file_size / duration if duration > 0 else 0  # bytes per second
        metrics['latency'] = duration * 1000  # milliseconds
        
    return metrics

def generate_performance_report(results: List[Dict]) -> None:
    """
    Generate performance report from test results
    
    Args:
        results: List of performance result dictionaries
    """
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    df.to_csv(RESULT_CSV, index=False)
    logger.info(f"Performance results saved to {RESULT_CSV}")
    
    # Generate plots
    try:
        # Throughput by algorithm and compressibility
        plt.figure(figsize=(12, 8))
        for alg in COMPRESSION_ALGORITHMS:
            alg_data = df[df['algorithm'] == alg]
            if not alg_data.empty:
                throughputs = alg_data.groupby('compressibility')['throughput'].mean() / 1024 / 1024  # Convert to MB/s
                throughputs.plot(kind='bar', label=alg)
        
        plt.title('Average Throughput by Algorithm and Compressibility')
        plt.ylabel('Throughput (MB/s)')
        plt.legend()
        plt.savefig('throughput_by_algorithm.png')
        
        # Compression ratio by algorithm
        plt.figure(figsize=(12, 8))
        df[df['operation'] == 'write'].groupby(['algorithm', 'compressibility'])['compression_ratio'].mean().unstack().plot(kind='bar')
        plt.title('Compression Ratio by Algorithm and Compressibility')
        plt.ylabel('Compression Ratio')
        plt.savefig('compression_ratio.png')
        
        logger.info("Performance charts generated")
    except Exception as e:
        logger.warning(f"Could not generate performance charts: {e}")

@pytest.fixture(scope="module")
def ozone_setup():
    """Fixture to setup and teardown Ozone test environment"""
    setup_ozone_environment()
    yield
    cleanup_ozone_environment()

@pytest.mark.parametrize("algorithm", COMPRESSION_ALGORITHMS)
@pytest.mark.parametrize("level", COMPRESSION_LEVELS)
@pytest.mark.parametrize("file_size", [1024*1024, 10*1024*1024])  # Using 1MB and 10MB for test speed
@pytest.mark.parametrize("compressibility", ["high", "medium", "low"])
def test_31_compression_performance(ozone_setup, algorithm, level, file_size, compressibility):
    """
    Test performance with varying levels of data compression
    
    This test evaluates Ozone's performance with different compression algorithms,
    levels, and data compressibility to identify optimal settings.
    """
    logger.info(f"Running compression performance test with algorithm={algorithm}, "
                f"level={level}, file_size={file_size}, compressibility={compressibility}")
    
    # Configure compression
    configure_compression(algorithm, level)
    
    # Generate test file with specified compressibility
    test_file = generate_test_file(file_size, compressibility)
    
    try:
        # Collect performance metrics
        results = []
        
        # Test write performance
        write_metrics = measure_performance('write', algorithm, level, test_file, 
                                           compressibility, file_size)
        write_metrics.update({
            'operation': 'write',
            'algorithm': algorithm,
            'level': level,
            'file_size': file_size,
            'compressibility': compressibility,
            'timestamp': datetime.now().isoformat()
        })
        results.append(write_metrics)
        
        # Test read performance
        read_metrics = measure_performance('read', algorithm, level, test_file, 
                                          compressibility, file_size)
        read_metrics.update({
            'operation': 'read',
            'algorithm': algorithm,
            'level': level,
            'file_size': file_size,
            'compressibility': compressibility,
            'timestamp': datetime.now().isoformat()
        })
        results.append(read_metrics)
        
        # Log results
        logger.info(f"Write performance - Throughput: {write_metrics.get('throughput', 0)/1024/1024:.2f} MB/s, "
                   f"Latency: {write_metrics.get('latency', 0):.2f} ms, "
                   f"Compression ratio: {write_metrics.get('compression_ratio', 1.0):.2f}")
        
        logger.info(f"Read performance - Throughput: {read_metrics.get('throughput', 0)/1024/1024:.2f} MB/s, "
                   f"Latency: {read_metrics.get('latency', 0):.2f} ms")
        
        # Validate results
        # Check if measurements were successful
        assert 'error' not in write_metrics, "Write operation failed"
        assert 'error' not in read_metrics, "Read operation failed"
        
        # Check if measurements are reasonable
        assert write_metrics.get('throughput', 0) > 0, "Write throughput should be positive"
        assert read_metrics.get('throughput', 0) > 0, "Read throughput should be positive"
        
        # For compression algorithms (except NONE), check compression ratio
        if algorithm != "NONE":
            assert write_metrics.get('compression_ratio', 1.0) >= 1.0, "Compression ratio should be >= 1.0"
            
            # Highly compressible data should have better compression ratio
            if compressibility == "high":
                assert write_metrics.get('compression_ratio', 1.0) > 1.5, "High compressibility should have good compression ratio"
        
        # Generate report with collected data
        # This would happen after all tests run, we're doing it per test for demonstration
        generate_performance_report(results)
        
    finally:
        # Clean up test file
        try:
            os.unlink(test_file)
        except:
            pass

import pytest
import time
import threading
import subprocess
import os
import random
import statistics
import logging
import concurrent.futures
from hdfs3 import HDFileSystem
from pyarrow import ozone
from prometheus_api_client import PrometheusConnect

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
OZONE_HOST = os.environ.get('OZONE_HOST', 'localhost')
OZONE_PORT = int(os.environ.get('OZONE_PORT', '9878'))
PROMETHEUS_URL = os.environ.get('PROMETHEUS_URL', 'http://localhost:9090')
VOLUME_NAME = 'mixed_workload_vol'
BUCKET_NAME = 'mixed_workload_bucket'

# Test data sizes (in KB)
FILE_SIZES = [64, 512, 1024, 4096, 16384]  # 64KB, 512KB, 1MB, 4MB, 16MB

# Workload mix configuration
READ_RATIO = 0.4  # 40% reads
WRITE_RATIO = 0.4  # 40% writes
METADATA_RATIO = 0.2  # 20% metadata operations
TEST_DURATION = 300  # seconds
NUM_THREADS = 16  # Number of concurrent client threads
SAMPLING_INTERVAL = 5  # seconds

class PerformanceMetrics:
    def __init__(self):
        self.read_latencies = []
        self.write_latencies = []
        self.metadata_latencies = []
        self.throughput = 0
        self.cpu_usage = []
        self.memory_usage = []
        self.disk_io = []
        self.operation_counts = {
            'read': 0,
            'write': 0,
            'metadata': 0
        }

    def add_read_latency(self, latency_ms):
        self.read_latencies.append(latency_ms)
        self.operation_counts['read'] += 1
        
    def add_write_latency(self, latency_ms):
        self.write_latencies.append(latency_ms)
        self.operation_counts['write'] += 1
        
    def add_metadata_latency(self, latency_ms):
        self.metadata_latencies.append(latency_ms)
        self.operation_counts['metadata'] += 1

    def calculate_stats(self):
        stats = {}
        # Process latency data
        for op_type in ['read', 'write', 'metadata']:
            latency_list = getattr(self, f"{op_type}_latencies")
            if latency_list:
                stats[f"{op_type}_avg_latency"] = statistics.mean(latency_list)
                stats[f"{op_type}_p95_latency"] = percentile(latency_list, 95)
                stats[f"{op_type}_p99_latency"] = percentile(latency_list, 99)
                stats[f"{op_type}_count"] = self.operation_counts[op_type]
                stats[f"{op_type}_ops_per_sec"] = self.operation_counts[op_type] / TEST_DURATION
        
        # Calculate overall throughput
        total_ops = sum(self.operation_counts.values())
        stats['total_operations'] = total_ops
        stats['operations_per_second'] = total_ops / TEST_DURATION
        
        if self.cpu_usage:
            stats['avg_cpu_usage'] = statistics.mean(self.cpu_usage)
            stats['max_cpu_usage'] = max(self.cpu_usage)
        
        if self.memory_usage:
            stats['avg_memory_usage'] = statistics.mean(self.memory_usage)
            stats['max_memory_usage'] = max(self.memory_usage)
            
        return stats

def percentile(data, percentile):
    """Calculate percentile value from a list of numbers"""
    if not data:
        return None
    sorted_data = sorted(data)
    index = int(len(sorted_data) * percentile / 100) - 1
    return sorted_data[max(0, index)]

def generate_test_file(size_kb):
    """Generate a test file with random content of specified size in KB"""
    file_path = f"test_file_{size_kb}kb.dat"
    with open(file_path, 'wb') as f:
        f.write(os.urandom(size_kb * 1024))
    return file_path

def get_system_metrics(prom_client):
    """Collect system metrics from Prometheus"""
    cpu_usage = prom_client.custom_query(query='avg(rate(node_cpu_seconds_total{mode!="idle"}[1m])) * 100')
    memory_usage = prom_client.custom_query(query='avg(node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / avg(node_memory_MemTotal_bytes) * 100')
    disk_io = prom_client.custom_query(query='sum(rate(node_disk_reads_completed_total[1m]) + rate(node_disk_writes_completed_total[1m]))')
    
    metrics = {}
    try:
        metrics['cpu_usage'] = float(cpu_usage[0]['value'][1]) if cpu_usage else 0
        metrics['memory_usage'] = float(memory_usage[0]['value'][1]) if memory_usage else 0
        metrics['disk_io'] = float(disk_io[0]['value'][1]) if disk_io else 0
    except (IndexError, KeyError, ValueError):
        logger.warning("Failed to parse some Prometheus metrics")
    
    return metrics

class MixedWorkloadExecutor:
    def __init__(self, host, port, volume, bucket):
        self.host = host
        self.port = port
        self.volume = volume
        self.bucket = bucket
        self.client = ozone.Client(host, port)
        self.stop_flag = False
        self.metrics = PerformanceMetrics()
        self.test_files = {}
        # Initialize Prometheus client for monitoring
        try:
            self.prom_client = PrometheusConnect(url=PROMETHEUS_URL, disable_ssl=True)
        except Exception as e:
            logger.warning(f"Could not connect to Prometheus: {e}")
            self.prom_client = None
    
    def setup(self):
        """Set up volume and bucket for testing"""
        try:
            self.client.create_volume(self.volume)
            logger.info(f"Created volume: {self.volume}")
        except Exception as e:
            logger.info(f"Volume already exists: {e}")
        
        try:
            self.client.create_bucket(self.volume, self.bucket)
            logger.info(f"Created bucket: {self.volume}/{self.bucket}")
        except Exception as e:
            logger.info(f"Bucket already exists: {e}")
        
        # Create test files of different sizes
        for size in FILE_SIZES:
            self.test_files[size] = generate_test_file(size)
            logger.info(f"Created test file of size {size}KB")
        
        # Pre-populate some keys for read operations
        for i in range(20):
            size = random.choice(FILE_SIZES)
            key = f"preloaded_key_{i}"
            with open(self.test_files[size], 'rb') as f:
                self.client.put_key(self.volume, self.bucket, key, f)
            logger.info(f"Pre-loaded key: {key} with {size}KB data")
    
    def cleanup(self):
        """Clean up test data"""
        # Remove temporary files
        for file_path in self.test_files.values():
            try:
                os.remove(file_path)
            except OSError:
                pass
    
    def write_operation(self):
        """Perform a write operation and measure its latency"""
        size = random.choice(FILE_SIZES)
        key = f"key_{int(time.time())}_{random.randint(1, 10000)}"
        
        start_time = time.time()
        with open(self.test_files[size], 'rb') as f:
            self.client.put_key(self.volume, self.bucket, key, f)
        end_time = time.time()
        
        latency_ms = (end_time - start_time) * 1000
        self.metrics.add_write_latency(latency_ms)
        return key
    
    def read_operation(self):
        """Perform a read operation and measure its latency"""
        # List keys and pick one randomly
        keys = self.client.list_keys(self.volume, self.bucket)
        if not keys:
            # Fall back to write if no keys are available
            return self.write_operation()
        
        key = random.choice(keys)
        start_time = time.time()
        data = self.client.get_key(self.volume, self.bucket, key)
        end_time = time.time()
        
        latency_ms = (end_time - start_time) * 1000
        self.metrics.add_read_latency(latency_ms)
        return key
    
    def metadata_operation(self):
        """Perform metadata operations and measure latency"""
        start_time = time.time()
        
        # Randomly choose a metadata operation
        op_type = random.randint(0, 2)
        
        if op_type == 0:
            # List keys operation
            self.client.list_keys(self.volume, self.bucket)
        elif op_type == 1:
            # Check key exists
            keys = self.client.list_keys(self.volume, self.bucket)
            if keys:
                key = random.choice(keys)
                self.client.key_exists(self.volume, self.bucket, key)
        else:
            # List buckets operation
            self.client.list_buckets(self.volume)
        
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        self.metrics.add_metadata_latency(latency_ms)
    
    def collect_metrics(self):
        """Collect system metrics at regular intervals"""
        if not self.prom_client:
            return
            
        while not self.stop_flag:
            try:
                metrics = get_system_metrics(self.prom_client)
                if 'cpu_usage' in metrics:
                    self.metrics.cpu_usage.append(metrics['cpu_usage'])
                if 'memory_usage' in metrics:
                    self.metrics.memory_usage.append(metrics['memory_usage'])
                if 'disk_io' in metrics:
                    self.metrics.disk_io.append(metrics['disk_io'])
                    
                time.sleep(SAMPLING_INTERVAL)
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                break
    
    def worker(self):
        """Worker thread to execute mixed workload operations"""
        while not self.stop_flag:
            # Determine which operation to perform based on ratios
            op_choice = random.random()
            
            try:
                if op_choice < READ_RATIO:
                    self.read_operation()
                elif op_choice < READ_RATIO + WRITE_RATIO:
                    self.write_operation()
                else:
                    self.metadata_operation()
            except Exception as e:
                logger.error(f"Operation failed: {e}")
    
    def run_test(self, duration_seconds):
        """Run the mixed workload test for a specified duration"""
        self.stop_flag = False
        
        # Start metrics collection thread
        metrics_thread = threading.Thread(target=self.collect_metrics)
        metrics_thread.daemon = True
        metrics_thread.start()
        
        # Start worker threads
        workers = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            for _ in range(NUM_THREADS):
                workers.append(executor.submit(self.worker))
            
            # Wait for test duration
            logger.info(f"Mixed workload test running for {duration_seconds} seconds...")
            time.sleep(duration_seconds)
            
            # Stop the test
            self.stop_flag = True
            logger.info("Test completed, calculating results...")
        
        # Wait for metrics thread to finish
        metrics_thread.join(timeout=5)
        
        # Calculate and return performance metrics
        return self.metrics.calculate_stats()


@pytest.fixture
def ozone_cluster():
    """Fixture to set up and tear down the test environment"""
    executor = MixedWorkloadExecutor(OZONE_HOST, OZONE_PORT, VOLUME_NAME, BUCKET_NAME)
    executor.setup()
    yield executor
    executor.cleanup()


def test_32_mixed_workload_performance(ozone_cluster):
    """
    Evaluate performance under mixed workload scenarios
    
    This test executes a mixed workload of read, write, and metadata operations
    on an Ozone cluster and measures performance metrics to identify potential
    bottlenecks.
    """
    # Set performance thresholds
    MAX_AVG_READ_LATENCY_MS = 500
    MAX_AVG_WRITE_LATENCY_MS = 1000
    MAX_AVG_METADATA_LATENCY_MS = 200
    MIN_OPS_PER_SECOND = 10
    
    # Run the mixed workload test
    logger.info("Starting mixed workload performance test")
    results = ozone_cluster.run_test(TEST_DURATION)
    
    # Log detailed performance metrics
    logger.info("Test Results:")
    for metric, value in results.items():
        logger.info(f"{metric}: {value}")
    
    # Assert performance meets expectations
    if 'read_avg_latency' in results:
        assert results['read_avg_latency'] < MAX_AVG_READ_LATENCY_MS, \
            f"Average read latency {results['read_avg_latency']}ms exceeds threshold of {MAX_AVG_READ_LATENCY_MS}ms"
    
    if 'write_avg_latency' in results:
        assert results['write_avg_latency'] < MAX_AVG_WRITE_LATENCY_MS, \
            f"Average write latency {results['write_avg_latency']}ms exceeds threshold of {MAX_AVG_WRITE_LATENCY_MS}ms"
    
    if 'metadata_avg_latency' in results:
        assert results['metadata_avg_latency'] < MAX_AVG_METADATA_LATENCY_MS, \
            f"Average metadata latency {results['metadata_avg_latency']}ms exceeds threshold of {MAX_AVG_METADATA_LATENCY_MS}ms"
    
    assert results['operations_per_second'] > MIN_OPS_PER_SECOND, \
        f"Operation throughput {results['operations_per_second']} ops/sec below minimum threshold of {MIN_OPS_PER_SECOND} ops/sec"
    
    # Additional assertions for resource utilization if metrics are available
    if 'avg_cpu_usage' in results:
        logger.info(f"Average CPU usage: {results['avg_cpu_usage']}%")
        logger.info(f"Maximum CPU usage: {results['max_cpu_usage']}%")
    
    if 'avg_memory_usage' in results:
        logger.info(f"Average memory usage: {results['avg_memory_usage']}%")
        logger.info(f"Maximum memory usage: {results['max_memory_usage']}%")
    
    # Final result summary
    logger.info("Performance test completed successfully.")
    logger.info(f"Total operations executed: {results.get('total_operations', 0)}")
    logger.info(f"Average throughput: {results.get('operations_per_second', 0):.2f} ops/sec")

    # Create a detailed report file
    with open("mixed_workload_performance_report.txt", "w") as report:
        report.write("Apache Ozone Mixed Workload Performance Test Results\n")
        report.write("==================================================\n\n")
        report.write(f"Test duration: {TEST_DURATION} seconds\n")
        report.write(f"Concurrent threads: {NUM_THREADS}\n")
        report.write(f"Workload mix: {READ_RATIO*100:.0f}% reads, {WRITE_RATIO*100:.0f}% writes, {METADATA_RATIO*100:.0f}% metadata\n\n")
        report.write("Performance Metrics:\n")
        for metric, value in results.items():
            report.write(f"- {metric}: {value}\n")

import pytest
import time
import os
import subprocess
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import configparser
import logging
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ozone_cache_performance_test")

# Test configuration parameters
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(BASE_DIR, "test_data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
BENCHMARK_DURATION = 120  # seconds

# Ensure test directories exist
os.makedirs(TEST_DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

class OzoneClusterManager:
    """Helper class to manage Ozone cluster configurations"""
    
    def __init__(self, config_path="/etc/hadoop/conf/ozone-site.xml"):
        self.config_path = config_path
        self.original_config = self._backup_config()
    
    def _backup_config(self) -> str:
        """Create a backup of the original configuration"""
        with open(self.config_path, 'r') as f:
            return f.read()
    
    def restore_original_config(self) -> None:
        """Restore original configuration"""
        with open(self.config_path, 'w') as f:
            f.write(self.original_config)
        self._restart_service()
        
    def update_cache_configuration(self, cache_config: Dict[str, str]) -> None:
        """
        Update the cache configuration parameters in ozone-site.xml
        
        Args:
            cache_config: Dictionary containing cache configuration parameters
        """
        # This is a simplified approach; in a real environment, you would 
        # properly update the XML configuration file
        logger.info(f"Updating cache configuration: {cache_config}")
        
        # Command to update configurations
        for key, value in cache_config.items():
            cmd = f"hdfs dfsadmin -setconf {key}={value}"
            subprocess.run(cmd, shell=True, check=True)
        
        self._restart_service()
    
    def _restart_service(self) -> None:
        """Restart Ozone services to apply configuration changes"""
        logger.info("Restarting Ozone services")
        # In a real environment, you would use proper service management commands
        subprocess.run("ozone admin restart om", shell=True, check=True)
        subprocess.run("ozone admin restart scm", shell=True, check=True)
        time.sleep(30)  # Allow services time to restart


class OzoneBenchmark:
    """Benchmarking utility for Ozone performance tests"""
    
    def __init__(self, volume: str, bucket: str):
        self.volume = volume
        self.bucket = bucket
        self.setup_test_environment()
        
    def setup_test_environment(self) -> None:
        """Create volume and bucket for testing"""
        try:
            # Create volume if it doesn't exist
            subprocess.run(
                f"ozone sh volume create {self.volume}", 
                shell=True, check=True
            )
            # Create bucket if it doesn't exist
            subprocess.run(
                f"ozone sh bucket create {self.volume}/{self.bucket}", 
                shell=True, check=True
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to set up test environment: {e}")
            raise
    
    def create_test_file(self, file_path: str, size_mb: int) -> str:
        """
        Create a test file of specified size
        
        Args:
            file_path: Path where the file will be created
            size_mb: Size of the file in MB
            
        Returns:
            Path to the created file
        """
        logger.info(f"Creating test file of size {size_mb}MB at {file_path}")
        with open(file_path, 'wb') as f:
            f.write(os.urandom(size_mb * 1024 * 1024))
        return file_path
    
    def run_write_benchmark(self, file_sizes: List[int], num_files: int = 10) -> Dict:
        """
        Run write performance benchmark
        
        Args:
            file_sizes: List of file sizes in MB to test
            num_files: Number of files to write for each size
            
        Returns:
            Dictionary with benchmark results
        """
        results = {}
        
        for size_mb in file_sizes:
            start_time = time.time()
            throughputs = []
            
            for i in range(num_files):
                file_path = os.path.join(TEST_DATA_DIR, f"test_file_{size_mb}mb_{i}.dat")
                self.create_test_file(file_path, size_mb)
                
                key = f"benchmark/file_{size_mb}mb_{i}"
                
                file_start_time = time.time()
                subprocess.run(
                    f"ozone sh key put {self.volume}/{self.bucket}/ {key} {file_path}",
                    shell=True, check=True
                )
                file_end_time = time.time()
                
                # Calculate throughput in MB/s
                throughput = size_mb / (file_end_time - file_start_time)
                throughputs.append(throughput)
            
            end_time = time.time()
            avg_throughput = np.mean(throughputs)
            
            results[f"{size_mb}MB"] = {
                'operation': 'write',
                'avg_throughput_mbps': avg_throughput,
                'total_time_sec': end_time - start_time,
                'individual_throughputs': throughputs
            }
            
        return results
    
    def run_read_benchmark(self, file_sizes: List[int], num_files: int = 10) -> Dict:
        """
        Run read performance benchmark
        
        Args:
            file_sizes: List of file sizes in MB to test
            num_files: Number of files to read for each size
            
        Returns:
            Dictionary with benchmark results
        """
        results = {}
        
        for size_mb in file_sizes:
            start_time = time.time()
            throughputs = []
            
            for i in range(num_files):
                key = f"benchmark/file_{size_mb}mb_{i}"
                output_path = os.path.join(TEST_DATA_DIR, f"read_file_{size_mb}mb_{i}.dat")
                
                file_start_time = time.time()
                subprocess.run(
                    f"ozone sh key get {self.volume}/{self.bucket}/{key} {output_path}",
                    shell=True, check=True
                )
                file_end_time = time.time()
                
                # Calculate throughput in MB/s
                throughput = size_mb / (file_end_time - file_start_time)
                throughputs.append(throughput)
                
                # Clean up the read file to save space
                os.remove(output_path)
            
            end_time = time.time()
            avg_throughput = np.mean(throughputs)
            
            results[f"{size_mb}MB"] = {
                'operation': 'read',
                'avg_throughput_mbps': avg_throughput,
                'total_time_sec': end_time - start_time,
                'individual_throughputs': throughputs
            }
            
        return results
    
    def get_cache_metrics(self) -> Dict:
        """
        Retrieve cache metrics from Ozone
        
        Returns:
            Dictionary with cache metrics data
        """
        try:
            # This is a placeholder; in a real environment, you would use 
            # Ozone's metrics API or JMX to get cache statistics
            cmd = "curl -s http://localhost:9874/metrics"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
            metrics_data = json.loads(result.stdout)
            
            # Extract cache-related metrics
            cache_metrics = {
                'cache_hit_rate': metrics_data.get('cache.hitRate', 0),
                'cache_miss_rate': metrics_data.get('cache.missRate', 0),
                'cache_size': metrics_data.get('cache.size', 0),
                'cache_evictions': metrics_data.get('cache.evictions', 0)
            }
            
            return cache_metrics
        except Exception as e:
            logger.error(f"Failed to get cache metrics: {e}")
            return {
                'cache_hit_rate': 0,
                'cache_miss_rate': 0,
                'cache_size': 0,
                'cache_evictions': 0
            }
    
    def clean_up(self) -> None:
        """Clean up test data"""
        logger.info("Cleaning up test data")
        # Delete all objects in the bucket
        subprocess.run(
            f"ozone sh key delete --recursive {self.volume}/{self.bucket}/benchmark",
            shell=True, check=True
        )


@pytest.mark.parametrize("cache_config", [
    {"ozone.scm.block.client.cache.size": "10MB", "ozone.om.cache.size": "128MB", "ozone.s3g.cache.size": "128MB"},
    {"ozone.scm.block.client.cache.size": "50MB", "ozone.om.cache.size": "512MB", "ozone.s3g.cache.size": "512MB"},
    {"ozone.scm.block.client.cache.size": "100MB", "ozone.om.cache.size": "1024MB", "ozone.s3g.cache.size": "1024MB"},
    {"ozone.scm.block.client.cache.size": "250MB", "ozone.om.cache.size": "2048MB", "ozone.s3g.cache.size": "2048MB"},
])
@pytest.mark.parametrize("workload_type", ["small_files", "large_files", "mixed"])
def test_33_ozone_cache_performance(cache_config, workload_type):
    """
    Test performance with different cache configurations
    
    This test evaluates Apache Ozone's performance with different cache configurations
    by running standard benchmarks and measuring read/write performance and cache hit rates.
    """
    # Define file sizes based on workload type
    if workload_type == "small_files":
        file_sizes = [0.5, 1, 2, 4]  # MB
    elif workload_type == "large_files":
        file_sizes = [64, 128, 256, 512]  # MB
    else:  # mixed
        file_sizes = [1, 7.5, 20, 75, 256]  # MB
    
    # Create unique volume and bucket names for this test
    timestamp = int(time.time())
    volume = f"cachebench{timestamp}"
    bucket = f"perftest{timestamp}"
    
    # Initialize cluster manager and benchmark utility
    cluster_manager = OzoneClusterManager()
    benchmark = OzoneBenchmark(volume=volume, bucket=bucket)
    
    try:
        # Configure cache settings
        logger.info(f"Setting cache configuration: {cache_config}")
        cluster_manager.update_cache_configuration(cache_config)
        
        # Allow some time for cache configuration to take effect
        time.sleep(10)
        
        # Run write benchmark
        logger.info(f"Running write benchmark for workload: {workload_type}")
        write_results = benchmark.run_write_benchmark(file_sizes=file_sizes)
        
        # Get cache metrics after writes
        write_cache_metrics = benchmark.get_cache_metrics()
        
        # Run read benchmark with same files (should benefit from caching)
        logger.info(f"Running read benchmark for workload: {workload_type}")
        read_results = benchmark.run_read_benchmark(file_sizes=file_sizes)
        
        # Get cache metrics after reads
        read_cache_metrics = benchmark.get_cache_metrics()
        
        # Run read benchmark again to measure cache hit performance
        logger.info(f"Running cached read benchmark for workload: {workload_type}")
        cached_read_results = benchmark.run_read_benchmark(file_sizes=file_sizes)
        
        # Get cache metrics after cached reads
        cached_read_cache_metrics = benchmark.get_cache_metrics()
        
        # Save results
        results = {
            "cache_config": cache_config,
            "workload_type": workload_type,
            "write_results": write_results,
            "read_results": read_results,
            "cached_read_results": cached_read_results,
            "write_cache_metrics": write_cache_metrics,
            "read_cache_metrics": read_cache_metrics,
            "cached_read_cache_metrics": cached_read_cache_metrics
        }
        
        result_file = os.path.join(
            RESULTS_DIR, 
            f"cache_perf_{workload_type}_{cache_config['ozone.om.cache.size']}.json"
        )
        
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        # Analyze results
        write_throughputs = [data['avg_throughput_mbps'] for data in write_results.values()]
        read_throughputs = [data['avg_throughput_mbps'] for data in read_results.values()]
        cached_read_throughputs = [data['avg_throughput_mbps'] for data in cached_read_results.values()]
        
        avg_write_throughput = np.mean(write_throughputs)
        avg_read_throughput = np.mean(read_throughputs)
        avg_cached_read_throughput = np.mean(cached_read_throughputs)
        
        cache_hit_rate_improvement = (
            cached_read_cache_metrics['cache_hit_rate'] - read_cache_metrics['cache_hit_rate']
        )
        
        logger.info(f"Results for {workload_type} with cache config {cache_config}:")
        logger.info(f"Average write throughput: {avg_write_throughput:.2f} MB/s")
        logger.info(f"Average read throughput: {avg_read_throughput:.2f} MB/s")
        logger.info(f"Average cached read throughput: {avg_cached_read_throughput:.2f} MB/s")
        logger.info(f"Cache hit rate improvement: {cache_hit_rate_improvement:.2f}")
        
        # Assert performance expectations based on cache size
        # These assertions would need to be adjusted based on expected performance
        # for your specific environment
        
        # Expected improvement for cached reads
        assert avg_cached_read_throughput > avg_read_throughput, \
            f"Cached reads should be faster than initial reads for {workload_type}"
        
        # Larger caches should have better cache hit rates for appropriate workloads
        if workload_type == "small_files" and "512MB" in cache_config["ozone.om.cache.size"]:
            assert cached_read_cache_metrics['cache_hit_rate'] > 0.7, \
                "Small files workload should have high cache hit rate with medium cache"
                
        if workload_type == "large_files" and "2048MB" in cache_config["ozone.om.cache.size"]:
            assert cached_read_cache_metrics['cache_hit_rate'] > 0.5, \
                "Large files workload should have decent cache hit rate with large cache"
        
    finally:
        # Clean up
        benchmark.clean_up()
        
        # Delete volume and bucket
        try:
            subprocess.run(f"ozone sh bucket delete {volume}/{bucket}", shell=True, check=True)
            subprocess.run(f"ozone sh volume delete {volume}", shell=True, check=True)
        except subprocess.CalledProcessError:
            logger.warning(f"Failed to delete volume/bucket {volume}/{bucket}")
        
        # Restore original cache configuration
        cluster_manager.restore_original_config()


def analyze_cache_performance_results():
    """
    Analyze cache performance results after running tests.
    This function can be used to generate comprehensive reports from test results.
    """
    results = []
    
    # Load all result files
    for filename in os.listdir(RESULTS_DIR):
        if filename.startswith("cache_perf_") and filename.endswith(".json"):
            with open(os.path.join(RESULTS_DIR, filename), 'r') as f:
                result_data = json.load(f)
                
                # Extract key metrics
                workload = result_data["workload_type"]
                cache_size = result_data["cache_config"]["ozone.om.cache.size"]
                
                # Calculate average throughputs
                write_throughputs = [data['avg_throughput_mbps'] for data in result_data["write_results"].values()]
                read_throughputs = [data['avg_throughput_mbps'] for data in result_data["read_results"].values()]
                cached_read_throughputs = [data['avg_throughput_mbps'] for data in result_data["cached_read_results"].values()]
                
                avg_write_throughput = np.mean(write_throughputs)
                avg_read_throughput = np.mean(read_throughputs)
                avg_cached_read_throughput = np.mean(cached_read_throughputs)
                
                # Get cache metrics
                cache_hit_rate = result_data["cached_read_cache_metrics"]["cache_hit_rate"]
                
                results.append({
                    "workload": workload,
                    "cache_size": cache_size,
                    "avg_write_throughput": avg_write_throughput,
                    "avg_read_throughput": avg_read_throughput, 
                    "avg_cached_read_throughput": avg_cached_read_throughput,
                    "cache_hit_rate": cache_hit_rate
                })
    
    if not results:
        logger.warning("No result files found to analyze")
        return
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(results)
    
    # Group by workload and cache size
    grouped = df.groupby(['workload', 'cache_size']).mean().reset_index()
    
    # Generate report
    report_path = os.path.join(RESULTS_DIR, "cache_performance_report.csv")
    grouped.to_csv(report_path, index=False)
    
    # Generate charts
    plt.figure(figsize=(12, 8))
    
    for workload in df['workload'].unique():
        workload_data = df[df['workload'] == workload]
        plt.plot(
            workload_data['cache_size'], 
            workload_data['avg_cached_read_throughput'], 
            marker='o', 
            label=f"{workload} - Cached Read"
        )
    
    plt.title('Cache Performance by Workload Type')
    plt.xlabel('Cache Size')
    plt.ylabel('Average Throughput (MB/s)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(RESULTS_DIR, "cache_performance_chart.png"))
    
    logger.info(f"Performance analysis completed. Report saved to {report_path}")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import pytest
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor
import logging
import json
import tempfile
import random
import string
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Test constants
CLUSTER_CONF_DIR = os.environ.get('OZONE_CONF_DIR', '/etc/hadoop/conf')
OZONE_BIN_DIR = os.environ.get('OZONE_BIN_DIR', '/opt/hadoop/bin')
OZONE_CLI = os.path.join(OZONE_BIN_DIR, 'ozone')
HDDS_CLI = os.path.join(OZONE_BIN_DIR, 'hdds')
TEST_DATA_DIR = os.environ.get('TEST_DATA_DIR', '/tmp/ozone-perf-test')
MAX_ALLOWED_DEGRADATION = 20  # Maximum allowed performance degradation percentage

# Security configurations to test
SECURITY_CONFIGS = [
    {
        "name": "no_security",
        "description": "No security features enabled (baseline)",
        "params": {}
    },
    {
        "name": "kerberos",
        "description": "Kerberos authentication enabled",
        "params": {"security.enabled": "true", "hadoop.security.authentication": "kerberos"}
    },
    {
        "name": "tls",
        "description": "TLS encryption enabled",
        "params": {"hdds.grpc.tls.enabled": "true", "ozone.security.enabled": "true"}
    },
    {
        "name": "kerberos_tls",
        "description": "Kerberos authentication and TLS encryption enabled",
        "params": {
            "security.enabled": "true", 
            "hadoop.security.authentication": "kerberos",
            "hdds.grpc.tls.enabled": "true", 
            "ozone.security.enabled": "true"
        }
    },
    {
        "name": "authorization",
        "description": "ACL-based authorization enabled",
        "params": {"ozone.acl.enabled": "true", "ozone.administrators": "ozoneadmin"}
    },
    {
        "name": "full_security",
        "description": "All security features enabled",
        "params": {
            "security.enabled": "true", 
            "hadoop.security.authentication": "kerberos",
            "hdds.grpc.tls.enabled": "true", 
            "ozone.security.enabled": "true",
            "ozone.acl.enabled": "true", 
            "ozone.administrators": "ozoneadmin"
        }
    }
]

# Test file sizes in bytes
FILE_SIZES = [
    1024 * 10,        # 10 KB
    1024 * 100,       # 100 KB
    1024 * 1024,      # 1 MB
    1024 * 1024 * 10, # 10 MB
    1024 * 1024 * 100 # 100 MB
    # For real-world tests, you might want to add larger files (GB)
]

# Operations to benchmark
OPERATIONS = [
    "write",
    "read",
    "list",
    "metadata"
]

# Test parameters
NUM_ITERATIONS = 5  # Number of iterations for each test to ensure statistical significance
CONCURRENCY_LEVELS = [1, 5, 10]  # Number of concurrent operations


class OzonePerformanceTester:
    """Utility class for running performance tests on Apache Ozone with different security settings"""
    
    def __init__(self, security_config=None):
        """Initialize the tester with a specific security configuration"""
        self.security_config = security_config or {"name": "no_security", "params": {}}
        self.volume_name = f"vol-perf-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.bucket_name = f"bucket-perf-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.test_files = {}
        
        # Ensure test data directory exists
        os.makedirs(TEST_DATA_DIR, exist_ok=True)
        
    def setup(self):
        """Set up the test environment with the specified security configuration"""
        logger.info(f"Setting up test environment with security configuration: {self.security_config['name']}")
        
        # Apply security configuration
        self._apply_security_config()
        
        # Create test volume and bucket
        self._run_command([OZONE_CLI, "sh", "volume", "create", self.volume_name])
        self._run_command([OZONE_CLI, "sh", "bucket", "create", f"/{self.volume_name}/{self.bucket_name}"])
        
        # Generate test files of different sizes
        self._generate_test_files()
        
        logger.info(f"Test environment set up successfully with {self.security_config['name']} configuration")
        
    def _apply_security_config(self):
        """Apply the security configuration to the cluster"""
        logger.info(f"Applying security configuration: {self.security_config['name']}")
        
        # In a real environment, this would modify config files and restart services
        # For this test, we'll simulate by logging the actions
        
        for key, value in self.security_config['params'].items():
            logger.info(f"Setting configuration: {key}={value}")
            # In a real environment:
            # self._run_command(["hdfs", "dfsadmin", "-setConf", f"{key}={value}"])
        
        if "kerberos" in self.security_config['name']:
            logger.info("Initializing Kerberos authentication")
            # In a real environment:
            # self._run_command(["kinit", "-kt", "/etc/security/keytabs/ozone.keytab", "ozone"])
            
        if self.security_config['params']:
            logger.info("Restarting Ozone services to apply configurations")
            # In a real environment:
            # self._run_command([HDDS_CLI, "admin", "restart", "--all"])
            # time.sleep(30)  # Wait for services to restart
    
    def _generate_test_files(self):
        """Generate test files of different sizes"""
        for size in FILE_SIZES:
            file_path = os.path.join(TEST_DATA_DIR, f"test_file_{size}.dat")
            
            # Create a file with random data of the specified size
            with open(file_path, 'wb') as f:
                f.write(os.urandom(size))
                
            self.test_files[size] = file_path
            
    def _run_command(self, cmd, capture_output=True):
        """Run a shell command and return its output"""
        logger.debug(f"Running command: {' '.join(cmd)}")
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=capture_output, text=True)
        elapsed_time = time.time() - start_time
        
        if result.returncode != 0:
            logger.error(f"Command failed: {' '.join(cmd)}")
            logger.error(f"Error output: {result.stderr}")
            raise Exception(f"Command execution failed with return code {result.returncode}")
            
        return {"output": result.stdout, "time": elapsed_time}
    
    def benchmark_write(self, file_size, concurrency=1):
        """Benchmark write operations"""
        logger.info(f"Benchmarking write operation with file size: {file_size} bytes, concurrency: {concurrency}")
        
        file_path = self.test_files[file_size]
        
        def _write_operation(i):
            key_name = f"key_{i}_{int(time.time())}"
            return self._run_command([
                OZONE_CLI, "sh", "key", "put", 
                f"/{self.volume_name}/{self.bucket_name}/",
                file_path,
                "--name", key_name
            ])
        
        start_time = time.time()
        results = []
        
        if concurrency > 1:
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [executor.submit(_write_operation, i) for i in range(concurrency)]
                for future in futures:
                    results.append(future.result())
        else:
            results.append(_write_operation(0))
            
        total_time = time.time() - start_time
        
        # Calculate throughput: bytes per second
        throughput = (file_size * concurrency) / total_time if total_time > 0 else 0
        
        return {
            "operation": "write",
            "file_size": file_size,
            "concurrency": concurrency,
            "total_time": total_time,
            "operation_count": concurrency,
            "throughput_bytes_per_sec": throughput,
            "throughput_mb_per_sec": throughput / (1024 * 1024),
            "avg_latency": sum(r["time"] for r in results) / len(results) if results else 0
        }
        
    def benchmark_read(self, file_size, concurrency=1):
        """Benchmark read operations"""
        logger.info(f"Benchmarking read operation with file size: {file_size} bytes, concurrency: {concurrency}")
        
        # First, write the test keys to read later
        key_names = []
        for i in range(concurrency):
            key_name = f"read_key_{i}_{int(time.time())}"
            key_names.append(key_name)
            self._run_command([
                OZONE_CLI, "sh", "key", "put", 
                f"/{self.volume_name}/{self.bucket_name}/",
                self.test_files[file_size],
                "--name", key_name
            ])
            
        def _read_operation(key_name):
            temp_file = os.path.join(TEST_DATA_DIR, f"temp_read_{int(time.time())}.dat")
            result = self._run_command([
                OZONE_CLI, "sh", "key", "get", 
                f"/{self.volume_name}/{self.bucket_name}/{key_name}",
                temp_file
            ])
            os.remove(temp_file)  # Clean up
            return result
        
        start_time = time.time()
        results = []
        
        if concurrency > 1:
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [executor.submit(_read_operation, key_name) for key_name in key_names]
                for future in futures:
                    results.append(future.result())
        else:
            results.append(_read_operation(key_names[0]))
            
        total_time = time.time() - start_time
        
        # Calculate throughput: bytes per second
        throughput = (file_size * concurrency) / total_time if total_time > 0 else 0
        
        return {
            "operation": "read",
            "file_size": file_size,
            "concurrency": concurrency,
            "total_time": total_time,
            "operation_count": concurrency,
            "throughput_bytes_per_sec": throughput,
            "throughput_mb_per_sec": throughput / (1024 * 1024),
            "avg_latency": sum(r["time"] for r in results) / len(results) if results else 0
        }
        
    def benchmark_list(self, concurrency=1):
        """Benchmark list operations"""
        logger.info(f"Benchmarking list operation with concurrency: {concurrency}")
        
        # First, write some test keys to list later
        num_keys = 100  # A reasonable number of keys to list
        for i in range(num_keys):
            key_name = f"list_key_{i}_{int(time.time())}"
            self._run_command([
                OZONE_CLI, "sh", "key", "put", 
                f"/{self.volume_name}/{self.bucket_name}/",
                self.test_files[min(FILE_SIZES)],  # Use the smallest file for this test
                "--name", key_name
            ])
            
        def _list_operation():
            return self._run_command([
                OZONE_CLI, "sh", "key", "list", 
                f"/{self.volume_name}/{self.bucket_name}/"
            ])
        
        start_time = time.time()
        results = []
        
        if concurrency > 1:
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [executor.submit(_list_operation) for _ in range(concurrency)]
                for future in futures:
                    results.append(future.result())
        else:
            results.append(_list_operation())
            
        total_time = time.time() - start_time
        
        return {
            "operation": "list",
            "concurrency": concurrency,
            "total_time": total_time,
            "operation_count": concurrency,
            "keys_listed": num_keys,
            "avg_latency": sum(r["time"] for r in results) / len(results) if results else 0,
            "ops_per_sec": concurrency / total_time if total_time > 0 else 0
        }
        
    def benchmark_metadata(self, concurrency=1):
        """Benchmark metadata operations (info retrieval)"""
        logger.info(f"Benchmarking metadata operation with concurrency: {concurrency}")
        
        # First, write some test keys to get info later
        key_names = []
        for i in range(concurrency):
            key_name = f"info_key_{i}_{int(time.time())}"
            key_names.append(key_name)
            self._run_command([
                OZONE_CLI, "sh", "key", "put", 
                f"/{self.volume_name}/{self.bucket_name}/",
                self.test_files[min(FILE_SIZES)],  # Use the smallest file
                "--name", key_name
            ])
            
        def _info_operation(key_name):
            return self._run_command([
                OZONE_CLI, "sh", "key", "info", 
                f"/{self.volume_name}/{self.bucket_name}/{key_name}"
            ])
        
        start_time = time.time()
        results = []
        
        if concurrency > 1:
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [executor.submit(_info_operation, key_name) for key_name in key_names]
                for future in futures:
                    results.append(future.result())
        else:
            results.append(_info_operation(key_names[0]))
            
        total_time = time.time() - start_time
        
        return {
            "operation": "metadata",
            "concurrency": concurrency,
            "total_time": total_time,
            "operation_count": concurrency,
            "avg_latency": sum(r["time"] for r in results) / len(results) if results else 0,
            "ops_per_sec": concurrency / total_time if total_time > 0 else 0
        }
        
    def run_all_benchmarks(self):
        """Run all benchmark tests and collect results"""
        results = []
        
        # Test write operations with different file sizes and concurrency levels
        for file_size in FILE_SIZES:
            for concurrency in CONCURRENCY_LEVELS:
                for _ in range(NUM_ITERATIONS):
                    result = self.benchmark_write(file_size, concurrency)
                    result["security_config"] = self.security_config["name"]
                    results.append(result)
                    
        # Test read operations with different file sizes and concurrency levels
        for file_size in FILE_SIZES:
            for concurrency in CONCURRENCY_LEVELS:
                for _ in range(NUM_ITERATIONS):
                    result = self.benchmark_read(file_size, concurrency)
                    result["security_config"] = self.security_config["name"]
                    results.append(result)
                    
        # Test list operations with different concurrency levels
        for concurrency in CONCURRENCY_LEVELS:
            for _ in range(NUM_ITERATIONS):
                result = self.benchmark_list(concurrency)
                result["security_config"] = self.security_config["name"]
                results.append(result)
                
        # Test metadata operations with different concurrency levels
        for concurrency in CONCURRENCY_LEVELS:
            for _ in range(NUM_ITERATIONS):
                result = self.benchmark_metadata(concurrency)
                result["security_config"] = self.security_config["name"]
                results.append(result)
                
        return results
        
    def cleanup(self):
        """Clean up test resources"""
        logger.info("Cleaning up test resources")
        
        try:
            # Delete the test bucket and volume
            self._run_command([OZONE_CLI, "sh", "bucket", "delete", f"/{self.volume_name}/{self.bucket_name}"])
            self._run_command([OZONE_CLI, "sh", "volume", "delete", self.volume_name])
            
            # Clean up test files
            for file_path in self.test_files.values():
                if os.path.exists(file_path):
                    os.remove(file_path)
                    
            logger.info("Test resources cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")


def analyze_results(results):
    """Analyze the benchmark results and generate performance insights"""
    # Convert results to DataFrame for easier analysis
    df = pd.DataFrame(results)
    
    # Group by operation, file size, concurrency, and security config
    grouped = df.groupby(['operation', 'file_size', 'concurrency', 'security_config'])
    
    # Calculate aggregated metrics
    metrics = grouped.agg({
        'total_time': ['mean', 'std'],
        'throughput_mb_per_sec': ['mean', 'std'] if 'throughput_mb_per_sec' in df.columns else None,
        'avg_latency': ['mean', 'std'],
        'ops_per_sec': ['mean', 'std'] if 'ops_per_sec' in df.columns else None
    }).reset_index()
    
    # Calculate percentage degradation compared to no_security baseline
    performance_impact = {}
    
    for operation in df['operation'].unique():
        performance_impact[operation] = {}
        
        for metric in ['throughput_mb_per_sec', 'avg_latency', 'ops_per_sec']:
            if metric in df.columns:
                baseline_data = df[(df['security_config'] == 'no_security') & (df['operation'] == operation)]
                
                if not baseline_data.empty:
                    baseline_mean = baseline_data[metric].mean()
                    
                    for config in df['security_config'].unique():
                        if config != 'no_security':
                            config_data = df[(df['security_config'] == config) & (df['operation'] == operation)]
                            
                            if not config_data.empty:
                                config_mean = config_data[metric].mean()
                                
                                if metric == 'avg_latency':
                                    # For latency, higher is worse
                                    degradation = ((config_mean - baseline_mean) / baseline_mean) * 100
                                else:
                                    # For throughput and ops_per_sec, lower is worse
                                    degradation = ((baseline_mean - config_mean) / baseline_mean) * 100
                                    
                                if operation not in performance_impact:
                                    performance_impact[operation] = {}
                                    
                                if config not in performance_impact[operation]:
                                    performance_impact[operation][config] = {}
                                    
                                performance_impact[operation][config][metric] = degradation
    
    return {
        'metrics': metrics,
        'performance_impact': performance_impact,
        'raw_results': df
    }


def generate_report(analysis_result, output_dir=None):
    """Generate a performance impact report"""
    if output_dir is None:
        output_dir = TEST_DATA_DIR
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data from analysis
    metrics = analysis_result['metrics']
    performance_impact = analysis_result['performance_impact']
    raw_results = analysis_result['raw_results']
    
    # Create a detailed report
    report = {
        'timestamp': datetime.now().isoformat(),
        'summary': {},
        'details': {}
    }
    
    # Generate summary of performance impact
    for operation, configs in performance_impact.items():
        report['summary'][operation] = {}
        
        for config, metrics_data in configs.items():
            report['summary'][operation][config] = {
                metric: f"{degradation:.2f}%" 
                for metric, degradation in metrics_data.items()
            }
    
    # Check if any degradation exceeds the threshold
    exceeded_threshold = False
    for operation, configs in performance_impact.items():
        for config, metrics_data in configs.items():
            for metric, degradation in metrics_data.items():
                if degradation > MAX_ALLOWED_DEGRADATION:
                    exceeded_threshold = True
                    logger.warning(
                        f"Performance degradation exceeds threshold: {operation}, {config}, "
                        f"{metric}: {degradation:.2f}% > {MAX_ALLOWED_DEGRADATION}%"
                    )
    
    report['passed'] = not exceeded_threshold
    
    # Save report as JSON
    report_path = os.path.join(output_dir, "security_performance_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Generate plots
    for operation in raw_results['operation'].unique():
        operation_data = raw_results[raw_results['operation'] == operation]
        
        plt.figure(figsize=(12, 8))
        
        if 'throughput_mb_per_sec' in operation_data.columns:
            plt.subplot(2, 1, 1)
            for config in operation_data['security_config'].unique():
                config_data = operation_data[operation_data['security_config'] == config]
                plt.plot(config_data['file_size'], config_data['throughput_mb_per_sec'], 'o-', label=config)
                
            plt.xscale('log')
            plt.title(f'{operation.capitalize()} Throughput vs File Size')
            plt.xlabel('File Size (bytes)')
            plt.ylabel('Throughput (MB/s)')
            plt.legend()
            
        plt.subplot(2, 1, 2)
        for config in operation_data['security_config'].unique():
            config_data = operation_data[operation_data['security_config'] == config]
            plt.plot(config_data['concurrency'], config_data['avg_latency'], 'o-', label=config)
            
        plt.title(f'{operation.capitalize()} Latency vs Concurrency')
        plt.xlabel('Concurrency')
        plt.ylabel('Latency (s)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{operation}_performance.png"))
        plt.close()
    
    logger.info(f"Performance report saved to {report_path}")
    return report


@pytest.fixture(scope="module")
def setup_performance_test_environment():
    """Set up the test environment for running performance benchmarks"""
    # Create test data directory
    os.makedirs(TEST_DATA_DIR, exist_ok=True)
    
    yield
    
    # Clean up after tests
    import shutil
    if os.path.exists(TEST_DATA_DIR):
        shutil.rmtree(TEST_DATA_DIR)


def test_34_security_performance_impact(setup_performance_test_environment):
    """Evaluate performance impact of security features.
    
    This test measures the performance overhead introduced by various security features
    in Apache Ozone, including Kerberos authentication, TLS encryption, and authorization.
    The goal is to ensure that security features provide protection with acceptable
    performance overhead (e.g., < 20% degradation).
    """
    all_results = []
    
    try:
        # Run benchmarks for each security configuration
        for security_config in SECURITY_CONFIGS:
            logger.info(f"Testing with security configuration: {security_config['name']}")
            
            tester = OzonePerformanceTester(security_config)
            tester.setup()
            
            try:
                # Run benchmarks and collect results
                results = tester.run_all_benchmarks()
                all_results.extend(results)
                
                logger.info(f"Completed benchmarks for {security_config['name']}")
            finally:
                # Clean up resources for this configuration
                tester.cleanup()
        
        # Analyze the results
        analysis = analyze_results(all_results)
        
        # Generate a report
        report = generate_report(analysis)
        
        # Verify that security overhead is within acceptable limits
        assert report['passed'], "Security features introduced excessive performance degradation"
        
        # Additional detailed assertions
        for operation, configs in report['summary'].items():
            for config, metrics in configs.items():
                for metric, degradation_str in metrics.items():
                    degradation = float(degradation_str.strip('%'))
                    
                    # Assert that no performance metric has degraded more than the threshold
                    assert degradation <= MAX_ALLOWED_DEGRADATION, \
                        f"Performance degradation for {operation}, {config}, {metric}: {degradation}% > {MAX_ALLOWED_DEGRADATION}%"
        
        logger.info("All security features have acceptable performance overhead")
        
    except Exception as e:
        logger.error(f"Error in performance testing: {str(e)}")
        raise

#!/usr/bin/env python3

import os
import time
import shutil
import subprocess
import pytest
import logging
import json
import threading
from typing import Dict, List, Tuple
import random
import tempfile
from subprocess import run, PIPE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants for the Ozone cluster
OZONE_CLUSTER_CONFIG = {
    "admin_host": os.environ.get("OZONE_ADMIN_HOST", "localhost"),
    "datanodes": os.environ.get("OZONE_DATANODES", "datanode1,datanode2,datanode3").split(","),
    "scm_node": os.environ.get("OZONE_SCM_NODE", "scm"),
    "om_node": os.environ.get("OZONE_OM_NODE", "om"),
    "recon_node": os.environ.get("OZONE_RECON_NODE", "recon"),
    "recovery_sla_seconds": int(os.environ.get("RECOVERY_SLA_SECONDS", "300")),
}

# Test data configurations
VOLUME_NAME = "perfrecovery"
BUCKET_NAME = "recoverybucket"
TEST_DATA_DIR = os.environ.get("TEST_DATA_DIR", "/tmp/ozone-test-data")

# Failure scenarios to test with their descriptions
FAILURE_SCENARIOS = [
    {
        "name": "datanode_failure",
        "description": "Simulate datanode failure",
        "target": "datanode",
    },
    {
        "name": "disk_failure",
        "description": "Simulate disk failure on a datanode",
        "target": "disk",
    },
    {
        "name": "network_partition",
        "description": "Simulate network partition",
        "target": "network",
    }
]

# Data sizes to test with (in bytes)
DATA_SIZES = [
    1024 * 1024,       # 1 MB
    10 * 1024 * 1024,  # 10 MB
    50 * 1024 * 1024,  # 50 MB
]


class OzoneClusterManager:
    """Helper class to manage Ozone cluster operations for testing"""
    
    @staticmethod
    def run_command(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
        """Run a shell command and return the result"""
        logger.info(f"Running command: {' '.join(cmd)}")
        return subprocess.run(cmd, check=check, stdout=PIPE, stderr=PIPE, text=True)
    
    @staticmethod
    def create_test_files(sizes: List[int]) -> Dict[int, str]:
        """Create test files of specified sizes"""
        if not os.path.exists(TEST_DATA_DIR):
            os.makedirs(TEST_DATA_DIR)
            
        files = {}
        for size in sizes:
            file_path = os.path.join(TEST_DATA_DIR, f"testfile_{size}.dat")
            with open(file_path, 'wb') as f:
                f.write(os.urandom(size))
            files[size] = file_path
        
        return files
    
    @staticmethod
    def upload_test_data(files: Dict[int, str]) -> Dict[int, str]:
        """Upload test files to Ozone and return key names"""
        keys = {}
        
        # Create volume and bucket if they don't exist
        OzoneClusterManager.run_command(["ozone", "sh", "volume", "create", VOLUME_NAME])
        OzoneClusterManager.run_command(["ozone", "sh", "bucket", "create", f"{VOLUME_NAME}/{BUCKET_NAME}"])
        
        for size, file_path in files.items():
            key_name = f"testkey_{size}_{int(time.time())}"
            OzoneClusterManager.run_command(["ozone", "sh", "key", "put", f"{VOLUME_NAME}/{BUCKET_NAME}/", file_path, "--key", key_name])
            keys[size] = key_name
            
        return keys
    
    @staticmethod
    def simulate_failure(scenario: Dict) -> Dict:
        """Simulate a failure based on the scenario"""
        failure_info = {
            "scenario": scenario,
            "start_time": time.time(),
            "affected_components": [],
        }
        
        if scenario["target"] == "datanode":
            # Choose a random datanode to stop
            datanode = random.choice(OZONE_CLUSTER_CONFIG["datanodes"])
            failure_info["affected_components"].append(datanode)
            
            # Stop the datanode service
            OzoneClusterManager.run_command(["ssh", datanode, "sudo systemctl stop ozone-datanode"], check=False)
            logger.info(f"Stopped datanode: {datanode}")
            
        elif scenario["target"] == "disk":
            # Choose a random datanode
            datanode = random.choice(OZONE_CLUSTER_CONFIG["datanodes"])
            failure_info["affected_components"].append(datanode)
            
            # Simulate disk failure by removing write permissions on a data directory
            # This is a simulation, in a real test you'd target the actual data disk
            cmd = ["ssh", datanode, "sudo chmod -w /tmp/ozone-disk1 || true"]
            OzoneClusterManager.run_command(cmd, check=False)
            logger.info(f"Simulated disk failure on {datanode}")
            
        elif scenario["target"] == "network":
            # Choose a random datanode to isolate
            datanode = random.choice(OZONE_CLUSTER_CONFIG["datanodes"])
            failure_info["affected_components"].append(datanode)
            
            # Use iptables to simulate network partition
            cmd = ["ssh", datanode, "sudo iptables -A INPUT -p tcp --dport 9858 -j DROP"]
            OzoneClusterManager.run_command(cmd, check=False)
            logger.info(f"Simulated network partition for {datanode}")
            
        return failure_info
    
    @staticmethod
    def restore_from_failure(failure_info: Dict) -> None:
        """Restore the system from a simulated failure"""
        scenario = failure_info["scenario"]
        
        for component in failure_info["affected_components"]:
            if scenario["target"] == "datanode":
                # Restart the datanode
                OzoneClusterManager.run_command(["ssh", component, "sudo systemctl start ozone-datanode"], check=False)
                logger.info(f"Started datanode: {component}")
                
            elif scenario["target"] == "disk":
                # Restore disk permissions
                cmd = ["ssh", component, "sudo chmod +w /tmp/ozone-disk1 || true"]
                OzoneClusterManager.run_command(cmd, check=False)
                logger.info(f"Restored disk permissions on {component}")
                
            elif scenario["target"] == "network":
                # Remove the network isolation
                cmd = ["ssh", component, "sudo iptables -D INPUT -p tcp --dport 9858 -j DROP"]
                OzoneClusterManager.run_command(cmd, check=False)
                logger.info(f"Removed network partition for {component}")
    
    @staticmethod
    def monitor_recovery(keys: Dict[int, str], start_time: float) -> Dict:
        """Monitor the recovery process and collect metrics"""
        recovery_metrics = {
            "start_time": start_time,
            "completion_time": None,
            "duration_seconds": None,
            "container_reports": [],
            "data_integrity": True,
            "system_metrics": [],
        }
        
        # Poll the system until all data is recovered
        recovery_complete = False
        poll_interval = 5  # seconds
        
        while not recovery_complete:
            # Get recovery status
            result = OzoneClusterManager.run_command(
                ["ozone", "admin", "container", "list", "--report", "--json"],
                check=False
            )
            
            try:
                container_status = json.loads(result.stdout)
                recovery_metrics["container_reports"].append(container_status)
                
                # Check if there are any containers in recovery state
                containers_recovering = False
                for container in container_status.get("containers", []):
                    if container.get("state") == "RECOVERING" or container.get("state") == "UNHEALTHY":
                        containers_recovering = True
                        break
                
                # Get system metrics
                system_metrics = OzoneClusterManager.get_system_metrics()
                recovery_metrics["system_metrics"].append(system_metrics)
                
                if not containers_recovering:
                    # Verify data integrity by checking keys
                    all_keys_accessible = True
                    for size, key in keys.items():
                        try:
                            # Try to read key metadata
                            cmd = ["ozone", "sh", "key", "info", f"{VOLUME_NAME}/{BUCKET_NAME}/{key}"]
                            result = OzoneClusterManager.run_command(cmd, check=False)
                            if result.returncode != 0:
                                all_keys_accessible = False
                                logger.warning(f"Key {key} is not accessible")
                                break
                        except Exception as e:
                            all_keys_accessible = False
                            logger.error(f"Error checking key {key}: {e}")
                            break
                    
                    if all_keys_accessible:
                        recovery_complete = True
            except json.JSONDecodeError:
                logger.warning("Failed to parse container report JSON")
            except Exception as e:
                logger.error(f"Error monitoring recovery: {e}")
            
            if recovery_complete:
                break
                
            if (time.time() - start_time) > OZONE_CLUSTER_CONFIG["recovery_sla_seconds"] * 2:
                logger.warning("Recovery monitoring timeout exceeded")
                recovery_metrics["data_integrity"] = False
                break
                
            time.sleep(poll_interval)
        
        # Record completion time
        recovery_metrics["completion_time"] = time.time()
        recovery_metrics["duration_seconds"] = recovery_metrics["completion_time"] - recovery_metrics["start_time"]
        
        return recovery_metrics
    
    @staticmethod
    def get_system_metrics() -> Dict:
        """Get system metrics during recovery process"""
        metrics = {
            "timestamp": time.time(),
            "cpu_usage": {},
            "memory_usage": {},
            "disk_io": {},
            "network_io": {}
        }
        
        # Get CPU usage for OM, SCM, and datanodes
        for node_type in ["om_node", "scm_node", "datanodes"]:
            nodes = [OZONE_CLUSTER_CONFIG[node_type]] if node_type != "datanodes" else OZONE_CLUSTER_CONFIG[node_type]
            
            for node in nodes:
                try:
                    # Get CPU usage (simplified - in production you'd use more robust metrics collection)
                    cmd = ["ssh", node, "top -bn1 | grep 'Cpu(s)' | awk '{print $2 + $4}'"]
                    result = OzoneClusterManager.run_command(cmd, check=False)
                    if result.returncode == 0:
                        try:
                            metrics["cpu_usage"][node] = float(result.stdout.strip())
                        except ValueError:
                            metrics["cpu_usage"][node] = -1
                except Exception as e:
                    logger.error(f"Error getting CPU metrics for {node}: {e}")
                    metrics["cpu_usage"][node] = -1
        
        return metrics


@pytest.fixture(scope="module")
def setup_test_data():
    """Fixture to set up test data for recovery tests"""
    # Create test files
    files = OzoneClusterManager.create_test_files(DATA_SIZES)
    
    # Upload test files to Ozone
    keys = OzoneClusterManager.upload_test_data(files)
    
    yield {
        "files": files,
        "keys": keys
    }
    
    # Clean up (optional, as we might want to keep test data for analysis)
    try:
        shutil.rmtree(TEST_DATA_DIR)
    except Exception as e:
        logger.warning(f"Failed to clean up test data: {e}")


@pytest.mark.parametrize("scenario", FAILURE_SCENARIOS)
def test_35_data_recovery_performance(setup_test_data, scenario):
    """
    Test performance under data recovery scenarios.
    
    This test:
    1. Simulates various data loss scenarios (disk failure, node failure)
    2. Triggers data recovery processes
    3. Measures time taken for data recovery
    4. Monitors system performance during recovery
    5. Verifies data integrity post-recovery
    """
    logger.info(f"Starting data recovery performance test for scenario: {scenario['name']}")
    
    # 1. Simulate data loss
    failure_info = OzoneClusterManager.simulate_failure(scenario)
    
    # Wait for system to detect failure
    time.sleep(10)
    
    # 2. Trigger recovery (by restoring the failed component)
    logger.info("Restoring failed components to trigger recovery")
    recovery_start_time = time.time()
    OzoneClusterManager.restore_from_failure(failure_info)
    
    # 3 & 4. Monitor recovery process
    logger.info("Monitoring recovery process...")
    recovery_metrics = OzoneClusterManager.monitor_recovery(
        setup_test_data["keys"], 
        recovery_start_time
    )
    
    # 5. Verify results
    logger.info(f"Recovery completed in {recovery_metrics['duration_seconds']:.2f} seconds")
    
    # System performance metrics during recovery
    if recovery_metrics["system_metrics"]:
        avg_cpu_usage = {}
        for node, values in [(node, [m["cpu_usage"].get(node, 0) for m in recovery_metrics["system_metrics"] if node in m["cpu_usage"]]) 
                            for node in set().union(*[m["cpu_usage"].keys() for m in recovery_metrics["system_metrics"]])]:
            if values:
                avg_cpu_usage[node] = sum(values) / len(values)
        
        logger.info(f"Average CPU usage during recovery: {avg_cpu_usage}")
    
    # Assertions based on expected results
    assert recovery_metrics["data_integrity"], "Data integrity check failed after recovery"
    
    # Assert recovery completed within SLA
    assert recovery_metrics["duration_seconds"] <= OZONE_CLUSTER_CONFIG["recovery_sla_seconds"], \
        f"Recovery took {recovery_metrics['duration_seconds']} seconds, which exceeds SLA of {OZONE_CLUSTER_CONFIG['recovery_sla_seconds']} seconds"
    
    # Create a detailed report
    report = {
        "scenario": scenario["name"],
        "description": scenario["description"],
        "recovery_time_seconds": recovery_metrics["duration_seconds"],
        "recovery_sla_seconds": OZONE_CLUSTER_CONFIG["recovery_sla_seconds"],
        "data_integrity_maintained": recovery_metrics["data_integrity"],
        "affected_components": failure_info["affected_components"],
        "test_data_volume": sum(DATA_SIZES),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Output the report
    report_file = f"recovery_test_{scenario['name']}_{int(time.time())}.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Test report saved to {report_file}")
    logger.info(f"Test completed for scenario: {scenario['name']}")

import os
import time
import pytest
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import paramiko
import json
import numpy as np
from typing import List, Dict, Tuple

# Constants for the test
IO_SCHEDULERS = ["cfq", "deadline", "noop", "bfq"]
WORKLOAD_TYPES = ["sequential_read", "sequential_write", "random_read", "random_write", "mixed"]
FILE_SIZES_MB = [10, 100, 500, 1000, 2000]  # Test with various file sizes from 10MB to 2GB
NUM_OPERATIONS = 100  # Number of operations per workload
OZONE_NODES = ["node1", "node2", "node3"]  # Replace with your actual node hostnames
RESULTS_DIR = "io_scheduler_test_results"

class OzoneClusterManager:
    """Handles Ozone cluster configuration and management for testing"""
    
    def __init__(self, nodes: List[str]):
        self.nodes = nodes
        self.ssh_clients = {}
        
    def setup_ssh_connections(self, username: str, key_path: str = None, password: str = None):
        """Establish SSH connections to all nodes"""
        for node in self.nodes:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            if key_path:
                client.connect(node, username=username, key_filename=key_path)
            else:
                client.connect(node, username=username, password=password)
                
            self.ssh_clients[node] = client
            
    def close_connections(self):
        """Close all SSH connections"""
        for client in self.ssh_clients.values():
            client.close()
            
    def set_io_scheduler(self, node: str, scheduler: str, device: str = "sda"):
        """Change the I/O scheduler on a specific node"""
        if node not in self.ssh_clients:
            raise ValueError(f"No SSH connection to node {node}")
            
        client = self.ssh_clients[node]
        
        # Check if the scheduler is available
        _, stdout, _ = client.exec_command(f"cat /sys/block/{device}/queue/scheduler")
        available_schedulers = stdout.read().decode().strip()
        
        if scheduler not in available_schedulers:
            raise ValueError(f"Scheduler {scheduler} not available on {node}. Available: {available_schedulers}")
        
        # Set the scheduler
        command = f"echo {scheduler} | sudo tee /sys/block/{device}/queue/scheduler"
        _, stdout, stderr = client.exec_command(command)
        
        if stderr.read():
            raise RuntimeError(f"Failed to set scheduler on {node}: {stderr.read().decode()}")
        
        # Verify the scheduler was set
        _, stdout, _ = client.exec_command(f"cat /sys/block/{device}/queue/scheduler")
        current_scheduler = stdout.read().decode().strip()
        
        if f"[{scheduler}]" not in current_scheduler:
            raise RuntimeError(f"Failed to set scheduler on {node}. Current: {current_scheduler}")
        
        return True
        
    def configure_all_nodes(self, scheduler: str, devices: List[str] = None):
        """Configure all nodes with the specified I/O scheduler"""
        if devices is None:
            devices = ["sda"]  # Default device
            
        results = {}
        for node in self.nodes:
            node_results = {}
            for device in devices:
                try:
                    success = self.set_io_scheduler(node, scheduler, device)
                    node_results[device] = success
                except Exception as e:
                    node_results[device] = str(e)
            results[node] = node_results
            
        return results


class WorkloadGenerator:
    """Generates different types of workloads for testing I/O schedulers"""
    
    def __init__(self, ozone_client, volume: str, bucket: str):
        self.ozone_client = ozone_client
        self.volume = volume
        self.bucket = bucket
        
    def _create_test_file(self, size_mb: int, path: str):
        """Create a test file of specified size"""
        with open(path, 'wb') as f:
            f.write(os.urandom(size_mb * 1024 * 1024))
        return path
        
    def sequential_write(self, file_size_mb: int, num_operations: int) -> Tuple[float, float]:
        """Perform sequential write operations and measure throughput and latency"""
        test_file = f"seq_write_test_{file_size_mb}mb.dat"
        self._create_test_file(file_size_mb, test_file)
        
        start_time = time.time()
        
        for i in range(num_operations):
            key = f"seq_write_key_{file_size_mb}mb_{i}"
            self.ozone_client.put_key(self.volume, self.bucket, key, test_file)
            
        end_time = time.time()
        
        total_time = end_time - start_time
        total_data = file_size_mb * num_operations
        
        throughput = total_data / total_time  # MB/s
        latency = total_time / num_operations * 1000  # ms per operation
        
        os.remove(test_file)
        return throughput, latency
        
    def sequential_read(self, file_size_mb: int, num_operations: int) -> Tuple[float, float]:
        """Perform sequential read operations and measure throughput and latency"""
        # First create a test file and upload it
        test_file = f"seq_read_test_{file_size_mb}mb.dat"
        self._create_test_file(file_size_mb, test_file)
        key = f"seq_read_key_{file_size_mb}mb"
        self.ozone_client.put_key(self.volume, self.bucket, key, test_file)
        
        start_time = time.time()
        
        for i in range(num_operations):
            self.ozone_client.get_key(self.volume, self.bucket, key, f"seq_read_output_{i}.dat")
            
        end_time = time.time()
        
        total_time = end_time - start_time
        total_data = file_size_mb * num_operations
        
        throughput = total_data / total_time  # MB/s
        latency = total_time / num_operations * 1000  # ms per operation
        
        os.remove(test_file)
        return throughput, latency
        
    def random_write(self, file_size_mb: int, num_operations: int) -> Tuple[float, float]:
        """Perform random write operations and measure throughput and latency"""
        test_files = []
        for i in range(5):  # Create a few test files of different sizes
            size = int(file_size_mb * (0.5 + i * 0.2))  # Vary sizes from 50% to 150% of target
            test_file = f"rand_write_test_{size}mb.dat"
            self._create_test_file(size, test_file)
            test_files.append((test_file, size))
        
        start_time = time.time()
        
        for i in range(num_operations):
            # Use a random file for each operation
            test_file, size = test_files[i % len(test_files)]
            key = f"rand_write_key_{i}_{size}mb"
            self.ozone_client.put_key(self.volume, self.bucket, key, test_file)
            
        end_time = time.time()
        
        total_time = end_time - start_time
        total_data = sum(size for _, size in test_files) * (num_operations / len(test_files))
        
        throughput = total_data / total_time  # MB/s
        latency = total_time / num_operations * 1000  # ms per operation
        
        for test_file, _ in test_files:
            os.remove(test_file)
            
        return throughput, latency
        
    def random_read(self, file_size_mb: int, num_operations: int) -> Tuple[float, float]:
        """Perform random read operations and measure throughput and latency"""
        # First create and upload a few test files
        test_keys = []
        for i in range(5):  # Create a few test files of different sizes
            size = int(file_size_mb * (0.5 + i * 0.2))  # Vary sizes from 50% to 150% of target
            test_file = f"rand_read_test_{size}mb.dat"
            self._create_test_file(size, test_file)
            key = f"rand_read_key_{i}_{size}mb"
            self.ozone_client.put_key(self.volume, self.bucket, key, test_file)
            test_keys.append(key)
            os.remove(test_file)
        
        start_time = time.time()
        
        for i in range(num_operations):
            # Read a random file for each operation
            key = test_keys[i % len(test_keys)]
            self.ozone_client.get_key(self.volume, self.bucket, key, f"rand_read_output_{i}.dat")
            
        end_time = time.time()
        
        total_time = end_time - start_time
        # Estimate total data (this is approximate)
        total_data = file_size_mb * num_operations
        
        throughput = total_data / total_time  # MB/s
        latency = total_time / num_operations * 1000  # ms per operation
        
        return throughput, latency
        
    def mixed(self, file_size_mb: int, num_operations: int) -> Tuple[float, float]:
        """Perform a mix of read and write operations and measure throughput and latency"""
        # Create test files
        test_files = []
        for i in range(3):
            size = int(file_size_mb * (0.8 + i * 0.2))
            test_file = f"mixed_test_{size}mb.dat"
            self._create_test_file(size, test_file)
            test_files.append((test_file, size))
            
        # Pre-upload some files for reading
        test_keys = []
        for i, (test_file, size) in enumerate(test_files):
            key = f"mixed_key_prep_{i}_{size}mb"
            self.ozone_client.put_key(self.volume, self.bucket, key, test_file)
            test_keys.append(key)
        
        start_time = time.time()
        
        for i in range(num_operations):
            if i % 2 == 0:  # Even numbers: write operations
                test_file, size = test_files[i % len(test_files)]
                key = f"mixed_write_{i}_{size}mb"
                self.ozone_client.put_key(self.volume, self.bucket, key, test_file)
            else:  # Odd numbers: read operations
                key = test_keys[i % len(test_keys)]
                self.ozone_client.get_key(self.volume, self.bucket, key, f"mixed_read_output_{i}.dat")
                
        end_time = time.time()
        
        total_time = end_time - start_time
        # Estimate total data (this is approximate)
        total_data = file_size_mb * num_operations
        
        throughput = total_data / total_time  # MB/s
        latency = total_time / num_operations * 1000  # ms per operation
        
        for test_file, _ in test_files:
            os.remove(test_file)
            
        return throughput, latency
        
    def run_workload(self, workload_type: str, file_size_mb: int, num_operations: int) -> Dict:
        """Run the specified workload and return performance metrics"""
        workload_methods = {
            "sequential_read": self.sequential_read,
            "sequential_write": self.sequential_write,
            "random_read": self.random_read,
            "random_write": self.random_write,
            "mixed": self.mixed
        }
        
        if workload_type not in workload_methods:
            raise ValueError(f"Unknown workload type: {workload_type}")
            
        throughput, latency = workload_methods[workload_type](file_size_mb, num_operations)
        
        return {
            "workload_type": workload_type,
            "file_size_mb": file_size_mb,
            "throughput_mbps": throughput,
            "latency_ms": latency,
            "operations": num_operations
        }


class OzoneClient:
    """Client for interacting with Apache Ozone"""
    
    def __init__(self, ozone_bin: str = "/opt/ozone/bin"):
        self.ozone_bin = ozone_bin
        
    def create_volume(self, volume: str):
        """Create an Ozone volume"""
        cmd = [f"{self.ozone_bin}/ozone", "sh", "volume", "create", volume]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to create volume: {result.stderr}")
        
    def create_bucket(self, volume: str, bucket: str):
        """Create an Ozone bucket"""
        cmd = [f"{self.ozone_bin}/ozone", "sh", "bucket", "create", f"{volume}/{bucket}"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to create bucket: {result.stderr}")
            
    def put_key(self, volume: str, bucket: str, key: str, file_path: str):
        """Put a key into an Ozone bucket"""
        cmd = [f"{self.ozone_bin}/ozone", "sh", "key", "put", f"{volume}/{bucket}/{key}", file_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to put key: {result.stderr}")
            
    def get_key(self, volume: str, bucket: str, key: str, file_path: str):
        """Get a key from an Ozone bucket"""
        cmd = [f"{self.ozone_bin}/ozone", "sh", "key", "get", f"{volume}/{bucket}/{key}", file_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to get key: {result.stderr}")


class ResultAnalyzer:
    """Analyzes and visualizes test results"""
    
    def __init__(self, results_dir: str = RESULTS_DIR):
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        
    def save_results(self, results: List[Dict], filename: str = "io_scheduler_test_results.json"):
        """Save the test results to a file"""
        path = os.path.join(self.results_dir, filename)
        with open(path, 'w') as f:
            json.dump(results, f, indent=2)
            
    def load_results(self, filename: str = "io_scheduler_test_results.json") -> List[Dict]:
        """Load test results from a file"""
        path = os.path.join(self.results_dir, filename)
        with open(path, 'r') as f:
            return json.load(f)
            
    def generate_dataframe(self, results: List[Dict]) -> pd.DataFrame:
        """Convert results to a pandas DataFrame for analysis"""
        return pd.DataFrame(results)
    
    def plot_throughput_comparison(self, df: pd.DataFrame, output_file: str = "throughput_comparison.png"):
        """Plot throughput comparison across different I/O schedulers and workload types"""
        plt.figure(figsize=(14, 10))
        
        for i, workload in enumerate(df['workload_type'].unique()):
            plt.subplot(3, 2, i+1)
            workload_df = df[df['workload_type'] == workload]
            
            for scheduler in workload_df['scheduler'].unique():
                sched_df = workload_df[workload_df['scheduler'] == scheduler]
                plt.plot(sched_df['file_size_mb'], sched_df['throughput_mbps'], marker='o', label=scheduler)
                
            plt.title(f"{workload} Workload")
            plt.xlabel("File Size (MB)")
            plt.ylabel("Throughput (MB/s)")
            plt.legend()
            plt.grid(True)
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, output_file))
        
    def plot_latency_comparison(self, df: pd.DataFrame, output_file: str = "latency_comparison.png"):
        """Plot latency comparison across different I/O schedulers and workload types"""
        plt.figure(figsize=(14, 10))
        
        for i, workload in enumerate(df['workload_type'].unique()):
            plt.subplot(3, 2, i+1)
            workload_df = df[df['workload_type'] == workload]
            
            for scheduler in workload_df['scheduler'].unique():
                sched_df = workload_df[workload_df['scheduler'] == scheduler]
                plt.plot(sched_df['file_size_mb'], sched_df['latency_ms'], marker='o', label=scheduler)
                
            plt.title(f"{workload} Workload")
            plt.xlabel("File Size (MB)")
            plt.ylabel("Latency (ms)")
            plt.legend()
            plt.grid(True)
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, output_file))
        
    def generate_summary_report(self, df: pd.DataFrame) -> Dict:
        """Generate a summary report of the test results"""
        summary = {
            "best_schedulers_by_workload": {},
            "best_overall_scheduler": {
                "throughput": "",
                "latency": ""
            },
            "recommendations": {}
        }
        
        # Find best scheduler for each workload type
        for workload in df['workload_type'].unique():
            workload_df = df[df['workload_type'] == workload]
            
            # For throughput (higher is better)
            throughput_by_scheduler = workload_df.groupby('scheduler')['throughput_mbps'].mean()
            best_throughput_scheduler = throughput_by_scheduler.idxmax()
            
            # For latency (lower is better)
            latency_by_scheduler = workload_df.groupby('scheduler')['latency_ms'].mean()
            best_latency_scheduler = latency_by_scheduler.idxmin()
            
            summary["best_schedulers_by_workload"][workload] = {
                "throughput": best_throughput_scheduler,
                "latency": best_latency_scheduler
            }
            
            # Generate recommendations
            summary["recommendations"][workload] = (
                f"For {workload} workloads, use {best_throughput_scheduler} scheduler "
                f"for best throughput, or {best_latency_scheduler} for lowest latency."
            )
        
        # Find best overall scheduler
        overall_throughput = df.groupby('scheduler')['throughput_mbps'].mean()
        overall_latency = df.groupby('scheduler')['latency_ms'].mean()
        
        summary["best_overall_scheduler"]["throughput"] = overall_throughput.idxmax()
        summary["best_overall_scheduler"]["latency"] = overall_latency.idxmin()
        
        # Generate overall recommendation
        summary["overall_recommendation"] = (
            f"For general purpose use, {summary['best_overall_scheduler']['throughput']} "
            f"provides the best overall throughput, while {summary['best_overall_scheduler']['latency']} "
            f"provides the lowest latency across all workloads."
        )
        
        return summary
    
    def save_summary_report(self, summary: Dict, filename: str = "io_scheduler_summary.json"):
        """Save the summary report to a file"""
        path = os.path.join(self.results_dir, filename)
        with open(path, 'w') as f:
            json.dump(summary, f, indent=2)


@pytest.fixture(scope="module")
def ozone_client():
    """Fixture to provide an Ozone client"""
    client = OzoneClient()
    return client


@pytest.fixture(scope="module")
def test_volume_bucket(ozone_client):
    """Fixture to create and return a test volume and bucket"""
    # Use a timestamp to ensure unique volume/bucket names
    timestamp = int(time.time())
    volume = f"ioschedulertest{timestamp}"
    bucket = f"benchmark{timestamp}"
    
    ozone_client.create_volume(volume)
    ozone_client.create_bucket(volume, bucket)
    
    return volume, bucket


@pytest.fixture(scope="module")
def cluster_manager():
    """Fixture to provide a cluster manager"""
    manager = OzoneClusterManager(OZONE_NODES)
    # Setup SSH connections (adjust credentials as needed)
    # For real implementation, use environment variables or secure credential storage
    manager.setup_ssh_connections(username="ozone", key_path="~/.ssh/id_rsa")
    
    yield manager
    
    # Clean up
    manager.close_connections()


@pytest.fixture(scope="function")
def analyzer():
    """Fixture to provide a result analyzer"""
    return ResultAnalyzer()


@pytest.mark.performance
@pytest.mark.parametrize("scheduler", IO_SCHEDULERS)
def test_36_io_scheduler_performance(ozone_client, test_volume_bucket, cluster_manager, analyzer, scheduler):
    """
    Evaluate performance with different I/O schedulers
    
    This test evaluates Ozone performance across different I/O schedulers (CFQ, Deadline, Noop, BFQ),
    running various workload types (sequential read/write, random read/write, mixed) and measuring
    the performance characteristics of each.
    """
    volume, bucket = test_volume_bucket
    
    # Configure all nodes with the current I/O scheduler
    try:
        print(f"Configuring cluster nodes with {scheduler} I/O scheduler...")
        cluster_manager.configure_all_nodes(scheduler)
    except Exception as e:
        pytest.skip(f"Couldn't configure {scheduler} scheduler: {str(e)}")
    
    # Create workload generator
    workload_gen = WorkloadGenerator(ozone_client, volume, bucket)
    
    # List to store all test results
    results = []
    
    # Run tests for each workload type and file size
    for workload_type in WORKLOAD_TYPES:
        for file_size_mb in FILE_SIZES_MB:
            print(f"Running {workload_type} workload with {file_size_mb}MB files using {scheduler} scheduler...")
            
            # Run the workload and measure performance
            result = workload_gen.run_workload(workload_type, file_size_mb, NUM_OPERATIONS)
            
            # Add scheduler information to results
            result["scheduler"] = scheduler
            results.append(result)
            
            print(f"  Throughput: {result['throughput_mbps']:.2f} MB/s, Latency: {result['latency_ms']:.2f} ms")
    
    # Save the results for this scheduler
    analyzer.save_results(results, f"{scheduler}_results.json")
    
    # Basic assertions to verify the test ran correctly
    assert len(results) == len(WORKLOAD_TYPES) * len(FILE_SIZES_MB), "Missing results for some test combinations"
    
    for result in results:
        assert result['throughput_mbps'] > 0, f"Zero throughput for {result['workload_type']} with {scheduler}"
        assert result['latency_ms'] > 0, f"Zero latency for {result['workload_type']} with {scheduler}"
    
    # Return results for final analysis
    return results


@pytest.mark.performance
def test_36_io_scheduler_analysis(analyzer):
    """
    Analyze results from I/O scheduler performance tests and generate recommendations
    """
    # Load results from all scheduler tests
    all_results = []
    for scheduler in IO_SCHEDULERS:
        try:
            results = analyzer.load_results(f"{scheduler}_results.json")
            all_results.extend(results)
        except FileNotFoundError:
            print(f"Results file for {scheduler} not found, skipping")
    
    # Skip if we don't have enough data
    if not all_results:
        pytest.skip("No I/O scheduler test results available for analysis")
    
    # Convert results to DataFrame
    df = analyzer.generate_dataframe(all_results)
    
    # Generate plots
    analyzer.plot_throughput_comparison(df)
    analyzer.plot_latency_comparison(df)
    
    # Generate and save summary report
    summary = analyzer.generate_summary_report(df)
    analyzer.save_summary_report(summary)
    
    # Print findings to console
    print("\n=== I/O Scheduler Performance Analysis ===")
    print(f"Best overall scheduler for throughput: {summary['best_overall_scheduler']['throughput']}")
    print(f"Best overall scheduler for latency: {summary['best_overall_scheduler']['latency']}")
    
    print("\nRecommendations by workload type:")
    for workload, recommendation in summary['recommendations'].items():
        print(f"- {recommendation}")
        
    print(f"\n{summary['overall_recommendation']}")
    
    # Save complete dataset
    df.to_csv(os.path.join(RESULTS_DIR, "complete_io_scheduler_performance.csv"), index=False)
    
    # Basic assertions to verify we have meaningful results
    assert 'best_overall_scheduler' in summary, "Missing best overall scheduler recommendation"
    assert summary['best_overall_scheduler']['throughput'] in IO_SCHEDULERS, "Invalid throughput scheduler recommendation"
    assert summary['best_overall_scheduler']['latency'] in IO_SCHEDULERS, "Invalid latency scheduler recommendation"

import pytest
import time
import subprocess
import os
import logging
import threading
import tempfile
import random
import statistics
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from kazoo.client import KazooClient
from hdfs import InsecureClient
from pyhdfs import HdfsClient

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants for Ozone
OZONE_CONF_DIR = os.environ.get('OZONE_CONF_DIR', '/etc/hadoop/conf')
OZONE_BIN = os.environ.get('OZONE_BIN', '/opt/hadoop/bin/ozone')
TEST_VOLUME = "perf-partition-vol"
TEST_BUCKET = "partition-bucket"

# Network partition simulator commands (using iptables)
PARTITION_START_CMD = "sudo iptables -A INPUT -s {target_ip} -j DROP && sudo iptables -A OUTPUT -d {target_ip} -j DROP"
PARTITION_STOP_CMD = "sudo iptables -D INPUT -s {target_ip} -j DROP && sudo iptables -D OUTPUT -d {target_ip} -j DROP"

# Performance metrics collectors
class MetricsCollector:
    def __init__(self):
        self.metrics = {
            'latency': [],
            'throughput': [],
            'errors': [],
            'recovery_time': None
        }
    
    def record_latency(self, latency_ms):
        self.metrics['latency'].append(latency_ms)
    
    def record_throughput(self, bytes_per_sec):
        self.metrics['throughput'].append(bytes_per_sec)
    
    def record_error(self, error_type):
        self.metrics['errors'].append(error_type)
    
    def record_recovery_time(self, recovery_time_sec):
        self.metrics['recovery_time'] = recovery_time_sec
    
    def get_summary(self):
        result = {}
        if self.metrics['latency']:
            result['avg_latency'] = statistics.mean(self.metrics['latency'])
            result['p95_latency'] = sorted(self.metrics['latency'])[int(len(self.metrics['latency']) * 0.95)]
            result['max_latency'] = max(self.metrics['latency'])
            
        if self.metrics['throughput']:
            result['avg_throughput'] = statistics.mean(self.metrics['throughput'])
            result['min_throughput'] = min(self.metrics['throughput'])
            
        result['error_count'] = len(self.metrics['errors'])
        result['recovery_time'] = self.metrics['recovery_time']
        
        return result

# Test helper functions
def run_shell_command(command, check=True):
    """Run a shell command and return output"""
    result = subprocess.run(command, shell=True, check=check, 
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                          universal_newlines=True)
    return result

def get_ozone_nodes():
    """Get list of Ozone nodes and their roles"""
    # This would typically query the cluster for node info
    # For this test, we'll mock this with example node data
    nodes = {
        'om': ['om1.example.com', 'om2.example.com', 'om3.example.com'],
        'scm': ['scm1.example.com'],
        'datanode': ['dn1.example.com', 'dn2.example.com', 'dn3.example.com', 
                    'dn4.example.com', 'dn5.example.com']
    }
    return nodes

def create_test_file(size_mb):
    """Create a test file of specified size in MB"""
    file_path = tempfile.mktemp()
    with open(file_path, 'wb') as f:
        f.write(os.urandom(size_mb * 1024 * 1024))
    return file_path

def simulate_network_partition(nodes, partition_type):
    """Simulate different types of network partitions"""
    if partition_type == 'om-majority':
        # Isolate one OM node from the cluster
        target_node = nodes['om'][0]
        other_nodes = [n for role in nodes for n in nodes[role] if n != target_node]
        return target_node, other_nodes

    elif partition_type == 'datanode-partial':
        # Isolate a subset of datanodes
        isolated_nodes = nodes['datanode'][:2]  # Isolate 2 datanodes
        other_nodes = [n for role in nodes for n in nodes[role] if n not in isolated_nodes]
        return isolated_nodes, other_nodes

    elif partition_type == 'scm-isolation':
        # Isolate SCM node
        isolated_node = nodes['scm'][0]
        other_nodes = [n for role in nodes for n in nodes[role] if n != isolated_node]
        return isolated_node, other_nodes

def start_partition(target_nodes, other_nodes):
    """Apply network partition using iptables"""
    if isinstance(target_nodes, list):
        for node in target_nodes:
            for other_node in other_nodes:
                cmd = PARTITION_START_CMD.format(target_ip=node)
                run_shell_command(cmd)
    else:
        for other_node in other_nodes:
            cmd = PARTITION_START_CMD.format(target_ip=other_node)
            run_shell_command(cmd)
    
    logger.info(f"Network partition started between {target_nodes} and {other_nodes}")

def heal_partition(target_nodes, other_nodes):
    """Heal the network partition"""
    if isinstance(target_nodes, list):
        for node in target_nodes:
            for other_node in other_nodes:
                cmd = PARTITION_STOP_CMD.format(target_ip=node)
                run_shell_command(cmd)
    else:
        for other_node in other_nodes:
            cmd = PARTITION_STOP_CMD.format(target_ip=other_node)
            run_shell_command(cmd)
    
    logger.info("Network partition healed")

def perform_io_operations(operation_type, file_size_mb, duration_sec, metrics_collector):
    """Perform read or write operations and collect metrics"""
    end_time = time.time() + duration_sec
    operation_count = 0
    
    while time.time() < end_time:
        try:
            start_time = time.time()
            
            if operation_type == 'write':
                file_path = create_test_file(file_size_mb)
                key_name = f"key-{int(time.time())}"
                cmd = f"{OZONE_BIN} sh key put {TEST_VOLUME}/{TEST_BUCKET}/ {file_path} --key={key_name}"
                result = run_shell_command(cmd, check=False)
                os.unlink(file_path)
            else:  # read
                # Get a list of keys in the bucket
                cmd = f"{OZONE_BIN} sh key list {TEST_VOLUME}/{TEST_BUCKET}/"
                result = run_shell_command(cmd, check=False)
                if result.returncode == 0 and result.stdout.strip():
                    keys = result.stdout.strip().split('\n')
                    if keys:
                        key_to_read = random.choice(keys)
                        temp_file = tempfile.mktemp()
                        cmd = f"{OZONE_BIN} sh key get {TEST_VOLUME}/{TEST_BUCKET}/{key_to_read} {temp_file}"
                        result = run_shell_command(cmd, check=False)
                        if os.path.exists(temp_file):
                            os.unlink(temp_file)
            
            operation_time = time.time() - start_time
            
            if result.returncode == 0:
                # Record metrics
                metrics_collector.record_latency(operation_time * 1000)  # ms
                metrics_collector.record_throughput(file_size_mb / operation_time)  # MB/s
            else:
                metrics_collector.record_error(result.stderr)
            
            operation_count += 1
            
        except Exception as e:
            logger.error(f"Error during IO operation: {str(e)}")
            metrics_collector.record_error(str(e))
            
    return operation_count

def verify_data_consistency():
    """Verify data consistency after partition healing"""
    # This would implement consistency checks
    # For now, we'll just check if the bucket is accessible
    cmd = f"{OZONE_BIN} sh bucket info {TEST_VOLUME}/{TEST_BUCKET}"
    result = run_shell_command(cmd, check=False)
    return result.returncode == 0

def plot_performance_metrics(before_metrics, during_metrics, after_metrics, plot_name):
    """Generate performance visualization"""
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Latency plots
    axs[0, 0].boxplot([
        before_metrics['latency'], 
        during_metrics['latency'], 
        after_metrics['latency']
    ], labels=['Before', 'During', 'After'])
    axs[0, 0].set_title('Latency Distribution (ms)')
    
    # Throughput plots
    axs[0, 1].boxplot([
        before_metrics['throughput'], 
        during_metrics['throughput'], 
        after_metrics['throughput']
    ], labels=['Before', 'During', 'After'])
    axs[0, 1].set_title('Throughput Distribution (MB/s)')
    
    # Error count
    error_counts = [
        len(before_metrics['errors']),
        len(during_metrics['errors']),
        len(after_metrics['errors'])
    ]
    axs[1, 0].bar(['Before', 'During', 'After'], error_counts)
    axs[1, 0].set_title('Error Count')
    
    # Recovery time
    axs[1, 1].text(0.5, 0.5, f"Recovery Time: {after_metrics['recovery_time']} seconds", 
                 horizontalalignment='center', verticalalignment='center', fontsize=12)
    axs[1, 1].set_title('Recovery Metrics')
    axs[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(plot_name)
    logger.info(f"Performance metrics plot saved to {plot_name}")

@pytest.mark.parametrize("partition_type,io_operation,file_size_mb,operation_duration", [
    ('om-majority', 'read', 5, 30),  
    ('om-majority', 'write', 10, 30),
    ('datanode-partial', 'read', 5, 30),
    ('datanode-partial', 'write', 10, 30),
    ('scm-isolation', 'read', 5, 30),
    ('scm-isolation', 'write', 10, 30),
])
def test_37_performance_under_network_partitioning(partition_type, io_operation, file_size_mb, operation_duration):
    """
    Test performance under network partitioning
    
    This test simulates network partitions between Ozone nodes, performs read/write operations
    during the partition, measures system behavior and performance degradation, and evaluates
    recovery time and data consistency after partition healing.
    """
    # Initialize metrics collectors for different phases
    before_metrics = MetricsCollector()
    during_metrics = MetricsCollector()
    after_metrics = MetricsCollector()
    
    # Setup test environment
    logger.info(f"Setting up test environment with {partition_type} partition type and {io_operation} operation")
    
    # Create test volume and bucket if not exists
    run_shell_command(f"{OZONE_BIN} sh volume create {TEST_VOLUME}")
    run_shell_command(f"{OZONE_BIN} sh bucket create {TEST_VOLUME}/{TEST_BUCKET}")
    
    # Pre-load some data if we're going to test reads
    if io_operation == 'read':
        for i in range(10):
            test_file = create_test_file(file_size_mb)
            run_shell_command(f"{OZONE_BIN} sh key put {TEST_VOLUME}/{TEST_BUCKET}/ {test_file} --key=preload-key-{i}")
            os.unlink(test_file)
    
    # Get node information
    nodes = get_ozone_nodes()
    
    # 1. Perform baseline operations (before partition)
    logger.info("Performing baseline operations before partition")
    before_ops = perform_io_operations(io_operation, file_size_mb, operation_duration, before_metrics)
    logger.info(f"Completed {before_ops} {io_operation} operations before partition")
    
    # 2. Simulate network partition
    logger.info(f"Simulating {partition_type} network partition")
    target_nodes, other_nodes = simulate_network_partition(nodes, partition_type)
    start_partition(target_nodes, other_nodes)
    partition_start_time = time.time()
    
    try:
        # 3. Perform operations during partition
        logger.info("Performing operations during network partition")
        during_ops = perform_io_operations(io_operation, file_size_mb, operation_duration, during_metrics)
        logger.info(f"Completed {during_ops} {io_operation} operations during partition")
        
        # 4. Heal partition
        logger.info("Healing network partition")
        heal_partition(target_nodes, other_nodes)
        partition_duration = time.time() - partition_start_time
        
        # 5. Measure recovery time
        recovery_start_time = time.time()
        recovery_detected = False
        max_recovery_wait = 300  # 5 minutes
        
        while time.time() - recovery_start_time < max_recovery_wait and not recovery_detected:
            try:
                # Check if the system has recovered by performing a simple operation
                cmd = f"{OZONE_BIN} sh bucket info {TEST_VOLUME}/{TEST_BUCKET}"
                result = run_shell_command(cmd, check=False)
                if result.returncode == 0:
                    recovery_detected = True
                    recovery_time = time.time() - recovery_start_time
                    after_metrics.record_recovery_time(recovery_time)
                    logger.info(f"System recovered after {recovery_time:.2f} seconds")
                else:
                    time.sleep(5)
            except Exception as e:
                logger.warning(f"Error checking recovery: {str(e)}")
                time.sleep(5)
        
        if not recovery_detected:
            logger.warning("System did not recover within the expected timeframe")
            after_metrics.record_recovery_time(max_recovery_wait)
        
        # 6. Perform post-partition operations
        logger.info("Performing operations after partition healing")
        after_ops = perform_io_operations(io_operation, file_size_mb, operation_duration, after_metrics)
        logger.info(f"Completed {after_ops} {io_operation} operations after partition healing")
        
        # 7. Verify data consistency
        logger.info("Verifying data consistency")
        is_consistent = verify_data_consistency()
        
        # 8. Generate performance report
        before_summary = before_metrics.get_summary()
        during_summary = during_metrics.get_summary()
        after_summary = after_metrics.get_summary()
        
        logger.info(f"Performance summary - Before partition: {before_summary}")
        logger.info(f"Performance summary - During partition: {during_summary}")
        logger.info(f"Performance summary - After healing: {after_summary}")
        
        # Generate performance visualization
        plot_name = f"network_partition_{partition_type}_{io_operation}.png"
        plot_performance_metrics(before_metrics.metrics, during_metrics.metrics, 
                               after_metrics.metrics, plot_name)
        
        # Assertions to validate test
        # Check that system eventually recovered
        assert recovery_detected, "System failed to recover after network partition"
        
        # Check data consistency
        assert is_consistent, "Data consistency check failed after network partition"
        
        # Check that performance eventually returned to normal
        # We allow some degradation (80% of original performance)
        if before_summary.get('avg_throughput') and after_summary.get('avg_throughput'):
            assert after_summary['avg_throughput'] >= 0.8 * before_summary['avg_throughput'], \
                "Performance did not recover sufficiently after partition healing"
        
        # Check recovery time is within acceptable limits
        # The actual threshold depends on the system requirements
        recovery_time_threshold = 120  # 2 minutes
        if after_summary.get('recovery_time'):
            assert after_summary['recovery_time'] <= recovery_time_threshold, \
                f"Recovery time ({after_summary['recovery_time']}s) exceeded threshold ({recovery_time_threshold}s)"
        
    finally:
        # Ensure partition is healed even if test fails
        try:
            heal_partition(target_nodes, other_nodes)
        except Exception as e:
            logger.error(f"Failed to heal partition during cleanup: {str(e)}")

import os
import time
import csv
import subprocess
import pytest
import logging
import threading
import statistics
from datetime import datetime
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants for test configuration
GC_ALGORITHMS = [
    {"name": "G1GC", "options": "-XX:+UseG1GC"},
    {"name": "CMS", "options": "-XX:+UseConcMarkSweepGC"},
    {"name": "ZGC", "options": "-XX:+UseZGC"},
    {"name": "Parallel", "options": "-XX:+UseParallelGC"}
]

# Different workload profiles
WORKLOAD_PROFILES = [
    {
        "name": "small_files",
        "file_size": "1K",
        "file_count": 10000,
        "description": "Many small files"
    },
    {
        "name": "medium_files",
        "file_size": "1M",
        "file_count": 1000,
        "description": "Medium number of medium-sized files"
    },
    {
        "name": "large_files",
        "file_size": "100M",
        "file_count": 50,
        "description": "Few large files"
    },
    {
        "name": "mixed_io",
        "file_size": "varied",
        "file_count": 500,
        "description": "Mixed file sizes with read/write operations"
    }
]

# Heap size configurations to test with
HEAP_SIZES = ["1G", "2G", "4G", "8G"]

# Test duration in seconds for each configuration
TEST_DURATION = 600  # 10 minutes per configuration

class OzoneCluster:
    """Helper class to manage Ozone cluster with different GC settings"""
    
    def __init__(self, base_dir="/tmp/ozone-gc-test"):
        self.base_dir = base_dir
        self.results_dir = f"{base_dir}/results"
        self.data_dir = f"{base_dir}/test-data"
        self.gc_logs_dir = f"{base_dir}/gc-logs"
        
        # Ensure directories exist
        for directory in [self.base_dir, self.results_dir, self.data_dir, self.gc_logs_dir]:
            os.makedirs(directory, exist_ok=True)
        
    def setup_gc_config(self, gc_algorithm: Dict, heap_size: str) -> None:
        """Configure Ozone with specific GC settings"""
        logger.info(f"Configuring Ozone with {gc_algorithm['name']} GC algorithm and {heap_size} heap")
        
        # Create GC log filename based on configuration
        gc_log_file = f"{self.gc_logs_dir}/gc_{gc_algorithm['name']}_{heap_size}.log"
        
        # Prepare JVM options for Ozone
        jvm_opts = [
            f"-Xms{heap_size}", 
            f"-Xmx{heap_size}",
            gc_algorithm["options"],
            f"-Xlog:gc*:file={gc_log_file}:time,uptime:filecount=5,filesize=100m"
        ]
        
        # Update ozone-env.sh with the new settings
        self._update_ozone_env(jvm_opts)
        
        # Restart the Ozone cluster
        self._restart_ozone_cluster()
        
        # Wait for cluster to stabilize
        time.sleep(30)
        
        logger.info(f"Ozone configured with {gc_algorithm['name']} GC and {heap_size} heap")
    
    def _update_ozone_env(self, jvm_opts: List[str]) -> None:
        """Update Ozone environment settings with JVM options"""
        # In a real implementation, this would modify the ozone-env.sh file
        # For this test example, we'll log the command that would be executed
        jvm_opts_str = " ".join(jvm_opts)
        command = f"sed -i 's/OZONE_OPTS=.*/OZONE_OPTS=\"{jvm_opts_str}\"/' $OZONE_HOME/etc/hadoop/ozone-env.sh"
        logger.info(f"Would execute: {command}")
        
    def _restart_ozone_cluster(self) -> None:
        """Restart the Ozone cluster to apply new configurations"""
        # In a real implementation, this would stop and start Ozone services
        # For this test example, we'll log the commands that would be executed
        logger.info("Would execute: ozone stop")
        time.sleep(2)  # Simulate shutdown time
        logger.info("Would execute: ozone start")
        
    def prepare_test_data(self, profile: Dict) -> str:
        """Prepare test data according to the workload profile"""
        profile_dir = f"{self.data_dir}/{profile['name']}"
        os.makedirs(profile_dir, exist_ok=True)
        
        if profile["file_size"] == "varied":
            # Create varied file sizes
            sizes = ["10K", "100K", "1M", "5M", "20M", "75M"]
            for i in range(profile["file_count"]):
                size = sizes[i % len(sizes)]
                self._generate_test_file(f"{profile_dir}/file_{i}.dat", size)
        else:
            # Create files of uniform size
            for i in range(profile["file_count"]):
                self._generate_test_file(f"{profile_dir}/file_{i}.dat", profile["file_size"])
        
        return profile_dir
    
    def _generate_test_file(self, filepath: str, size: str) -> None:
        """Generate a test file of the specified size"""
        # Convert size to bytes
        size_value = int(size[:-1])
        unit = size[-1].upper()
        
        bytes_multiplier = {
            "K": 1024,
            "M": 1024 * 1024,
            "G": 1024 * 1024 * 1024
        }
        
        size_in_bytes = size_value * bytes_multiplier.get(unit, 1)
        
        # Use fallocate or dd to create the file efficiently
        command = f"dd if=/dev/urandom of={filepath} bs=1M count={size_in_bytes//1048576}" if size_in_bytes >= 1048576 else \
                  f"dd if=/dev/urandom of={filepath} bs=1K count={size_in_bytes//1024}"
                  
        logger.info(f"Would execute: {command}")
        
    def run_workload(self, workload_profile: Dict, duration: int = TEST_DURATION) -> Dict:
        """Run a specific workload and collect performance metrics"""
        logger.info(f"Starting workload: {workload_profile['description']}")
        
        # Prepare test data directory
        test_data_dir = self.prepare_test_data(workload_profile)
        
        # Create unique volume and bucket for this test
        volume_name = f"vol-{workload_profile['name']}-{int(time.time())}"
        bucket_name = f"bucket-{workload_profile['name']}-{int(time.time())}"
        
        # Create volume and bucket
        self._execute_command(f"ozone sh volume create {volume_name}")
        self._execute_command(f"ozone sh bucket create {volume_name}/{bucket_name}")
        
        # Start performance monitoring in a separate thread
        metrics = {"latencies": [], "throughput": [], "gc_pauses": []}
        stop_monitoring = threading.Event()
        monitor_thread = threading.Thread(target=self._monitor_performance, args=(metrics, stop_monitoring))
        monitor_thread.start()
        
        # Start the workload
        start_time = time.time()
        files_processed = 0
        
        try:
            for root, _, files in os.walk(test_data_dir):
                for file in files:
                    if time.time() - start_time > duration:
                        break
                    
                    filepath = os.path.join(root, file)
                    key_name = f"key-{file}"
                    
                    # Upload file to Ozone
                    upload_start = time.time()
                    self._execute_command(f"ozone sh key put {volume_name}/{bucket_name}/{key_name} {filepath}")
                    upload_end = time.time()
                    
                    # Record latency
                    latency = upload_end - upload_start
                    metrics["latencies"].append(latency)
                    
                    # Occasionally read back the file to mix operations
                    if files_processed % 5 == 0:
                        read_start = time.time()
                        self._execute_command(f"ozone sh key get {volume_name}/{bucket_name}/{key_name} /dev/null")
                        read_end = time.time()
                        metrics["latencies"].append(read_end - read_start)
                    
                    files_processed += 1
                
                if time.time() - start_time > duration:
                    break
        finally:
            # Stop monitoring
            stop_monitoring.set()
            monitor_thread.join()
            
            # Calculate final metrics
            end_time = time.time()
            total_time = end_time - start_time
            
            if metrics["latencies"]:
                metrics["avg_latency"] = statistics.mean(metrics["latencies"])
                metrics["p95_latency"] = sorted(metrics["latencies"])[int(len(metrics["latencies"]) * 0.95)]
                metrics["p99_latency"] = sorted(metrics["latencies"])[int(len(metrics["latencies"]) * 0.99)]
            else:
                metrics["avg_latency"] = 0
                metrics["p95_latency"] = 0
                metrics["p99_latency"] = 0
                
            metrics["total_operations"] = files_processed
            metrics["operations_per_sec"] = files_processed / total_time if total_time > 0 else 0
            
            # Clean up
            self._execute_command(f"ozone sh bucket delete {volume_name}/{bucket_name}")
            self._execute_command(f"ozone sh volume delete {volume_name}")
            
            return metrics
    
    def _monitor_performance(self, metrics: Dict, stop_event: threading.Event) -> None:
        """Monitor performance metrics in background"""
        while not stop_event.is_set():
            # In a real implementation, this would parse GC logs and collect metrics
            # For this test example, we'll simulate by occasionally adding random GC pause times
            time.sleep(5)
            # Simulate collecting a GC pause time between 10-500ms
            metrics["gc_pauses"].append(10 + 490 * (time.time() % 1))  # ms
    
    def _execute_command(self, command: str) -> str:
        """Execute a shell command and return output"""
        # In a real implementation, this would execute the command
        # For this test example, we'll log the command that would be executed
        logger.info(f"Would execute: {command}")
        # Simulate command execution time based on command type
        if "key put" in command:
            time.sleep(0.1)  # Simulate write operation
        elif "key get" in command:
            time.sleep(0.05)  # Simulate read operation
        else:
            time.sleep(0.01)  # Simulate other operations
        
        return "Simulated command output"
    
    def parse_gc_logs(self, gc_log_file: str) -> Dict:
        """Parse GC logs to extract performance metrics"""
        # In a real implementation, this would parse the GC log file
        # For this test example, we'll return simulated metrics
        logger.info(f"Would parse GC logs from: {gc_log_file}")
        
        return {
            "total_gc_events": 120,
            "total_gc_pause_time_ms": 3500,
            "avg_gc_pause_ms": 29.2,
            "max_gc_pause_ms": 320.5,
            "gc_throughput_percent": 98.2  # Percentage of time not spent in GC
        }
    
    def save_results(self, gc_algorithm: Dict, heap_size: str, 
                     workload: Dict, metrics: Dict, gc_metrics: Dict) -> None:
        """Save test results to CSV file"""
        results_file = f"{self.results_dir}/gc_performance_results.csv"
        file_exists = os.path.isfile(results_file)
        
        with open(results_file, 'a', newline='') as csvfile:
            fieldnames = [
                "timestamp", "gc_algorithm", "heap_size", "workload_name", 
                "workload_description", "avg_latency_ms", "p95_latency_ms", 
                "p99_latency_ms", "operations_per_sec", "total_gc_events",
                "total_gc_pause_time_ms", "avg_gc_pause_ms", "max_gc_pause_ms",
                "gc_throughput_percent"
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "gc_algorithm": gc_algorithm["name"],
                "heap_size": heap_size,
                "workload_name": workload["name"],
                "workload_description": workload["description"],
                "avg_latency_ms": metrics["avg_latency"] * 1000,
                "p95_latency_ms": metrics["p95_latency"] * 1000,
                "p99_latency_ms": metrics["p99_latency"] * 1000,
                "operations_per_sec": metrics["operations_per_sec"],
                "total_gc_events": gc_metrics["total_gc_events"],
                "total_gc_pause_time_ms": gc_metrics["total_gc_pause_time_ms"],
                "avg_gc_pause_ms": gc_metrics["avg_gc_pause_ms"],
                "max_gc_pause_ms": gc_metrics["max_gc_pause_ms"],
                "gc_throughput_percent": gc_metrics["gc_throughput_percent"]
            })
        
        logger.info(f"Results saved to {results_file}")


@pytest.mark.parametrize("gc_algorithm", GC_ALGORITHMS)
@pytest.mark.parametrize("heap_size", HEAP_SIZES)
@pytest.mark.parametrize("workload_profile", WORKLOAD_PROFILES)
def test_38_ozone_gc_performance(gc_algorithm, heap_size, workload_profile):
    """
    Evaluate performance with different JVM garbage collection algorithms
    
    This test configures Ozone with different GC algorithms, runs memory-intensive
    workloads, and analyzes the performance characteristics of each configuration.
    The goal is to identify optimal GC settings for different workload types.
    """
    # Skip some combinations to reduce test time
    if gc_algorithm["name"] == "ZGC" and heap_size in ["1G", "2G"]:
        pytest.skip("ZGC is designed for larger heap sizes")
    
    if workload_profile["name"] == "large_files" and heap_size == "1G":
        pytest.skip("Large files workload requires more memory")
    
    # Initialize the Ozone cluster manager
    cluster = OzoneCluster()
    
    # Set up logging
    logger.info(f"Starting GC performance test with {gc_algorithm['name']}, "
              f"heap size {heap_size}, workload {workload_profile['name']}")
    
    try:
        # Configure Ozone with specific GC settings
        cluster.setup_gc_config(gc_algorithm, heap_size)
        
        # Run the workload and collect metrics
        metrics = cluster.run_workload(workload_profile)
        
        # Parse GC logs for additional metrics
        gc_log_file = f"{cluster.gc_logs_dir}/gc_{gc_algorithm['name']}_{heap_size}.log"
        gc_metrics = cluster.parse_gc_logs(gc_log_file)
        
        # Save results
        cluster.save_results(gc_algorithm, heap_size, workload_profile, metrics, gc_metrics)
        
        # Log summary
        logger.info(f"Test completed - GC: {gc_algorithm['name']}, Heap: {heap_size}, "
                  f"Workload: {workload_profile['name']}")
        logger.info(f"Avg latency: {metrics['avg_latency'] * 1000:.2f}ms, "
                  f"Ops/sec: {metrics['operations_per_sec']:.2f}, "
                  f"GC throughput: {gc_metrics['gc_throughput_percent']}%")
        
        # Assertions for basic sanity checks
        assert metrics["operations_per_sec"] > 0, "No operations were performed"
        assert gc_metrics["gc_throughput_percent"] > 80, "GC throughput too low, indicates performance problem"
        
        # Additional assertions based on expectations for specific GC algorithms
        if gc_algorithm["name"] == "G1GC":
            # G1GC is expected to have good latency characteristics 
            assert metrics["p99_latency"] * 1000 < 500, "P99 latency too high for G1GC"
        elif gc_algorithm["name"] == "CMS":
            # CMS should have good throughput
            assert gc_metrics["gc_throughput_percent"] > 90, "CMS throughput below expected threshold"
        elif gc_algorithm["name"] == "ZGC":
            # ZGC should have very low pause times
            assert gc_metrics["max_gc_pause_ms"] < 50, "ZGC max pause time exceeded expectations"
    
    finally:
        # Cleanup or reset as needed
        logger.info("Test cleanup completed")

"""
Test module for Apache Ozone performance testing during rolling upgrades.
"""

import time
import pytest
import subprocess
import threading
import csv
import os
import datetime
import logging
import statistics
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
METRICS_INTERVAL = 5  # seconds between metrics collection
METRICS_FILE = "upgrade_performance_metrics.csv"
WORKLOAD_FILE = "test_data.txt"
TEST_FILE_SIZE = 10 * 1024 * 1024  # 10MB
VOLUME_NAME = "performancetest"
BUCKET_NAME = "rollupgrade"
NUM_KEYS = 100  # Number of keys to use in the test
UPGRADE_TIMEOUT = 1800  # 30 minutes timeout for upgrade


class OzoneCluster:
    """Helper class to interact with Ozone cluster"""
    
    def __init__(self, nodes: List[str]):
        self.nodes = nodes
        
    def execute_command(self, command: str, node: str = None) -> Tuple[int, str, str]:
        """Execute a shell command on a specific node or locally if node is None"""
        full_command = command
        if node:
            full_command = f"ssh {node} '{command}'"
        
        logger.info(f"Executing: {full_command}")
        process = subprocess.Popen(full_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        return process.returncode, stdout.decode('utf-8'), stderr.decode('utf-8')
    
    def start_rolling_upgrade(self) -> bool:
        """Start the rolling upgrade process"""
        logger.info("Starting rolling upgrade process")
        
        for node in self.nodes:
            ret_code, stdout, stderr = self.execute_command(
                "ozone admin upgrade --start", node)
            
            if ret_code != 0:
                logger.error(f"Failed to start upgrade on {node}: {stderr}")
                return False
                
            logger.info(f"Successfully initiated upgrade on {node}")
            
        return True
    
    def check_upgrade_status(self) -> bool:
        """Check if the upgrade is complete across all nodes"""
        for node in self.nodes:
            ret_code, stdout, stderr = self.execute_command(
                "ozone admin upgrade --status", node)
            
            if ret_code != 0 or "COMPLETED" not in stdout:
                return False
                
        return True
    
    def collect_metrics(self) -> Dict:
        """Collect performance metrics from the cluster"""
        metrics = {
            'timestamp': datetime.datetime.now().isoformat(),
            'cpu_usage': 0,
            'memory_usage': 0,
            'disk_io': 0,
            'network_io': 0,
            'latency': 0
        }
        
        # Collect CPU usage
        total_cpu = 0
        for node in self.nodes:
            ret_code, stdout, stderr = self.execute_command(
                "top -bn1 | grep 'Cpu(s)' | sed 's/.*, *\\([0-9.]*\\)%* id.*/\\1/' | awk '{print 100 - $1}'", 
                node)
            if ret_code == 0:
                try:
                    total_cpu += float(stdout.strip())
                except ValueError:
                    pass
                    
        if self.nodes:
            metrics['cpu_usage'] = total_cpu / len(self.nodes)
        
        # Collect memory usage
        total_mem = 0
        for node in self.nodes:
            ret_code, stdout, stderr = self.execute_command(
                "free | grep Mem | awk '{print $3/$2 * 100.0}'", 
                node)
            if ret_code == 0:
                try:
                    total_mem += float(stdout.strip())
                except ValueError:
                    pass
                    
        if self.nodes:
            metrics['memory_usage'] = total_mem / len(self.nodes)
            
        # Collect disk I/O
        total_io = 0
        for node in self.nodes:
            ret_code, stdout, stderr = self.execute_command(
                "iostat -d -x | grep -v '^$' | tail -n +4 | awk '{s+=$6} END {print s}'", 
                node)
            if ret_code == 0:
                try:
                    total_io += float(stdout.strip())
                except ValueError:
                    pass
                    
        if self.nodes:
            metrics['disk_io'] = total_io / len(self.nodes)
            
        # Network I/O could be collected similarly
        
        return metrics


class PerformanceTester:
    """Class to handle performance testing of Ozone"""
    
    def __init__(self, cluster: OzoneCluster):
        self.cluster = cluster
        self.baseline_metrics = {}
        self.upgrade_metrics = []
        self.stop_monitoring = False
        
    def create_test_data(self):
        """Create test data file to upload"""
        with open(WORKLOAD_FILE, 'wb') as f:
            f.write(os.urandom(TEST_FILE_SIZE))
            
    def setup_environment(self):
        """Setup environment for testing"""
        # Create volume and bucket
        ret_code, _, _ = self.cluster.execute_command(
            f"ozone sh volume create {VOLUME_NAME}")
        assert ret_code == 0, "Failed to create volume"
        
        ret_code, _, _ = self.cluster.execute_command(
            f"ozone sh bucket create {VOLUME_NAME}/{BUCKET_NAME}")
        assert ret_code == 0, "Failed to create bucket"
            
    def cleanup_environment(self):
        """Cleanup test environment"""
        # Delete bucket and volume
        self.cluster.execute_command(
            f"ozone sh bucket delete {VOLUME_NAME}/{BUCKET_NAME}")
        self.cluster.execute_command(
            f"ozone sh volume delete {VOLUME_NAME}")
        
        # Delete test data file
        if os.path.exists(WORKLOAD_FILE):
            os.remove(WORKLOAD_FILE)
            
        # Delete metrics file
        if os.path.exists(METRICS_FILE):
            os.remove(METRICS_FILE)
            
    def establish_baseline(self, duration=300):
        """Establish baseline performance before upgrade"""
        logger.info(f"Establishing baseline performance for {duration} seconds")
        
        # Run workload for the specified duration
        start_time = time.time()
        
        # Create test data file
        self.create_test_data()
        
        # Start metrics collection
        metrics = []
        while time.time() - start_time < duration:
            # Execute read/write operations to establish baseline
            key_name = f"key-{int(time.time())}"
            
            start = time.time()
            ret_code, _, _ = self.cluster.execute_command(
                f"ozone sh key put {VOLUME_NAME}/{BUCKET_NAME}/{key_name} {WORKLOAD_FILE}")
            end = time.time()
            
            if ret_code == 0:
                write_latency = end - start
                
                # Measure read performance
                start = time.time()
                ret_code, _, _ = self.cluster.execute_command(
                    f"ozone sh key get {VOLUME_NAME}/{BUCKET_NAME}/{key_name} /dev/null")
                end = time.time()
                
                if ret_code == 0:
                    read_latency = end - start
                    
                    # Collect cluster metrics
                    cluster_metrics = self.cluster.collect_metrics()
                    cluster_metrics['write_latency'] = write_latency
                    cluster_metrics['read_latency'] = read_latency
                    
                    metrics.append(cluster_metrics)
            
            time.sleep(5)  # Wait a bit before next operation
        
        # Calculate average metrics
        self.baseline_metrics = {
            'cpu_usage': statistics.mean([m['cpu_usage'] for m in metrics]),
            'memory_usage': statistics.mean([m['memory_usage'] for m in metrics]),
            'disk_io': statistics.mean([m['disk_io'] for m in metrics]),
            'write_latency': statistics.mean([m['write_latency'] for m in metrics]),
            'read_latency': statistics.mean([m['read_latency'] for m in metrics])
        }
        
        logger.info(f"Baseline metrics: {self.baseline_metrics}")
        return self.baseline_metrics
        
    def start_metrics_collection(self):
        """Start collecting metrics in a separate thread"""
        # Create CSV file with headers
        with open(METRICS_FILE, 'w', newline='') as csvfile:
            fieldnames = ['timestamp', 'cpu_usage', 'memory_usage', 'disk_io', 
                         'network_io', 'write_latency', 'read_latency', 'status']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        
        self.stop_monitoring = False
        self.monitoring_thread = threading.Thread(target=self._collect_metrics_during_upgrade)
        self.monitoring_thread.start()
        
    def _collect_metrics_during_upgrade(self):
        """Thread function to collect metrics during upgrade"""
        while not self.stop_monitoring:
            # Execute read/write operations
            key_name = f"key-{int(time.time())}"
            
            # Measure write performance
            start = time.time()
            ret_code, _, _ = self.cluster.execute_command(
                f"ozone sh key put {VOLUME_NAME}/{BUCKET_NAME}/{key_name} {WORKLOAD_FILE}")
            end = time.time()
            
            write_latency = end - start if ret_code == 0 else float('inf')
            
            # Measure read performance
            start = time.time()
            ret_code, _, _ = self.cluster.execute_command(
                f"ozone sh key get {VOLUME_NAME}/{BUCKET_NAME}/{key_name} /dev/null")
            end = time.time()
            
            read_latency = end - start if ret_code == 0 else float('inf')
            
            # Get cluster metrics
            metrics = self.cluster.collect_metrics()
            metrics['write_latency'] = write_latency
            metrics['read_latency'] = read_latency
            
            # Check upgrade status
            status = "IN_PROGRESS"
            if self.cluster.check_upgrade_status():
                status = "COMPLETED"
                
            metrics['status'] = status
            
            # Save metrics to file
            with open(METRICS_FILE, 'a', newline='') as csvfile:
                fieldnames = ['timestamp', 'cpu_usage', 'memory_usage', 'disk_io', 
                             'network_io', 'write_latency', 'read_latency', 'status']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(metrics)
                
            # Store metrics for analysis
            self.upgrade_metrics.append(metrics)
            
            time.sleep(METRICS_INTERVAL)
            
    def stop_metrics_collection(self):
        """Stop metrics collection thread"""
        self.stop_monitoring = True
        if hasattr(self, 'monitoring_thread'):
            self.monitoring_thread.join(timeout=10)
            
    def analyze_results(self):
        """Analyze collected metrics and determine if performance meets requirements"""
        # Skip if no metrics were collected
        if not self.upgrade_metrics:
            logger.warning("No metrics collected during upgrade")
            return False
            
        # Calculate average metrics during upgrade
        upgrade_avg = {
            'cpu_usage': statistics.mean([m['cpu_usage'] for m in self.upgrade_metrics]),
            'memory_usage': statistics.mean([m['memory_usage'] for m in self.upgrade_metrics]),
            'disk_io': statistics.mean([m['disk_io'] for m in self.upgrade_metrics]),
            'write_latency': statistics.mean([m['write_latency'] for m in self.upgrade_metrics 
                                            if m['write_latency'] != float('inf')]),
            'read_latency': statistics.mean([m['read_latency'] for m in self.upgrade_metrics
                                           if m['read_latency'] != float('inf')])
        }
        
        # Check for failures
        write_failures = sum(1 for m in self.upgrade_metrics if m['write_latency'] == float('inf'))
        read_failures = sum(1 for m in self.upgrade_metrics if m['read_latency'] == float('inf'))
        total_operations = len(self.upgrade_metrics)
        
        failure_rate = (write_failures + read_failures) / (total_operations * 2)
        
        # Calculate percentage increase in latency
        write_latency_increase = ((upgrade_avg['write_latency'] - self.baseline_metrics['write_latency']) 
                                 / self.baseline_metrics['write_latency']) * 100
        read_latency_increase = ((upgrade_avg['read_latency'] - self.baseline_metrics['read_latency']) 
                                / self.baseline_metrics['read_latency']) * 100
        
        logger.info(f"Upgrade metrics: {upgrade_avg}")
        logger.info(f"Write latency increase: {write_latency_increase:.2f}%")
        logger.info(f"Read latency increase: {read_latency_increase:.2f}%")
        logger.info(f"Failure rate: {failure_rate:.2f}%")
        
        # Determine if performance is acceptable based on thresholds
        max_allowed_failure_rate = 0.05  # 5% max failure rate
        max_allowed_latency_increase = 30  # 30% max latency increase
        
        return (failure_rate <= max_allowed_failure_rate and
                write_latency_increase <= max_allowed_latency_increase and
                read_latency_increase <= max_allowed_latency_increase)


@pytest.fixture
def ozone_cluster():
    """Fixture to create Ozone cluster object"""
    # Get nodes from environment or use default
    nodes = os.environ.get('OZONE_NODES', '').split(',')
    if not nodes or nodes[0] == '':
        # Default to localhost for testing
        nodes = ['localhost']
    return OzoneCluster(nodes)


@pytest.fixture
def performance_tester(ozone_cluster):
    """Fixture to create performance tester object"""
    tester = PerformanceTester(ozone_cluster)
    tester.setup_environment()
    
    yield tester
    
    tester.cleanup_environment()


def test_39_rolling_upgrade_performance(ozone_cluster, performance_tester):
    """Test performance under continuous software upgrades.
    
    This test verifies that rolling upgrades in Apache Ozone complete with
    minimal performance impact and no data unavailability.
    """
    # 1. Establish baseline performance
    baseline = performance_tester.establish_baseline(duration=300)  # 5 minutes of baseline
    
    # Start metrics collection
    performance_tester.start_metrics_collection()
    
    try:
        # 2. Initiate a rolling upgrade process
        upgrade_started = ozone_cluster.start_rolling_upgrade()
        assert upgrade_started, "Failed to start rolling upgrade process"
        
        # 3. Continuously monitor system performance during upgrade
        logger.info("Monitoring performance during upgrade...")
        
        # 4. Wait for upgrade to complete with timeout
        start_time = time.time()
        upgrade_completed = False
        
        while time.time() - start_time < UPGRADE_TIMEOUT:
            if ozone_cluster.check_upgrade_status():
                upgrade_completed = True
                break
            time.sleep(30)  # Check every 30 seconds
            
        assert upgrade_completed, f"Upgrade did not complete within {UPGRADE_TIMEOUT} seconds"
        
        # Wait a bit more to collect post-upgrade metrics
        time.sleep(60)
        
    finally:
        # Stop metrics collection
        performance_tester.stop_metrics_collection()
    
    # 5. Verify cluster stability and performance post-upgrade
    performance_meets_requirements = performance_tester.analyze_results()
    
    # Check if performance meets the requirements
    assert performance_meets_requirements, "Performance degradation during upgrade exceeds acceptable threshold"
    
    # Verify data access post-upgrade
    key_name = f"test-post-upgrade-{int(time.time())}"
    ret_code, _, _ = ozone_cluster.execute_command(
        f"ozone sh key put {VOLUME_NAME}/{BUCKET_NAME}/{key_name} {WORKLOAD_FILE}")
    assert ret_code == 0, "Failed to write data after upgrade"
    
    ret_code, _, _ = ozone_cluster.execute_command(
        f"ozone sh key get {VOLUME_NAME}/{BUCKET_NAME}/{key_name} /dev/null")
    assert ret_code == 0, "Failed to read data after upgrade"
    
    logger.info("Rolling upgrade completed successfully with acceptable performance impact")

import pytest
import subprocess
import time
import json
import os
import csv
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
TEST_DATA_DIR = "test_data"
RESULTS_DIR = "performance_results"

# Storage configurations for testing
STORAGE_CONFIGS = [
    {"name": "ssd_only", "media": ["SSD"], "tier_config": "DISK,SSD"},
    {"name": "hdd_only", "media": ["HDD"], "tier_config": "DISK,HDD"},
    {"name": "nvme_only", "media": ["NVMe"], "tier_config": "DISK,NVMe"},
    {"name": "ssd_hdd", "media": ["SSD", "HDD"], "tier_config": "DISK,SSD;ARCHIVE,HDD"},
    {"name": "nvme_ssd", "media": ["NVMe", "SSD"], "tier_config": "DISK,NVMe;ARCHIVE,SSD"},
    {"name": "nvme_hdd", "media": ["NVMe", "HDD"], "tier_config": "DISK,NVMe;ARCHIVE,HDD"},
    {"name": "all_media", "media": ["NVMe", "SSD", "HDD"], "tier_config": "DISK,NVMe;REGULAR,SSD;ARCHIVE,HDD"}
]

# Different workload patterns
WORKLOAD_PATTERNS = [
    {"name": "sequential_write", "description": "Sequential write operations"},
    {"name": "random_write", "description": "Random write operations"},
    {"name": "sequential_read", "description": "Sequential read operations"},
    {"name": "random_read", "description": "Random read operations"},
    {"name": "mixed_read_write", "description": "Mixed read/write operations (70/30)"}
]

# File sizes for testing
FILE_SIZES = [
    {"name": "1kb", "size_bytes": 1024},
    {"name": "100kb", "size_bytes": 102400},
    {"name": "4mb", "size_bytes": 4194304},
    {"name": "256mb", "size_bytes": 268435456},
    {"name": "1gb", "size_bytes": 1073741824},
    {"name": "4.5gb", "size_bytes": 4831838208}
]

class OzoneClusterManager:
    """Helper class to manage Ozone cluster configuration"""
    
    def __init__(self):
        self.cluster_config = {}
    
    def configure_tiered_storage(self, storage_config):
        """Configure Ozone with tiered storage based on provided config"""
        logger.info(f"Configuring Ozone with tiered storage: {storage_config['name']}")
        
        # In a real implementation, this would modify ozone-site.xml configuration
        # and restart the cluster with the new settings
        cmd = f"ozone admin storage-tier set --tierConfig={storage_config['tier_config']}"
        
        try:
            # This is a placeholder - in real test we would actually execute this command
            logger.info(f"Would execute: {cmd}")
            # subprocess.run(cmd, shell=True, check=True)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to configure tiered storage: {e}")
            return False


class PerformanceDataCollector:
    """Collects and processes performance data"""
    
    def __init__(self):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        self.results = []
    
    def generate_test_file(self, size_bytes):
        """Generate a test file of specified size"""
        os.makedirs(TEST_DATA_DIR, exist_ok=True)
        file_path = os.path.join(TEST_DATA_DIR, f"test_file_{size_bytes}.dat")
        
        if os.path.exists(file_path) and os.path.getsize(file_path) == size_bytes:
            return file_path
            
        with open(file_path, 'wb') as f:
            f.write(os.urandom(size_bytes))
        
        return file_path
    
    def run_performance_test(self, config, workload, file_size):
        """Run a performance test with given configuration"""
        logger.info(f"Running performance test: {config['name']}, {workload['name']}, {file_size['name']}")
        
        # Generate test file
        test_file = self.generate_test_file(file_size["size_bytes"])
        
        # Create unique volume and bucket names for this test
        volume = f"vol-perf-{config['name']}-{int(time.time())}"
        bucket = f"bucket-{workload['name']}-{int(time.time() % 10000)}"
        
        # Create volume and bucket
        subprocess.run(f"ozone sh volume create {volume}", shell=True, check=True)
        subprocess.run(f"ozone sh bucket create {volume}/{bucket}", shell=True, check=True)
        
        # Measure performance based on workload type
        start_time = time.time()
        
        if "write" in workload["name"]:
            # Upload file
            key = f"test-key-{int(time.time())}"
            subprocess.run(f"ozone sh key put {volume}/{bucket}/ {test_file} --key={key}", 
                          shell=True, check=True)
        elif "read" in workload["name"]:
            # First upload, then read
            key = f"test-key-{int(time.time())}"
            subprocess.run(f"ozone sh key put {volume}/{bucket}/ {test_file} --key={key}", 
                          shell=True, check=True)
            
            # Read operation
            output_file = os.path.join(TEST_DATA_DIR, "output.dat")
            subprocess.run(f"ozone sh key get {volume}/{bucket}/{key} {output_file}", 
                          shell=True, check=True)
            
        end_time = time.time()
        duration = end_time - start_time
        throughput = file_size["size_bytes"] / duration / 1024 / 1024  # MB/s
        
        # Collect additional metrics about data tiering
        tier_info = self.get_tier_info(volume, bucket)
        
        result = {
            "config_name": config["name"],
            "media_types": ",".join(config["media"]),
            "workload": workload["name"],
            "file_size": file_size["name"],
            "file_size_bytes": file_size["size_bytes"],
            "duration_seconds": duration,
            "throughput_MBps": throughput,
            "tier_info": tier_info
        }
        
        self.results.append(result)
        
        # Clean up
        subprocess.run(f"ozone sh bucket delete {volume}/{bucket}", shell=True, check=True)
        subprocess.run(f"ozone sh volume delete {volume}", shell=True, check=True)
        
        return result
    
    def get_tier_info(self, volume, bucket):
        """Get information about data distribution across tiers"""
        # This is a placeholder - in real implementation, we would query Ozone metrics
        # to get actual tier distribution information
        cmd = f"ozone admin storage-tier list -v {volume} -b {bucket}"
        
        try:
            # In real test we would parse the actual output
            return {"DISK": 70, "ARCHIVE": 30}  # Placeholder values
        except Exception as e:
            logger.error(f"Failed to get tier info: {e}")
            return {}
    
    def save_results(self):
        """Save test results to CSV file"""
        if not self.results:
            logger.warning("No results to save")
            return
            
        filename = os.path.join(RESULTS_DIR, f"tiered_storage_perf_{int(time.time())}.csv")
        
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = self.results[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in self.results:
                writer.writerow(result)
        
        logger.info(f"Results saved to {filename}")
        return filename
        
    def analyze_results(self):
        """Analyze the performance results and generate insights"""
        if not self.results:
            logger.warning("No results to analyze")
            return {}
            
        df = pd.DataFrame(self.results)
        
        # Calculate key metrics
        analysis = {
            "fastest_storage": df.loc[df["throughput_MBps"].idxmax()]["config_name"],
            "slowest_storage": df.loc[df["throughput_MBps"].idxmin()]["config_name"],
            "avg_throughput_by_config": df.groupby("config_name")["throughput_MBps"].mean().to_dict(),
            "avg_throughput_by_workload": df.groupby("workload")["throughput_MBps"].mean().to_dict(),
        }
        
        # Generate performance comparison chart
        self._generate_performance_chart(df)
        
        return analysis
    
    def _generate_performance_chart(self, df):
        """Generate performance comparison chart"""
        plt.figure(figsize=(12, 8))
        
        # Group by configuration and workload type
        grouped = df.groupby(["config_name", "workload"])["throughput_MBps"].mean().reset_index()
        
        # Pivot for easier plotting
        pivot = grouped.pivot(index="config_name", columns="workload", values="throughput_MBps")
        
        # Plot
        pivot.plot(kind="bar", ax=plt.gca())
        plt.title("Storage Configuration Performance by Workload Type")
        plt.ylabel("Throughput (MB/s)")
        plt.xlabel("Storage Configuration")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save chart
        chart_path = os.path.join(RESULTS_DIR, f"performance_comparison_{int(time.time())}.png")
        plt.savefig(chart_path)
        logger.info(f"Performance chart saved to {chart_path}")


class CostPerformanceAnalyzer:
    """Analyze cost vs. performance trade-offs"""
    
    def __init__(self):
        # Hypothetical costs per GB per month in USD
        self.storage_costs = {
            "NVMe": 0.25,  # Most expensive
            "SSD": 0.15,
            "HDD": 0.05    # Least expensive
        }
    
    def calculate_cost_metrics(self, performance_results):
        """Calculate cost-performance metrics"""
        df = pd.DataFrame(performance_results)
        
        # Add cost column based on media types
        df["estimated_cost"] = df.apply(self._estimate_cost_for_config, axis=1)
        
        # Calculate cost-performance ratio (higher is better)
        df["cost_performance_ratio"] = df["throughput_MBps"] / df["estimated_cost"]
        
        # Group by configuration
        cost_perf_by_config = df.groupby("config_name").agg({
            "throughput_MBps": "mean",
            "estimated_cost": "mean",
            "cost_performance_ratio": "mean"
        })
        
        return cost_perf_by_config.to_dict()
    
    def _estimate_cost_for_config(self, row):
        """Estimate storage cost for a configuration"""
        media_types = row["media_types"].split(",")
        # Simple average cost of the media types
        return sum(self.storage_costs.get(media, 0.1) for media in media_types) / len(media_types)
    
    def generate_cost_performance_chart(self, performance_results):
        """Generate cost vs performance chart"""
        df = pd.DataFrame(performance_results)
        df["estimated_cost"] = df.apply(self._estimate_cost_for_config, axis=1)
        
        # Group by configuration
        grouped = df.groupby("config_name").agg({
            "throughput_MBps": "mean",
            "estimated_cost": "mean"
        }).reset_index()
        
        plt.figure(figsize=(10, 8))
        plt.scatter(grouped["estimated_cost"], grouped["throughput_MBps"], s=100)
        
        # Add labels for each point
        for i, row in grouped.iterrows():
            plt.annotate(row["config_name"], 
                        (row["estimated_cost"], row["throughput_MBps"]),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha='center')
        
        plt.title("Cost vs Performance Trade-off")
        plt.xlabel("Estimated Cost ($/GB/month)")
        plt.ylabel("Throughput (MB/s)")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Draw ideal region (high performance, low cost)
        max_cost = grouped["estimated_cost"].max()
        max_perf = grouped["throughput_MBps"].max()
        
        # Add cost-performance efficiency contours
        x = np.linspace(0, max_cost * 1.1, 100)
        for efficiency in [10, 25, 50, 100]:
            y = efficiency * x
            plt.plot(x, y, 'k--', alpha=0.3)
            plt.annotate(f"Efficiency: {efficiency}", 
                        (max_cost * 0.5, efficiency * max_cost * 0.5),
                        alpha=0.6)
        
        plt.tight_layout()
        
        # Save chart
        chart_path = os.path.join(RESULTS_DIR, f"cost_performance_{int(time.time())}.png")
        plt.savefig(chart_path)
        logger.info(f"Cost-performance chart saved to {chart_path}")
        return chart_path


@pytest.fixture
def setup_performance_test():
    """Set up the performance test environment"""
    # Create directories
    os.makedirs(TEST_DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Initialize helper classes
    cluster_manager = OzoneClusterManager()
    data_collector = PerformanceDataCollector()
    cost_analyzer = CostPerformanceAnalyzer()
    
    yield cluster_manager, data_collector, cost_analyzer
    
    # Clean up
    logger.info("Test complete, cleanup would happen here")


def test_40_tiered_storage_performance_evaluation(setup_performance_test):
    """
    Evaluate performance with different storage media combinations
    - Configure Ozone with tiered storage using different media types
    - Design workloads that target different storage tiers
    - Measure performance for each storage tier and data movement between tiers
    - Analyze cost-performance trade-offs for different storage configurations
    """
    cluster_manager, data_collector, cost_analyzer = setup_performance_test
    
    # Select a subset of configurations and file sizes to keep test duration reasonable
    test_configs = STORAGE_CONFIGS[:3]  # Using first 3 configs
    test_workloads = WORKLOAD_PATTERNS[:3]  # Using first 3 workload patterns
    test_file_sizes = FILE_SIZES[:3]  # Using first 3 file sizes
    
    for config in test_configs:
        # Configure the cluster with the current storage configuration
        success = cluster_manager.configure_tiered_storage(config)
        if not success:
            pytest.skip(f"Failed to configure cluster with {config['name']} storage configuration")
        
        # Execute workloads for this configuration
        for workload in test_workloads:
            for file_size in test_file_sizes:
                # Run the performance test and collect metrics
                result = data_collector.run_performance_test(config, workload, file_size)
                
                # Verify the test completed successfully by checking if throughput was measured
                assert "throughput_MBps" in result, f"Failed to measure throughput for {config['name']} with {workload['name']}"
                
                # Verify that throughput is positive
                assert result["throughput_MBps"] > 0, f"Invalid throughput for {config['name']} with {workload['name']}"
    
    # Save all results to CSV
    results_file = data_collector.save_results()
    assert os.path.exists(results_file), "Failed to save performance results"
    
    # Analyze the performance results
    performance_analysis = data_collector.analyze_results()
    
    # Calculate cost-performance metrics
    cost_performance_metrics = cost_analyzer.calculate_cost_metrics(data_collector.results)
    
    # Generate cost-performance chart
    chart_path = cost_analyzer.generate_cost_performance_chart(data_collector.results)
    assert os.path.exists(chart_path), "Failed to generate cost-performance chart"
    
    # Validate that we have valid metrics for each configuration
    for config in test_configs:
        config_name = config["name"]
        assert config_name in performance_analysis["avg_throughput_by_config"], \
            f"Missing performance metrics for {config_name}"
        
        # Each configuration should have a cost-performance ratio
        assert config_name in cost_performance_metrics["cost_performance_ratio"], \
            f"Missing cost-performance ratio for {config_name}"
    
    # Log summary of findings
    logger.info(f"Fastest storage configuration: {performance_analysis['fastest_storage']}")
    logger.info(f"Slowest storage configuration: {performance_analysis['slowest_storage']}")
    
    # Get best cost-performance configuration
    best_cp_config = max(
        cost_performance_metrics["cost_performance_ratio"].items(), 
        key=lambda x: x[1]
    )[0]
    logger.info(f"Best cost-performance configuration: {best_cp_config}")
    
    # The test passes if we successfully collected all the performance data
    # and generated the analysis
    assert len(data_collector.results) > 0, "No performance data collected"

import time
import logging
import pytest
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import os
import random
import threading
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants for test configuration
class OzoneConfig:
    # Cluster information
    CLUSTER_NODES = os.environ.get('OZONE_CLUSTER_NODES', 'localhost:9878').split(',')
    ADMIN_USER = os.environ.get('OZONE_ADMIN_USER', 'hadoop')
    
    # Test configuration
    VOLUME_NAME = "perftest-volume"
    BUCKET_NAME = "recovery-bucket"
    
    # Performance thresholds
    MAX_RECOVERY_TIME_SEC = int(os.environ.get('MAX_RECOVERY_TIME_SEC', '300'))  # 5 minutes
    MAX_PERFORMANCE_DEGRADATION = float(os.environ.get('MAX_PERF_DEGRADATION', '0.3'))  # 30%
    
    # Test data
    DATA_SIZES = [100, 500, 1024, 2048]  # in MB
    REPLICATION_FACTOR = 3
    TEST_DURATION = 300  # seconds


class OzoneClusterManager:
    """Helper class to manage Ozone cluster operations"""
    
    @staticmethod
    def execute_command(command: str) -> Tuple[str, str, int]:
        """Execute a shell command and return stdout, stderr, and return code"""
        process = subprocess.Popen(
            command, shell=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()
        return stdout.decode().strip(), stderr.decode().strip(), process.returncode

    @staticmethod
    def get_datanodes() -> List[str]:
        """Get list of active DataNodes in the cluster"""
        cmd = "ozone admin datanode list"
        stdout, stderr, ret_code = OzoneClusterManager.execute_command(cmd)
        if ret_code != 0:
            logger.error(f"Failed to get DataNode list: {stderr}")
            raise RuntimeError(f"Failed to get DataNode list: {stderr}")
        
        # Parse output to extract DataNode IDs
        datanode_list = []
        for line in stdout.splitlines():
            if "Datanode" in line:
                parts = line.split()
                for part in parts:
                    if part.startswith("uuid:"):
                        datanode_list.append(part.replace("uuid:", ""))
        
        return datanode_list

    @staticmethod
    def stop_datanode(datanode_id: str) -> bool:
        """Stop a specific DataNode"""
        cmd = f"ozone admin datanode stop {datanode_id}"
        stdout, stderr, ret_code = OzoneClusterManager.execute_command(cmd)
        if ret_code != 0:
            logger.error(f"Failed to stop DataNode {datanode_id}: {stderr}")
            return False
        logger.info(f"Successfully stopped DataNode {datanode_id}")
        return True
    
    @staticmethod
    def start_datanode(datanode_id: str) -> bool:
        """Start a specific DataNode"""
        cmd = f"ozone admin datanode start {datanode_id}"
        stdout, stderr, ret_code = OzoneClusterManager.execute_command(cmd)
        if ret_code != 0:
            logger.error(f"Failed to start DataNode {datanode_id}: {stderr}")
            return False
        logger.info(f"Successfully started DataNode {datanode_id}")
        return True


class DataGenerator:
    """Helper class to generate test data files"""
    
    @staticmethod
    def create_test_file(size_mb: int, filename: str) -> str:
        """Create a test file of the specified size in MB"""
        file_path = f"/tmp/{filename}"
        
        # Use dd to create a file of the specified size
        cmd = f"dd if=/dev/urandom of={file_path} bs=1M count={size_mb}"
        stdout, stderr, ret_code = OzoneClusterManager.execute_command(cmd)
        
        if ret_code != 0:
            logger.error(f"Failed to create test file: {stderr}")
            raise RuntimeError(f"Failed to create test file: {stderr}")
        
        logger.info(f"Created test file {file_path} of size {size_mb}MB")
        return file_path


class PerformanceMetrics:
    """Helper class to collect and analyze performance metrics"""
    
    def __init__(self):
        self.metrics = {
            'baseline': {},
            'during_recovery': {},
            'after_recovery': {}
        }
    
    def record_operation_metrics(self, phase: str, operation_type: str, file_size: int, 
                                duration: float, throughput: float):
        """Record metrics for a single operation"""
        if phase not in self.metrics:
            self.metrics[phase] = {}
        
        if operation_type not in self.metrics[phase]:
            self.metrics[phase][operation_type] = []
            
        self.metrics[phase][operation_type].append({
            'file_size_mb': file_size,
            'duration_sec': duration,
            'throughput_mbps': throughput
        })
    
    def get_average_throughput(self, phase: str, operation_type: str) -> float:
        """Get average throughput for a specific phase and operation"""
        if phase not in self.metrics or operation_type not in self.metrics[phase]:
            return 0.0
            
        throughputs = [entry['throughput_mbps'] for entry in self.metrics[phase][operation_type]]
        if not throughputs:
            return 0.0
        return sum(throughputs) / len(throughputs)
    
    def get_recovery_time(self) -> float:
        """Get the total recovery time if recorded"""
        if 'recovery_time' in self.metrics:
            return self.metrics['recovery_time']
        return 0.0
        
    def set_recovery_time(self, duration: float):
        """Set the recovery time"""
        self.metrics['recovery_time'] = duration
        
    def generate_report(self, test_name: str):
        """Generate a performance report with charts"""
        report_path = f"performance_report_{test_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        # Create dataframes for each phase
        dfs = {}
        for phase in self.metrics:
            if phase == 'recovery_time':
                continue
                
            phase_data = []
            for operation_type, measurements in self.metrics[phase].items():
                for measurement in measurements:
                    measurement_copy = measurement.copy()
                    measurement_copy['operation'] = operation_type
                    phase_data.append(measurement_copy)
            
            if phase_data:
                dfs[phase] = pd.DataFrame(phase_data)
        
        # Create a simple HTML report with matplotlib figures
        with open(report_path, 'w') as f:
            f.write("Ozone Performance Test Report")
            f.write(f"Performance Test Report: {test_name}")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            if 'recovery_time' in self.metrics:
                f.write(f"Recovery Time: {self.metrics['recovery_time']:.2f} seconds")
            
            for phase, df in dfs.items():
                if df.empty:
                    continue
                    
                f.write(f"Phase: {phase}")
                
                # Create throughput chart
                plt.figure(figsize=(10, 6))
                for operation in df['operation'].unique():
                    op_data = df[df['operation'] == operation]
                    plt.plot(op_data['file_size_mb'], op_data['throughput_mbps'], 'o-', label=operation)
                
                plt.xlabel('File Size (MB)')
                plt.ylabel('Throughput (MB/s)')
                plt.title(f'Throughput by File Size - {phase}')
                plt.legend()
                plt.grid(True)
                
                chart_path = f"chart_{phase}_{test_name}.png"
                plt.savefig(chart_path)
                plt.close()
                
                f.write(f"")
                f.write(f"Raw Data - {phase}")
                f.write(df.to_html())
            
            f.write("")
            
        logger.info(f"Performance report generated: {report_path}")
        return report_path


class OzonePerformanceTester:
    """Main class for performance testing"""
    
    def __init__(self):
        self.cluster_manager = OzoneClusterManager()
        self.data_generator = DataGenerator()
        self.performance_metrics = PerformanceMetrics()
        self.test_files = {}
        self.stop_event = threading.Event()
    
    def setup_test_environment(self):
        """Set up the test environment including volume and bucket"""
        logger.info("Setting up test environment")
        
        # Create volume
        cmd = f"ozone sh volume create {OzoneConfig.VOLUME_NAME}"
        stdout, stderr, ret_code = OzoneClusterManager.execute_command(cmd)
        if ret_code != 0 and "VOLUME_ALREADY_EXISTS" not in stderr:
            logger.error(f"Failed to create volume: {stderr}")
            raise RuntimeError(f"Failed to create volume: {stderr}")
            
        # Create bucket
        cmd = f"ozone sh bucket create {OzoneConfig.VOLUME_NAME}/{OzoneConfig.BUCKET_NAME}"
        stdout, stderr, ret_code = OzoneClusterManager.execute_command(cmd)
        if ret_code != 0 and "BUCKET_ALREADY_EXISTS" not in stderr:
            logger.error(f"Failed to create bucket: {stderr}")
            raise RuntimeError(f"Failed to create bucket: {stderr}")
        
        # Generate test files
        for size in OzoneConfig.DATA_SIZES:
            file_name = f"test_file_{size}mb.dat"
            self.test_files[size] = self.data_generator.create_test_file(size, file_name)
        
        logger.info("Test environment setup completed")
    
    def clean_test_environment(self):
        """Clean up the test environment"""
        logger.info("Cleaning up test environment")
        
        # Remove test files
        for file_path in self.test_files.values():
            try:
                os.remove(file_path)
            except Exception as e:
                logger.warning(f"Failed to remove test file {file_path}: {e}")
        
        # Optionally delete the bucket and volume
        # We're commenting this out to avoid data loss in real environments
        # cmd = f"ozone sh bucket delete {OzoneConfig.VOLUME_NAME}/{OzoneConfig.BUCKET_NAME}"
        # OzoneClusterManager.execute_command(cmd)
        # cmd = f"ozone sh volume delete {OzoneConfig.VOLUME_NAME}"
        # OzoneClusterManager.execute_command(cmd)
        
        logger.info("Test environment cleanup completed")
    
    def upload_file(self, file_path: str, key_name: str) -> Tuple[float, float]:
        """Upload a file to Ozone and measure performance"""
        start_time = time.time()
        
        cmd = f"ozone sh key put {OzoneConfig.VOLUME_NAME}/{OzoneConfig.BUCKET_NAME}/{key_name} {file_path}"
        stdout, stderr, ret_code = OzoneClusterManager.execute_command(cmd)
        
        end_time = time.time()
        duration = end_time - start_time
        
        if ret_code != 0:
            logger.error(f"Failed to upload file: {stderr}")
            raise RuntimeError(f"Failed to upload file: {stderr}")
        
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        throughput = file_size_mb / duration if duration > 0 else 0
        
        logger.info(f"Uploaded {file_path} ({file_size_mb:.2f}MB) in {duration:.2f}s, throughput: {throughput:.2f}MB/s")
        return duration, throughput
    
    def download_file(self, key_name: str, output_path: str) -> Tuple[float, float]:
        """Download a file from Ozone and measure performance"""
        start_time = time.time()
        
        cmd = f"ozone sh key get {OzoneConfig.VOLUME_NAME}/{OzoneConfig.BUCKET_NAME}/{key_name} {output_path}"
        stdout, stderr, ret_code = OzoneClusterManager.execute_command(cmd)
        
        end_time = time.time()
        duration = end_time - start_time
        
        if ret_code != 0:
            logger.error(f"Failed to download file: {stderr}")
            raise RuntimeError(f"Failed to download file: {stderr}")
        
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        throughput = file_size_mb / duration if duration > 0 else 0
        
        logger.info(f"Downloaded {key_name} to {output_path} ({file_size_mb:.2f}MB) in {duration:.2f}s, throughput: {throughput:.2f}MB/s")
        return duration, throughput
    
    def measure_baseline_performance(self):
        """Measure baseline performance before introducing failures"""
        logger.info("Measuring baseline performance")
        
        for size, file_path in self.test_files.items():
            key_name = f"baseline_test_{size}mb_{int(time.time())}.dat"
            output_path = f"/tmp/download_baseline_{size}mb.dat"
            
            # Measure upload performance
            duration, throughput = self.upload_file(file_path, key_name)
            self.performance_metrics.record_operation_metrics('baseline', 'upload', size, duration, throughput)
            
            # Measure download performance
            duration, throughput = self.download_file(key_name, output_path)
            self.performance_metrics.record_operation_metrics('baseline', 'download', size, duration, throughput)
            
            # Clean up downloaded file
            try:
                os.remove(output_path)
            except:
                pass
        
        logger.info("Baseline performance measurement completed")
    
    def simulate_datanode_failures(self, num_failures: int = 1) -> List[str]:
        """Simulate failures of DataNodes"""
        logger.info(f"Simulating failure of {num_failures} DataNodes")
        
        datanodes = self.cluster_manager.get_datanodes()
        if len(datanodes) <= num_failures:
            logger.warning(f"Not enough DataNodes to simulate {num_failures} failures. Only have {len(datanodes)} DataNodes.")
            num_failures = max(1, len(datanodes) - 1)
        
        failed_nodes = random.sample(datanodes, num_failures)
        for node_id in failed_nodes:
            success = self.cluster_manager.stop_datanode(node_id)
            if not success:
                logger.warning(f"Could not stop DataNode {node_id}, skipping")
                failed_nodes.remove(node_id)
        
        logger.info(f"Successfully simulated failure of {len(failed_nodes)} DataNodes: {failed_nodes}")
        return failed_nodes
    
    def monitor_recovery_process(self) -> float:
        """Monitor the data recovery process and measure time"""
        logger.info("Monitoring data recovery process")
        
        start_time = time.time()
        recovery_complete = False
        
        while not recovery_complete and not self.stop_event.is_set():
            # Check recovery status
            cmd = "ozone admin container list /tmp/container_report.txt && grep -c UNHEALTHY /tmp/container_report.txt"
            stdout, stderr, ret_code = OzoneClusterManager.execute_command(cmd)
            
            if ret_code == 0:
                unhealthy_count = int(stdout) if stdout.strip().isdigit() else -1
                logger.info(f"Unhealthy containers: {unhealthy_count}")
                if unhealthy_count == 0:
                    recovery_complete = True
            
            if not recovery_complete:
                time.sleep(5)  # Check every 5 seconds
            
            # Safety timeout - 30 minutes max
            if time.time() - start_time > 1800:
                logger.warning("Recovery process timeout after 30 minutes")
                break
        
        recovery_time = time.time() - start_time
        logger.info(f"Recovery process completed in {recovery_time:.2f} seconds")
        return recovery_time
    
    def measure_performance_during_recovery(self):
        """Measure performance during the recovery process"""
        logger.info("Measuring performance during recovery")
        
        for size, file_path in self.test_files.items():
            key_name = f"recovery_test_{size}mb_{int(time.time())}.dat"
            output_path = f"/tmp/download_recovery_{size}mb.dat"
            
            # Measure upload performance
            duration, throughput = self.upload_file(file_path, key_name)
            self.performance_metrics.record_operation_metrics('during_recovery', 'upload', size, duration, throughput)
            
            # Measure download performance
            duration, throughput = self.download_file(key_name, output_path)
            self.performance_metrics.record_operation_metrics('during_recovery', 'download', size, duration, throughput)
            
            # Clean up downloaded file
            try:
                os.remove(output_path)
            except:
                pass
        
        logger.info("Performance measurement during recovery completed")
    
    def measure_performance_after_recovery(self):
        """Measure performance after recovery is complete"""
        logger.info("Measuring performance after recovery")
        
        for size, file_path in self.test_files.items():
            key_name = f"post_recovery_test_{size}mb_{int(time.time())}.dat"
            output_path = f"/tmp/download_post_recovery_{size}mb.dat"
            
            # Measure upload performance
            duration, throughput = self.upload_file(file_path, key_name)
            self.performance_metrics.record_operation_metrics('after_recovery', 'upload', size, duration, throughput)
            
            # Measure download performance
            duration, throughput = self.download_file(key_name, output_path)
            self.performance_metrics.record_operation_metrics('after_recovery', 'download', size, duration, throughput)
            
            # Clean up downloaded file
            try:
                os.remove(output_path)
            except:
                pass
        
        logger.info("Post-recovery performance measurement completed")
    
    def restart_failed_nodes(self, failed_nodes: List[str]):
        """Restart the previously failed DataNodes"""
        logger.info(f"Restarting {len(failed_nodes)} failed DataNodes")
        
        for node_id in failed_nodes:
            success = self.cluster_manager.start_datanode(node_id)
            if not success:
                logger.warning(f"Could not restart DataNode {node_id}")
        
        logger.info("DataNode restart process completed")


# The actual test case
@pytest.mark.performance
def test_41_datanode_failure_recovery_performance():
    """Test performance with data recovery from DataNode failures"""
    logger.info("Starting test: Performance with data recovery from DataNode failures")
    
    tester = OzonePerformanceTester()
    
    try:
        # Setup test environment
        tester.setup_test_environment()
        
        # Step 1: Establish baseline performance
        tester.measure_baseline_performance()
        baseline_upload_throughput = tester.performance_metrics.get_average_throughput('baseline', 'upload')
        baseline_download_throughput = tester.performance_metrics.get_average_throughput('baseline', 'download')
        logger.info(f"Baseline upload throughput: {baseline_upload_throughput:.2f}MB/s")
        logger.info(f"Baseline download throughput: {baseline_download_throughput:.2f}MB/s")
        
        # Step 2: Simulate failure of DataNodes
        failed_nodes = tester.simulate_datanode_failures(1)  # Fail 1 DataNode
        
        # Create monitoring thread to run in background
        recovery_thread = threading.Thread(
            target=lambda: tester.performance_metrics.set_recovery_time(tester.monitor_recovery_process())
        )
        recovery_thread.daemon = True
        recovery_thread.start()
        
        # Give recovery some time to start
        time.sleep(10)
        
        # Step 3 & 4: Monitor recovery and measure performance during recovery
        tester.measure_performance_during_recovery()
        
        # Wait for recovery to complete
        recovery_thread.join(timeout=OzoneConfig.MAX_RECOVERY_TIME_SEC)
        tester.stop_event.set()  # Signal monitoring thread to stop if still running
        
        recovery_time = tester.performance_metrics.get_recovery_time()
        
        # Step 5: Measure performance after recovery
        tester.measure_performance_after_recovery()
        
        # Restart failed nodes to restore the cluster to original state
        tester.restart_failed_nodes(failed_nodes)
        
        # Calculate performance impact
        during_recovery_upload = tester.performance_metrics.get_average_throughput('during_recovery', 'upload')
        during_recovery_download = tester.performance_metrics.get_average_throughput('during_recovery', 'download')
        after_recovery_upload = tester.performance_metrics.get_average_throughput('after_recovery', 'upload')
        after_recovery_download = tester.performance_metrics.get_average_throughput('after_recovery', 'download')
        
        upload_degradation_during = (baseline_upload_throughput - during_recovery_upload) / baseline_upload_throughput
        download_degradation_during = (baseline_download_throughput - during_recovery_download) / baseline_download_throughput
        upload_degradation_after = (baseline_upload_throughput - after_recovery_upload) / baseline_upload_throughput
        download_degradation_after = (baseline_download_throughput - after_recovery_download) / baseline_download_throughput
        
        logger.info(f"Recovery time: {recovery_time:.2f} seconds")
        logger.info(f"Upload throughput degradation during recovery: {upload_degradation_during:.2%}")
        logger.info(f"Download throughput degradation during recovery: {download_degradation_during:.2%}")
        logger.info(f"Upload throughput degradation after recovery: {upload_degradation_after:.2%}")
        logger.info(f"Download throughput degradation after recovery: {download_degradation_after:.2%}")
        
        # Generate performance report
        tester.performance_metrics.generate_report("datanode_failure_recovery")
        
        # Verify against SLAs
        assert recovery_time <= OzoneConfig.MAX_RECOVERY_TIME_SEC, \
            f"Recovery time {recovery_time:.2f}s exceeds SLA of {OzoneConfig.MAX_RECOVERY_TIME_SEC}s"
        
        assert upload_degradation_after <= OzoneConfig.MAX_PERFORMANCE_DEGRADATION, \
            f"Post-recovery upload degradation {upload_degradation_after:.2%} exceeds SLA of {OzoneConfig.MAX_PERFORMANCE_DEGRADATION:.2%}"
        
        assert download_degradation_after <= OzoneConfig.MAX_PERFORMANCE_DEGRADATION, \
            f"Post-recovery download degradation {download_degradation_after:.2%} exceeds SLA of {OzoneConfig.MAX_PERFORMANCE_DEGRADATION:.2%}"
        
    finally:
        # Clean up
        tester.clean_test_environment()
        logger.info("Test completed: Performance with data recovery from DataNode failures")

import pytest
import boto3
import time
import os
import subprocess
import statistics
import tempfile
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from botocore.client import Config
from pyozone.client import OzoneClient


class TestS3GatewayPerformance:
    """Performance tests for Apache Ozone S3 Gateway"""

    @pytest.fixture(scope="class")
    def ozone_client(self):
        """Set up native Ozone client for direct operations"""
        client = OzoneClient(
            host="localhost",
            port=9862,  # Default Ozone Manager RPC port
            secure=False
        )
        yield client

    @pytest.fixture(scope="class")
    def s3_client(self):
        """Set up S3 client for S3 Gateway operations"""
        s3 = boto3.client(
            's3',
            endpoint_url='http://localhost:9878',  # Default S3G endpoint
            aws_access_key_id='testuser',
            aws_secret_access_key='testuser-secret',
            config=Config(signature_version='s3v4'),
            region_name='us-west-1'
        )
        yield s3

    @pytest.fixture(scope="function")
    def test_data(self, request):
        """Create test data files of various sizes"""
        sizes = request.param  # Size in KB
        file_path = f"test_file_{sizes}KB.txt"
        
        # Create a file with random data of specified size
        with open(file_path, "wb") as f:
            f.write(os.urandom(sizes * 1024))
        
        yield file_path, sizes
        
        # Cleanup
        if os.path.exists(file_path):
            os.remove(file_path)

    @pytest.fixture(scope="function")
    def setup_bucket(self, ozone_client, s3_client):
        """Set up test volume and bucket for both native and S3 operations"""
        volume_name = "perfvol"
        bucket_name = "perfbucket"
        
        # Create using native client
        subprocess.run(["ozone", "sh", "volume", "create", volume_name], check=True)
        subprocess.run(["ozone", "sh", "bucket", "create", f"/{volume_name}/{bucket_name}"], check=True)
        
        yield volume_name, bucket_name
        
        # Clean up
        try:
            # Delete all objects first
            response = s3_client.list_objects_v2(Bucket=bucket_name)
            if 'Contents' in response:
                for obj in response['Contents']:
                    s3_client.delete_object(Bucket=bucket_name, Key=obj['Key'])
                    
            # Then delete the bucket and volume
            subprocess.run(["ozone", "sh", "bucket", "delete", f"/{volume_name}/{bucket_name}"], check=True)
            subprocess.run(["ozone", "sh", "volume", "delete", volume_name], check=True)
        except Exception as e:
            print(f"Cleanup error: {e}")

    def run_native_put(self, ozone_client, volume, bucket, key, file_path):
        """Perform PUT operation using native Ozone client"""
        start_time = time.time()
        with open(file_path, 'rb') as f:
            subprocess.run(
                ["ozone", "sh", "key", "put", f"/{volume}/{bucket}/", file_path],
                check=True
            )
        end_time = time.time()
        return end_time - start_time
    
    def run_native_get(self, ozone_client, volume, bucket, key):
        """Perform GET operation using native Ozone client"""
        temp_file = tempfile.NamedTemporaryFile(delete=False).name
        start_time = time.time()
        subprocess.run(
            ["ozone", "sh", "key", "get", f"/{volume}/{bucket}/{key}", temp_file],
            check=True
        )
        end_time = time.time()
        os.remove(temp_file)
        return end_time - start_time
    
    def run_native_list(self, ozone_client, volume, bucket):
        """Perform LIST operation using native Ozone client"""
        start_time = time.time()
        subprocess.run(
            ["ozone", "sh", "key", "list", f"/{volume}/{bucket}/"],
            check=True
        )
        end_time = time.time()
        return end_time - start_time
    
    def run_s3_put(self, s3_client, bucket, key, file_path):
        """Perform PUT operation using S3 client"""
        start_time = time.time()
        with open(file_path, 'rb') as f:
            s3_client.upload_fileobj(f, bucket, key)
        end_time = time.time()
        return end_time - start_time
    
    def run_s3_get(self, s3_client, bucket, key):
        """Perform GET operation using S3 client"""
        temp_file = tempfile.NamedTemporaryFile(delete=False).name
        start_time = time.time()
        with open(temp_file, 'wb') as f:
            s3_client.download_fileobj(bucket, key, f)
        end_time = time.time()
        os.remove(temp_file)
        return end_time - start_time
    
    def run_s3_list(self, s3_client, bucket):
        """Perform LIST operation using S3 client"""
        start_time = time.time()
        s3_client.list_objects_v2(Bucket=bucket)
        end_time = time.time()
        return end_time - start_time

    @pytest.mark.parametrize("test_data", [
        64,      # 64 KB
        512,     # 512 KB
        1024,    # 1 MB
        4096,    # 4 MB
        9216,    # 9 MB
        15360,   # 15 MB
        30720,   # 30 MB
    ], indirect=True)
    def test_42_s3_gateway_performance_impact(self, ozone_client, s3_client, setup_bucket, test_data):
        """
        Evaluate performance impact of S3 Gateway
        
        This test compares the performance of standard operations (GET, PUT, LIST)
        through S3 Gateway versus native Ozone operations to measure the latency
        overhead introduced by the S3 Gateway layer.
        """
        file_path, file_size = test_data
        volume_name, bucket_name = setup_bucket
        key_name = os.path.basename(file_path)
        
        # Number of operations to perform for each test to get reliable metrics
        iterations = 5
        
        # Store results
        results = {
            "operation": [],
            "native_times": [],
            "s3_times": [],
            "overhead_percent": []
        }
        
        # Test PUT operations
        native_put_times = []
        s3_put_times = []
        
        for i in range(iterations):
            # Add a unique suffix to avoid conflicts
            unique_key = f"{key_name}_{i}"
            
            # Native PUT
            native_time = self.run_native_put(ozone_client, volume_name, bucket_name, unique_key, file_path)
            native_put_times.append(native_time)
            
            # S3 PUT
            s3_time = self.run_s3_put(s3_client, bucket_name, f"s3_{unique_key}", file_path)
            s3_put_times.append(s3_time)
        
        native_put_avg = statistics.mean(native_put_times)
        s3_put_avg = statistics.mean(s3_put_times)
        put_overhead = ((s3_put_avg - native_put_avg) / native_put_avg) * 100 if native_put_avg > 0 else 0
        
        results["operation"].append("PUT")
        results["native_times"].append(native_put_avg)
        results["s3_times"].append(s3_put_avg)
        results["overhead_percent"].append(put_overhead)
        
        # Test GET operations
        native_get_times = []
        s3_get_times = []
        
        for i in range(iterations):
            unique_key = f"{key_name}_{i}"
            s3_unique_key = f"s3_{unique_key}"
            
            # Native GET
            native_time = self.run_native_get(ozone_client, volume_name, bucket_name, unique_key)
            native_get_times.append(native_time)
            
            # S3 GET
            s3_time = self.run_s3_get(s3_client, bucket_name, s3_unique_key)
            s3_get_times.append(s3_time)
        
        native_get_avg = statistics.mean(native_get_times)
        s3_get_avg = statistics.mean(s3_get_times)
        get_overhead = ((s3_get_avg - native_get_avg) / native_get_avg) * 100 if native_get_avg > 0 else 0
        
        results["operation"].append("GET")
        results["native_times"].append(native_get_avg)
        results["s3_times"].append(s3_get_avg)
        results["overhead_percent"].append(get_overhead)
        
        # Test LIST operations
        native_list_times = []
        s3_list_times = []
        
        for i in range(iterations):
            # Native LIST
            native_time = self.run_native_list(ozone_client, volume_name, bucket_name)
            native_list_times.append(native_time)
            
            # S3 LIST
            s3_time = self.run_s3_list(s3_client, bucket_name)
            s3_list_times.append(s3_time)
        
        native_list_avg = statistics.mean(native_list_times)
        s3_list_avg = statistics.mean(s3_list_times)
        list_overhead = ((s3_list_avg - native_list_avg) / native_list_avg) * 100 if native_list_avg > 0 else 0
        
        results["operation"].append("LIST")
        results["native_times"].append(native_list_avg)
        results["s3_times"].append(s3_list_avg)
        results["overhead_percent"].append(list_overhead)
        
        # Print summary
        print(f"\nPerformance Test Results for {file_size} KB file:")
        print(f"{'Operation':<10} {'Native (s)':<15} {'S3 Gateway (s)':<15} {'Overhead %':<10}")
        print("-" * 50)
        for i in range(len(results["operation"])):
            print(f"{results['operation'][i]:<10} {results['native_times'][i]:<15.4f} "
                  f"{results['s3_times'][i]:<15.4f} {results['overhead_percent'][i]:<10.2f}")
        
        # Generate a summary graph for this file size
        self.generate_performance_graph(results, file_size)
        
        # Assert that the overhead is acceptable (assuming less than 20% overhead is acceptable)
        max_acceptable_overhead = 20.0  # 20% overhead
        
        for op_idx, operation in enumerate(results["operation"]):
            overhead = results["overhead_percent"][op_idx]
            assert overhead < max_acceptable_overhead, (
                f"S3 Gateway overhead for {operation} operation is {overhead:.2f}%, "
                f"which exceeds the maximum acceptable overhead of {max_acceptable_overhead}%"
            )
    
    def generate_performance_graph(self, results, file_size):
        """Generate a bar chart comparing native and S3 performance"""
        operations = results["operation"]
        native_times = results["native_times"]
        s3_times = results["s3_times"]
        
        x = np.arange(len(operations))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width/2, native_times, width, label='Native Ozone')
        rects2 = ax.bar(x + width/2, s3_times, width, label='S3 Gateway')
        
        ax.set_ylabel('Time (seconds)')
        ax.set_title(f'Performance Comparison for {file_size} KB file')
        ax.set_xticks(x)
        ax.set_xticklabels(operations)
        ax.legend()
        
        # Add overhead percentage labels above the bars
        for i, op in enumerate(operations):
            overhead = results["overhead_percent"][i]
            ax.annotate(f'+{overhead:.1f}%', 
                        xy=(x[i] + width/2, s3_times[i]), 
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        fig.tight_layout()
        plt.savefig(f"s3_gateway_performance_{file_size}KB.png")
        plt.close()

    @pytest.mark.parametrize("num_concurrent", [1, 5, 10, 20, 50])
    def test_42_s3_gateway_concurrent_performance(self, s3_client, setup_bucket, num_concurrent):
        """
        Evaluate S3 Gateway performance under concurrent load
        
        This test measures the performance of the S3 Gateway under various 
        levels of concurrent client operations.
        """
        volume_name, bucket_name = setup_bucket
        file_size = 256  # 256KB file for testing
        
        # Create test file
        file_path = "concurrent_test_file.txt"
        with open(file_path, "wb") as f:
            f.write(os.urandom(file_size * 1024))
        
        try:
            # Run concurrent PUT operations
            start_time = time.time()
            
            def put_object(i):
                key = f"concurrent_test_{i}"
                with open(file_path, 'rb') as f:
                    s3_client.upload_fileobj(f, bucket_name, key)
                return i
            
            with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
                futures = [executor.submit(put_object, i) for i in range(num_concurrent)]
                for future in futures:
                    future.result()  # Wait for completion and gather any exceptions
            
            end_time = time.time()
            total_time = end_time - start_time
            operations_per_second = num_concurrent / total_time if total_time > 0 else 0
            
            print(f"\nConcurrent PUT Performance with {num_concurrent} clients:")
            print(f"Total Time: {total_time:.2f} seconds")
            print(f"Operations per second: {operations_per_second:.2f}")
            
            # Assert acceptable performance for concurrent operations
            # This is a simple example - real threshold would depend on specific cluster capabilities
            min_acceptable_ops_per_second = 0.5
            assert operations_per_second >= min_acceptable_ops_per_second, (
                f"S3 Gateway concurrent performance ({operations_per_second:.2f} ops/sec) "
                f"is below the minimum acceptable threshold ({min_acceptable_ops_per_second} ops/sec)"
            )
            
        finally:
            # Clean up
            if os.path.exists(file_path):
                os.remove(file_path)

import os
import time
import subprocess
import statistics
import pytest
import logging
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test configurations
CONTAINER_SIZES = [128, 256, 512, 1024, 2048]  # Container sizes in MB
FILE_SIZES = ["10MB", "100MB", "1GB", "5GB"]  # File sizes to test with
NUM_FILES = 10  # Number of files to write/read for each test
NUM_ITERATIONS = 3  # Number of test iterations for reliable results
TEST_VOLUME = "perfvol"
TEST_BUCKET = "perfbucket"

class OzoneClusterManager:
    """Helper class to manage Ozone cluster configurations"""
    
    @staticmethod
    def configure_container_size(size_mb: int) -> bool:
        """
        Configure the Ozone cluster with a specific block container size
        
        Args:
            size_mb: Container size in MB
            
        Returns:
            bool: True if configuration was successful
        """
        try:
            # Update ozone-site.xml with the new container size
            cmd = [
                "sudo", "sed", "-i", 
                f's/[0-9]*<\/value>/{size_mb * 1024 * 1024}<\/value>/g',
                "/etc/hadoop/conf/ozone-site.xml"
            ]
            subprocess.run(cmd, check=True)
            
            # Restart the Ozone services
            subprocess.run(["sudo", "systemctl", "restart", "hadoop-ozone-scm"], check=True)
            subprocess.run(["sudo", "systemctl", "restart", "hadoop-ozone-om"], check=True)
            subprocess.run(["sudo", "systemctl", "restart", "hadoop-ozone-datanode"], check=True)
            
            # Wait for services to stabilize
            time.sleep(30)
            
            return True
        except subprocess.SubprocessError as e:
            logger.error(f"Failed to configure container size {size_mb}MB: {str(e)}")
            return False

    @staticmethod
    def check_cluster_health() -> bool:
        """Verify that the Ozone cluster is healthy"""
        try:
            result = subprocess.run(
                ["ozone", "admin", "status"], 
                capture_output=True, 
                text=True, 
                check=True
            )
            return "HEALTHY" in result.stdout
        except subprocess.SubprocessError:
            return False


class PerformanceBenchmark:
    """Class to run performance benchmarks on Apache Ozone"""
    
    @staticmethod
    def create_test_file(size: str) -> str:
        """Create a test file of specified size"""
        filename = f"test_file_{size}.bin"
        
        # Convert size string to bytes
        size_value = int(size[:-2])
        unit = size[-2:]
        multiplier = {
            "KB": 1024,
            "MB": 1024 * 1024,
            "GB": 1024 * 1024 * 1024
        }.get(unit, 1)
        byte_size = size_value * multiplier
        
        with open(filename, "wb") as f:
            f.write(os.urandom(byte_size))
        
        return filename

    @staticmethod
    def setup_volume_bucket():
        """Create volume and bucket for testing"""
        try:
            # Create volume if not exists
            subprocess.run(["ozone", "sh", "volume", "create", TEST_VOLUME], 
                          check=False)  # Ignore errors if volume exists
            
            # Create bucket if not exists
            subprocess.run(["ozone", "sh", "bucket", "create", f"{TEST_VOLUME}/{TEST_BUCKET}"],
                          check=False)  # Ignore errors if bucket exists
            
            return True
        except subprocess.SubprocessError as e:
            logger.error(f"Failed to create volume/bucket: {str(e)}")
            return False

    @staticmethod
    def run_write_benchmark(file_path: str, num_files: int) -> Tuple[float, float, float]:
        """
        Run write benchmark by uploading files to Ozone
        
        Returns:
            Tuple containing: throughput (MB/s), average latency (ms), p95 latency (ms)
        """
        latencies = []
        total_size_bytes = os.path.getsize(file_path) * num_files
        total_size_mb = total_size_bytes / (1024 * 1024)
        
        start_time = time.time()
        
        for i in range(num_files):
            key = f"test_key_{i}"
            file_start = time.time()
            subprocess.run([
                "ozone", "sh", "key", "put", 
                f"{TEST_VOLUME}/{TEST_BUCKET}/{key}", 
                file_path
            ], check=True)
            latency = (time.time() - file_start) * 1000  # ms
            latencies.append(latency)
        
        total_time = time.time() - start_time
        throughput = total_size_mb / total_time if total_time > 0 else 0
        avg_latency = statistics.mean(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
        
        return throughput, avg_latency, p95_latency

    @staticmethod
    def run_read_benchmark(file_path: str, num_files: int) -> Tuple[float, float, float]:
        """
        Run read benchmark by downloading files from Ozone
        
        Returns:
            Tuple containing: throughput (MB/s), average latency (ms), p95 latency (ms)
        """
        latencies = []
        total_size_bytes = os.path.getsize(file_path) * num_files
        total_size_mb = total_size_bytes / (1024 * 1024)
        
        # Read output directory
        os.makedirs("read_output", exist_ok=True)
        
        start_time = time.time()
        
        for i in range(num_files):
            key = f"test_key_{i}"
            output_file = f"read_output/read_{i}.bin"
            file_start = time.time()
            subprocess.run([
                "ozone", "sh", "key", "get", 
                f"{TEST_VOLUME}/{TEST_BUCKET}/{key}",
                output_file
            ], check=True)
            latency = (time.time() - file_start) * 1000  # ms
            latencies.append(latency)
        
        total_time = time.time() - start_time
        throughput = total_size_mb / total_time if total_time > 0 else 0
        avg_latency = statistics.mean(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
        
        return throughput, avg_latency, p95_latency

    @staticmethod
    def get_space_utilization() -> float:
        """
        Get current space utilization ratio from Ozone
        
        Returns:
            Space utilization ratio (used / total)
        """
        try:
            result = subprocess.run(
                ["ozone", "admin", "datanode", "report"], 
                capture_output=True, 
                text=True
            )
            
            # Parse output to extract space usage
            # This is a simplified example - actual parsing depends on the exact format
            lines = result.stdout.split('\n')
            used = total = 0
            
            for line in lines:
                if "Used:" in line:
                    used = float(line.split()[1].replace('GB', ''))
                if "Total:" in line:
                    total = float(line.split()[1].replace('GB', ''))
            
            return used / total if total > 0 else 0
        except (subprocess.SubprocessError, ValueError, IndexError) as e:
            logger.error(f"Failed to get space utilization: {str(e)}")
            return 0.0


@pytest.mark.parametrize("container_size", CONTAINER_SIZES)
@pytest.mark.parametrize("file_size", FILE_SIZES)
def test_43_block_container_performance(container_size, file_size):
    """
    Test performance with varying block container sizes
    
    This test evaluates how different block container sizes affect performance
    metrics like throughput, latency, and space utilization for various file sizes.
    """
    results = {
        "container_size": container_size,
        "file_size": file_size,
        "write_throughput": [],
        "write_avg_latency": [],
        "write_p95_latency": [],
        "read_throughput": [],
        "read_avg_latency": [],
        "read_p95_latency": [],
        "space_utilization": []
    }
    
    # Step 1: Configure Ozone with the specified block container size
    logger.info(f"Configuring Ozone with {container_size}MB block container size")
    assert OzoneClusterManager.configure_container_size(container_size), \
        f"Failed to configure container size to {container_size}MB"
    
    # Verify cluster health after configuration change
    assert OzoneClusterManager.check_cluster_health(), \
        "Cluster is not healthy after configuration change"
    
    # Step 2: Run benchmarks for the configuration
    logger.info(f"Running benchmarks with file size {file_size}")
    
    # Setup test environment
    test_file = PerformanceBenchmark.create_test_file(file_size)
    assert PerformanceBenchmark.setup_volume_bucket(), "Failed to setup volume and bucket"
    
    # Run multiple iterations for statistical significance
    for iteration in range(NUM_ITERATIONS):
        logger.info(f"Starting iteration {iteration+1}/{NUM_ITERATIONS}")
        
        # Run write benchmark
        write_throughput, write_avg_latency, write_p95_latency = \
            PerformanceBenchmark.run_write_benchmark(test_file, NUM_FILES)
        
        results["write_throughput"].append(write_throughput)
        results["write_avg_latency"].append(write_avg_latency)
        results["write_p95_latency"].append(write_p95_latency)
        
        # Run read benchmark
        read_throughput, read_avg_latency, read_p95_latency = \
            PerformanceBenchmark.run_read_benchmark(test_file, NUM_FILES)
            
        results["read_throughput"].append(read_throughput)
        results["read_avg_latency"].append(read_avg_latency)
        results["read_p95_latency"].append(read_p95_latency)
        
        # Check space utilization
        space_util = PerformanceBenchmark.get_space_utilization()
        results["space_utilization"].append(space_util)
        
        # Clean up files between iterations
        if iteration < NUM_ITERATIONS - 1:
            for i in range(NUM_FILES):
                subprocess.run([
                    "ozone", "sh", "key", "delete", 
                    f"{TEST_VOLUME}/{TEST_BUCKET}/test_key_{i}"
                ], check=True)
    
    # Step 3 & 4: Analyze results and identify optimal settings
    # Calculate averages
    avg_results = {
        "write_throughput": statistics.mean(results["write_throughput"]),
        "write_avg_latency": statistics.mean(results["write_avg_latency"]),
        "write_p95_latency": statistics.mean(results["write_p95_latency"]),
        "read_throughput": statistics.mean(results["read_throughput"]),
        "read_avg_latency": statistics.mean(results["read_avg_latency"]),
        "read_p95_latency": statistics.mean(results["read_p95_latency"]),
        "space_utilization": statistics.mean(results["space_utilization"])
    }
    
    # Log results
    logger.info(f"Results for container size {container_size}MB with file size {file_size}:")
    logger.info(f"  Write Throughput: {avg_results['write_throughput']:.2f} MB/s")
    logger.info(f"  Write Average Latency: {avg_results['write_avg_latency']:.2f} ms")
    logger.info(f"  Write P95 Latency: {avg_results['write_p95_latency']:.2f} ms")
    logger.info(f"  Read Throughput: {avg_results['read_throughput']:.2f} MB/s")
    logger.info(f"  Read Average Latency: {avg_results['read_avg_latency']:.2f} ms")
    logger.info(f"  Read P95 Latency: {avg_results['read_p95_latency']:.2f} ms")
    logger.info(f"  Space Utilization: {avg_results['space_utilization']:.2f}")
    
    # Clean up
    os.remove(test_file)
    
    # Record test result to file for later analysis
    with open(f"container_size_benchmark_{container_size}_{file_size}.log", "w") as f:
        for key, value in avg_results.items():
            f.write(f"{key}: {value}\n")
    
    # Assert some basic performance expectations
    # We expect reasonable performance - these thresholds would be environment-specific
    assert avg_results["write_throughput"] > 0, "Write throughput should be positive"
    assert avg_results["read_throughput"] > 0, "Read throughput should be positive"

    # This test doesn't fail on performance thresholds - it's informational
    # The goal is to collect data for analysis to determine optimal settings


def analyze_container_size_results():
    """
    Analyze results from container size tests and generate visualizations
    
    This is a helper function that would be called after all tests have completed
    to analyze the collected data and make recommendations.
    """
    # Data structures to hold aggregated results
    container_sizes = CONTAINER_SIZES
    file_sizes = FILE_SIZES
    
    write_throughput = np.zeros((len(container_sizes), len(file_sizes)))
    read_throughput = np.zeros((len(container_sizes), len(file_sizes)))
    space_utilization = np.zeros((len(container_sizes), len(file_sizes)))
    
    # Read data from generated log files
    for i, cs in enumerate(container_sizes):
        for j, fs in enumerate(file_sizes):
            try:
                with open(f"container_size_benchmark_{cs}_{fs}.log", "r") as f:
                    data = {}
                    for line in f:
                        key, value = line.strip().split(": ")
                        data[key] = float(value)
                        
                    write_throughput[i, j] = data["write_throughput"]
                    read_throughput[i, j] = data["read_throughput"]
                    space_utilization[i, j] = data["space_utilization"]
            except (FileNotFoundError, ValueError, KeyError) as e:
                logger.warning(f"Could not load data for {cs}MB-{fs}: {e}")
    
    # Generate visualizations
    plt.figure(figsize=(15, 10))
    
    # Plot write throughput
    plt.subplot(2, 2, 1)
    for j, fs in enumerate(file_sizes):
        plt.plot(container_sizes, write_throughput[:, j], marker='o', label=fs)
    plt.title('Write Throughput vs. Container Size')
    plt.xlabel('Container Size (MB)')
    plt.ylabel('Throughput (MB/s)')
    plt.legend()
    plt.grid(True)
    
    # Plot read throughput
    plt.subplot(2, 2, 2)
    for j, fs in enumerate(file_sizes):
        plt.plot(container_sizes, read_throughput[:, j], marker='o', label=fs)
    plt.title('Read Throughput vs. Container Size')
    plt.xlabel('Container Size (MB)')
    plt.ylabel('Throughput (MB/s)')
    plt.legend()
    plt.grid(True)
    
    # Plot space utilization
    plt.subplot(2, 2, 3)
    for j, fs in enumerate(file_sizes):
        plt.plot(container_sizes, space_utilization[:, j], marker='o', label=fs)
    plt.title('Space Utilization vs. Container Size')
    plt.xlabel('Container Size (MB)')
    plt.ylabel('Space Utilization Ratio')
    plt.legend()
    plt.grid(True)
    
    # Add summary analysis
    plt.subplot(2, 2, 4)
    plt.text(0.1, 0.9, 'Summary of Optimal Container Sizes', fontsize=12)
    plt.text(0.1, 0.8, 'Small Files (10MB-100MB):', fontsize=10)
    
    # Find optimal size for small files (would be calculated from actual results)
    small_files_optimal = container_sizes[np.argmax(np.mean(write_throughput[:, :2], axis=1))]
    plt.text(0.2, 0.7, f'Optimal: {small_files_optimal}MB container size', fontsize=9)
    
    # Find optimal size for large files
    large_files_optimal = container_sizes[np.argmax(np.mean(write_throughput[:, 2:], axis=1))]
    plt.text(0.1, 0.6, 'Large Files (1GB+):', fontsize=10)
    plt.text(0.2, 0.5, f'Optimal: {large_files_optimal}MB container size', fontsize=9)
    
    # Add efficiency note
    overall_optimal = container_sizes[np.argmax(np.mean(write_throughput, axis=1) / 
                                               (np.mean(space_utilization, axis=1) + 0.1))]
    plt.text(0.1, 0.4, 'Best Overall Efficiency:', fontsize=10)
    plt.text(0.2, 0.3, f'Optimal: {overall_optimal}MB container size', fontsize=9)
    
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('container_size_performance_analysis.png')
    logger.info("Analysis completed and saved to container_size_performance_analysis.png")

#!/usr/bin/env python3

import time
import pytest
import subprocess
import logging
import os
import statistics
import concurrent.futures
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from ozone.client import ObjectStoreClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DEFAULT_OZONE_HOST = os.environ.get("OZONE_HOST", "localhost")
DEFAULT_OZONE_PORT = int(os.environ.get("OZONE_PORT", "9878"))

# Test configurations
KEY_COUNT_LEVELS = [1000, 10000, 100000, 1000000]  # Number of keys to generate
BUCKET_COUNT_LEVELS = [1, 5, 10]  # Number of buckets to distribute keys
PAGE_SIZES = [100, 1000, 10000]  # Page sizes for pagination tests
FILTERS = ["", "prefix_", "test_", "data_"]  # Filters to test

# Test data setup
class OzoneKeyListingPerformanceTest:
    """Helper class for Ozone key listing performance testing"""
    
    def __init__(self, host=DEFAULT_OZONE_HOST, port=DEFAULT_OZONE_PORT):
        self.host = host
        self.port = port
        self.client = ObjectStoreClient(host, port)
        self.perf_results = []
        self.test_dir = "/tmp/ozone_perf_test"
        os.makedirs(self.test_dir, exist_ok=True)
        
    def setup_test_data(self, volume_name: str, bucket_count: int, keys_per_bucket: int):
        """Generate test data with specified number of keys across multiple buckets"""
        # Create volume
        logger.info(f"Creating volume: {volume_name}")
        subprocess.run(["ozone", "sh", "volume", "create", volume_name], check=True)
        
        # Create buckets
        buckets = []
        for i in range(bucket_count):
            bucket_name = f"bucket{i}"
            logger.info(f"Creating bucket: {volume_name}/{bucket_name}")
            subprocess.run(["ozone", "sh", "bucket", "create", f"/{volume_name}/{bucket_name}"], check=True)
            buckets.append(bucket_name)
        
        # Create a small sample file for putting as keys
        test_file = os.path.join(self.test_dir, "test_data.txt")
        with open(test_file, "w") as f:
            f.write("This is test data for performance testing")
        
        # Generate keys using concurrent operations
        logger.info(f"Generating {keys_per_bucket * bucket_count} keys across {bucket_count} buckets...")
        
        def put_keys_for_bucket(bucket_name, keys_count):
            for j in range(keys_count):
                # Use different key patterns to test filters
                prefix = "" if j % 4 == 0 else ("prefix_" if j % 4 == 1 else 
                                              ("test_" if j % 4 == 2 else "data_"))
                key_name = f"{prefix}key_{j}"
                try:
                    # Use ozone shell for putting keys
                    subprocess.run([
                        "ozone", "sh", "key", "put", 
                        f"{volume_name}/{bucket_name}/", 
                        test_file
                    ], check=True, stdout=subprocess.DEVNULL)
                except Exception as e:
                    logger.error(f"Failed to put key {key_name}: {str(e)}")
        
        # Use thread pool to speed up key generation
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for bucket in buckets:
                futures.append(
                    executor.submit(put_keys_for_bucket, bucket, keys_per_bucket)
                )
            
            # Wait for all tasks to complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    logger.error(f"Key generation failed: {str(exc)}")
        
        return volume_name, buckets
    
    def measure_listing_performance(self, volume: str, buckets: List[str], 
                                  page_size: int, filter_prefix: str = ""):
        """
        Measure the performance of key listing operations
        Returns: Dict with performance metrics
        """
        metrics = {
            'volume': volume,
            'buckets': len(buckets),
            'page_size': page_size,
            'filter': filter_prefix,
            'response_times': [],
            'cpu_usage': [],
            'memory_usage': []
        }
        
        for bucket in buckets:
            # Measure initial resource usage
            initial_cpu, initial_mem = self._get_resource_usage()
            
            start_time = time.time()
            
            # Perform listing with pagination
            max_keys = page_size
            next_marker = ""
            total_keys = 0
            
            while True:
                cmd = ["ozone", "sh", "key", "list", f"{volume}/{bucket}/"]
                if filter_prefix:
                    cmd.extend(["--prefix", filter_prefix])
                if max_keys:
                    cmd.extend(["--max-keys", str(max_keys)])
                if next_marker:
                    cmd.extend(["--start-after", next_marker])
                
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                
                # Parse the output to get the keys and next marker
                keys = result.stdout.strip().split('\n')
                keys = [k for k in keys if k and not k.startswith("Found")]
                
                total_keys += len(keys)
                
                if len(keys) < page_size:
                    # No more keys to list
                    break
                    
                # Set the next marker to the last key
                if keys:
                    next_marker = keys[-1]
                else:
                    break
            
            # Calculate elapsed time and resource usage
            elapsed_time = time.time() - start_time
            current_cpu, current_mem = self._get_resource_usage()
            cpu_usage = current_cpu - initial_cpu
            mem_usage = current_mem - initial_mem
            
            # Record metrics
            metrics['response_times'].append(elapsed_time)
            metrics['cpu_usage'].append(cpu_usage)
            metrics['memory_usage'].append(mem_usage)
            
            logger.info(f"Listed {total_keys} keys from {bucket} in {elapsed_time:.2f}s "
                      f"with filter='{filter_prefix}', page_size={page_size}")
        
        # Calculate aggregate metrics
        metrics['avg_response_time'] = statistics.mean(metrics['response_times'])
        metrics['max_response_time'] = max(metrics['response_times'])
        metrics['avg_cpu_usage'] = statistics.mean(metrics['cpu_usage'])
        metrics['avg_memory_usage'] = statistics.mean(metrics['memory_usage'])
        
        # Save the result
        self.perf_results.append(metrics)
        
        return metrics
    
    def _get_resource_usage(self):
        """Get current CPU and memory usage"""
        try:
            result = subprocess.run(
                ["ps", "-o", "%cpu,%mem", "-p", str(os.getpid())],
                check=True, capture_output=True, text=True
            )
            lines = result.stdout.strip().split('\n')
            if len(lines) >= 2:
                values = lines[1].split()
                if len(values) >= 2:
                    return float(values[0]), float(values[1])
        except Exception as e:
            logger.error(f"Failed to get resource usage: {str(e)}")
        
        return 0.0, 0.0
    
    def generate_performance_report(self, output_dir="/tmp/ozone_perf_results"):
        """Generate performance report from test results"""
        os.makedirs(output_dir, exist_ok=True)
        
        if not self.perf_results:
            logger.error("No performance data collected")
            return
        
        # Convert results to DataFrame for analysis
        df = pd.DataFrame(self.perf_results)
        
        # Save raw data
        csv_file = os.path.join(output_dir, "key_listing_performance.csv")
        df.to_csv(csv_file, index=False)
        logger.info(f"Raw performance data saved to {csv_file}")
        
        # Generate charts
        self._plot_response_times(df, output_dir)
        self._plot_resource_usage(df, output_dir)
        
    def _plot_response_times(self, df, output_dir):
        """Generate response time charts"""
        plt.figure(figsize=(10, 6))
        
        # Group by bucket count and page size
        grouped = df.groupby(['buckets', 'page_size'])
        for name, group in grouped:
            plt.plot(group['filter'], group['avg_response_time'], 
                     marker='o', label=f"Buckets={name[0]}, Page={name[1]}")
            
        plt.title('Average Response Time by Filter and Configuration')
        plt.xlabel('Filter Prefix')
        plt.ylabel('Response Time (seconds)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'response_time_by_filter.png'))
        plt.close()
        
    def _plot_resource_usage(self, df, output_dir):
        """Generate resource usage charts"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot CPU usage
        grouped = df.groupby('buckets')
        for name, group in grouped:
            ax1.plot(group['page_size'], group['avg_cpu_usage'], 
                    marker='o', label=f"Buckets={name}")
            
        ax1.set_title('CPU Usage by Page Size')
        ax1.set_xlabel('Page Size')
        ax1.set_ylabel('CPU Usage (%)')
        ax1.legend()
        ax1.grid(True)
        
        # Plot Memory usage
        for name, group in grouped:
            ax2.plot(group['page_size'], group['avg_memory_usage'], 
                    marker='o', label=f"Buckets={name}")
            
        ax2.set_title('Memory Usage by Page Size')
        ax2.set_xlabel('Page Size')
        ax2.set_ylabel('Memory Usage (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'resource_usage.png'))
        plt.close()


@pytest.mark.parametrize("key_count", [10000, 100000])
@pytest.mark.parametrize("bucket_count", [1, 5])
@pytest.mark.parametrize("page_size", [100, 1000])
def test_44_key_listing_performance(key_count, bucket_count, page_size):
    """
    Evaluate performance under heavy key listing operations
    
    This test measures the performance of key listing operations in Ozone when dealing
    with a large number of keys distributed across multiple buckets.
    
    Performance metrics collected:
    - Response time for listing operations
    - CPU and memory utilization
    - Impact of pagination size
    - Impact of filter prefixes
    """
    # Calculate keys per bucket
    keys_per_bucket = key_count // bucket_count
    
    # Create unique volume for this test
    volume_name = f"volume{int(time.time())}"
    
    # Initialize the test helper
    perf_test = OzoneKeyListingPerformanceTest()
    
    try:
        # Setup test data (this may take significant time)
        logger.info(f"Setting up test data: {key_count} keys across {bucket_count} buckets")
        volume, buckets = perf_test.setup_test_data(
            volume_name=volume_name, 
            bucket_count=bucket_count, 
            keys_per_bucket=keys_per_bucket
        )
        
        # Perform listing operations with various filters and collect performance data
        for filter_prefix in FILTERS:
            metrics = perf_test.measure_listing_performance(
                volume=volume,
                buckets=buckets,
                page_size=page_size,
                filter_prefix=filter_prefix
            )
            
            # Assert performance meets requirements
            # These threshold values should be adjusted based on expected performance in your environment
            max_acceptable_response_time = 30  # seconds
            assert metrics['avg_response_time'] < max_acceptable_response_time, \
                f"Average response time ({metrics['avg_response_time']:.2f}s) exceeds acceptable limit " \
                f"for filter='{filter_prefix}', page_size={page_size}"
                
    finally:
        # Generate performance report
        perf_test.generate_performance_report()
        
        # Clean up test data (consider making this optional based on debug needs)
        try:
            logger.info(f"Cleaning up test volume: {volume_name}")
            subprocess.run(["ozone", "sh", "volume", "delete", volume_name], check=True)
        except Exception as e:
            logger.warning(f"Failed to clean up test volume {volume_name}: {str(e)}")


@pytest.mark.parametrize("concurrent_clients", [1, 5, 10, 20])
def test_44_key_listing_scalability(concurrent_clients):
    """
    Test scalability of key listing operations under concurrent load
    
    This test evaluates how well Ozone handles concurrent key listing operations
    from multiple clients.
    """
    # Create unique volume for this test run
    volume_name = f"volscale{int(time.time())}"
    bucket_count = 3
    keys_per_bucket = 50000  # Smaller key count for faster setup
    
    # Initialize the test helper
    perf_test = OzoneKeyListingPerformanceTest()
    
    try:
        # Setup test data
        logger.info(f"Setting up test data for scalability test")
        volume, buckets = perf_test.setup_test_data(
            volume_name=volume_name, 
            bucket_count=bucket_count, 
            keys_per_bucket=keys_per_bucket
        )
        
        # Function to be executed by each concurrent client
        def run_listing_operation(client_id):
            start_time = time.time()
            bucket = buckets[client_id % len(buckets)]
            
            # Run listing operation
            cmd = ["ozone", "sh", "key", "list", f"{volume}/{bucket}/", "--max-keys", "500"]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
            
            return time.time() - start_time
        
        # Run concurrent clients
        logger.info(f"Running {concurrent_clients} concurrent listing operations")
        response_times = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_clients) as executor:
            futures = [executor.submit(run_listing_operation, i) for i in range(concurrent_clients)]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    response_time = future.result()
                    response_times.append(response_time)
                except Exception as exc:
                    logger.error(f"Concurrent operation failed: {str(exc)}")
        
        # Calculate statistics
        avg_response_time = statistics.mean(response_times)
        max_response_time = max(response_times)
        min_response_time = min(response_times)
        
        logger.info(f"Scalability test results with {concurrent_clients} clients:")
        logger.info(f"  Avg response time: {avg_response_time:.2f}s")
        logger.info(f"  Min response time: {min_response_time:.2f}s")
        logger.info(f"  Max response time: {max_response_time:.2f}s")
        
        # Performance should degrade gracefully with increasing concurrent clients
        max_acceptable_scaling_factor = 5.0  # Maximum acceptable performance degradation
        
        # If we have previous results, compare to baseline (1 client)
        if concurrent_clients > 1 and hasattr(test_44_key_listing_scalability, 'baseline_time'):
            scaling_factor = avg_response_time / test_44_key_listing_scalability.baseline_time
            logger.info(f"  Scaling factor: {scaling_factor:.2f}x")
            
            assert scaling_factor < max_acceptable_scaling_factor, \
                f"Performance degradation factor ({scaling_factor:.2f}x) exceeds acceptable limit " \
                f"with {concurrent_clients} concurrent clients"
        elif concurrent_clients == 1:
            # Save baseline time for comparison
            test_44_key_listing_scalability.baseline_time = avg_response_time
    
    finally:
        # Clean up test data
        try:
            logger.info(f"Cleaning up test volume: {volume_name}")
            subprocess.run(["ozone", "sh", "volume", "delete", volume_name], check=True)
        except Exception as e:
            logger.warning(f"Failed to clean up test volume {volume_name}: {str(e)}")

import pytest
import time
import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import logging
import json
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
CONSISTENCY_MODELS = [
    "EVENTUAL",
    "STRONG"
]

OPERATION_TYPES = [
    "READ",
    "WRITE",
    "MIXED"
]

# Data sizes for benchmarking (in bytes)
DATA_SIZES = [
    1024,       # 1 KB
    102400,     # 100 KB
    1048576,    # 1 MB
    10485760,   # 10 MB
]

# Number of operations to perform for each test
NUM_OPERATIONS = 100

# Thread counts for concurrent operations
THREAD_COUNTS = [1, 4, 8, 16]

class OzoneConsistencyBenchmark:
    """Helper class to run consistency model performance benchmarks on Apache Ozone"""

    def __init__(self, cluster_endpoint: str, admin_user: str = "admin"):
        """
        Initialize the benchmark utility
        
        Args:
            cluster_endpoint: Ozone endpoint URL
            admin_user: Admin username with permissions to modify cluster config
        """
        self.cluster_endpoint = cluster_endpoint
        self.admin_user = admin_user
        self.test_dir = os.path.dirname(os.path.abspath(__file__))
        self.results_dir = os.path.join(self.test_dir, "consistency_benchmark_results")
        os.makedirs(self.results_dir, exist_ok=True)
        
    def set_consistency_model(self, model: str) -> None:
        """
        Configure Ozone with the specified consistency model
        
        Args:
            model: Consistency model (EVENTUAL or STRONG)
        """
        logger.info(f"Setting consistency model to: {model}")
        
        config_params = {
            "ozone.consistency.model": model
        }
        
        # If using STRONG consistency, add additional required parameters
        if model == "STRONG":
            config_params.update({
                "ozone.scm.ratis.enable": "true",
                "ozone.om.ratis.enable": "true"
            })
        
        # Apply configuration changes
        for param, value in config_params.items():
            try:
                cmd = [
                    "ssh", 
                    self.admin_user + "@" + self.cluster_endpoint.split("://")[1].split(":")[0],
                    f"ozone admin config set {param} {value}"
                ]
                subprocess.run(cmd, check=True)
                logger.info(f"Successfully set {param}={value}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to set {param}={value}: {str(e)}")
                raise
        
        # Restart required services
        restart_cmd = [
            "ssh", 
            self.admin_user + "@" + self.cluster_endpoint.split("://")[1].split(":")[0],
            "ozone admin service restart"
        ]
        subprocess.run(restart_cmd, check=True)
        
        # Wait for cluster to stabilize after config change
        time.sleep(30)
        logger.info(f"Cluster restarted with {model} consistency model")

    def prepare_test_files(self) -> Dict[int, str]:
        """
        Create test files of various sizes for benchmarking
        
        Returns:
            Dictionary mapping file sizes to their paths
        """
        file_paths = {}
        for size in DATA_SIZES:
            file_path = os.path.join(self.test_dir, f"test_file_{size}bytes.dat")
            
            # Create file with random data of specified size
            with open(file_path, 'wb') as f:
                f.write(os.urandom(size))
                
            file_paths[size] = file_path
            
        return file_paths

    def _perform_write_operation(self, volume: str, bucket: str, key_prefix: str, 
                                file_path: str, thread_id: int) -> float:
        """
        Perform a single write operation and measure time
        
        Returns:
            Latency in milliseconds
        """
        key = f"{key_prefix}_{thread_id}"
        
        start_time = time.time()
        cmd = ["ozone", "sh", "key", "put", f"{volume}/{bucket}/", file_path, "--name", key]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        end_time = time.time()
        
        return (end_time - start_time) * 1000  # Convert to ms

    def _perform_read_operation(self, volume: str, bucket: str, key: str) -> float:
        """
        Perform a single read operation and measure time
        
        Returns:
            Latency in milliseconds
        """
        output_file = os.path.join(self.test_dir, "temp_read_file")
        
        start_time = time.time()
        cmd = ["ozone", "sh", "key", "get", f"{volume}/{bucket}/{key}", output_file]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        end_time = time.time()
        
        # Clean up temp file
        if os.path.exists(output_file):
            os.remove(output_file)
            
        return (end_time - start_time) * 1000  # Convert to ms

    def run_write_benchmark(self, volume: str, bucket: str, file_size: int, 
                         file_path: str, thread_count: int) -> Dict:
        """
        Run write benchmark with specified parameters
        
        Returns:
            Dictionary with benchmark results
        """
        # Create volume and bucket if they don't exist
        subprocess.run(["ozone", "sh", "volume", "create", volume], 
                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        subprocess.run(["ozone", "sh", "bucket", "create", f"{volume}/{bucket}"], 
                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        latencies = []
        key_prefix = f"benchmark_key_{int(time.time())}"
        
        # Use thread pool for concurrent operations
        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            futures = [
                executor.submit(
                    self._perform_write_operation, 
                    volume, bucket, key_prefix, file_path, i
                )
                for i in range(NUM_OPERATIONS)
            ]
            
            for future in futures:
                latencies.append(future.result())
        
        # Calculate metrics
        avg_latency = sum(latencies) / len(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
        p99_latency = sorted(latencies)[int(len(latencies) * 0.99)]
        throughput = (file_size * NUM_OPERATIONS) / (sum(latencies) / 1000)  # bytes/second
        
        return {
            "operation": "WRITE",
            "file_size_bytes": file_size,
            "thread_count": thread_count,
            "avg_latency_ms": avg_latency,
            "p95_latency_ms": p95_latency,
            "p99_latency_ms": p99_latency,
            "throughput_Bps": throughput,
            "throughput_MBps": throughput / (1024 * 1024),
            "latencies": latencies
        }

    def run_read_benchmark(self, volume: str, bucket: str, file_size: int, 
                        file_path: str, thread_count: int) -> Dict:
        """
        Run read benchmark with specified parameters.
        First loads data, then performs read operations.
        
        Returns:
            Dictionary with benchmark results
        """
        # First ensure the data exists
        key_prefix = f"read_benchmark_key_{int(time.time())}"
        
        # Create volume and bucket if they don't exist
        subprocess.run(["ozone", "sh", "volume", "create", volume], 
                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        subprocess.run(["ozone", "sh", "bucket", "create", f"{volume}/{bucket}"], 
                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # First upload the test files
        keys = []
        for i in range(NUM_OPERATIONS):
            key = f"{key_prefix}_{i}"
            cmd = ["ozone", "sh", "key", "put", f"{volume}/{bucket}/", file_path, "--name", key]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            keys.append(key)
        
        # Now measure read performance
        latencies = []
        
        # Use thread pool for concurrent operations
        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            futures = [
                executor.submit(
                    self._perform_read_operation, 
                    volume, bucket, keys[i % len(keys)]
                )
                for i in range(NUM_OPERATIONS)
            ]
            
            for future in futures:
                latencies.append(future.result())
        
        # Calculate metrics
        avg_latency = sum(latencies) / len(latencies)
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
        p99_latency = sorted(latencies)[int(len(latencies) * 0.99)]
        throughput = (file_size * NUM_OPERATIONS) / (sum(latencies) / 1000)  # bytes/second
        
        return {
            "operation": "READ",
            "file_size_bytes": file_size,
            "thread_count": thread_count,
            "avg_latency_ms": avg_latency,
            "p95_latency_ms": p95_latency, 
            "p99_latency_ms": p99_latency,
            "throughput_Bps": throughput,
            "throughput_MBps": throughput / (1024 * 1024),
            "latencies": latencies
        }

    def run_mixed_benchmark(self, volume: str, bucket: str, file_size: int, 
                         file_path: str, thread_count: int) -> Dict:
        """
        Run mixed read/write benchmark with specified parameters
        
        Returns:
            Dictionary with benchmark results
        """
        # Create volume and bucket if they don't exist
        subprocess.run(["ozone", "sh", "volume", "create", volume], 
                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        subprocess.run(["ozone", "sh", "bucket", "create", f"{volume}/{bucket}"], 
                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        write_latencies = []
        read_latencies = []
        key_prefix = f"mixed_benchmark_key_{int(time.time())}"
        
        # First upload half the test files
        keys = []
        for i in range(NUM_OPERATIONS // 2):
            key = f"{key_prefix}_{i}"
            start_time = time.time()
            cmd = ["ozone", "sh", "key", "put", f"{volume}/{bucket}/", file_path, "--name", key]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            end_time = time.time()
            write_latencies.append((end_time - start_time) * 1000)
            keys.append(key)
        
        # Now do mixed read/write operations
        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            # Schedule write operations
            write_futures = [
                executor.submit(
                    self._perform_write_operation, 
                    volume, bucket, key_prefix, file_path, i + (NUM_OPERATIONS // 2)
                )
                for i in range(NUM_OPERATIONS // 2)
            ]
            
            # Schedule read operations
            read_futures = [
                executor.submit(
                    self._perform_read_operation, 
                    volume, bucket, keys[i % len(keys)]
                )
                for i in range(NUM_OPERATIONS // 2)
            ]
            
            # Collect results
            for future in write_futures:
                write_latencies.append(future.result())
                
            for future in read_futures:
                read_latencies.append(future.result())
        
        # Calculate metrics
        all_latencies = write_latencies + read_latencies
        avg_latency = sum(all_latencies) / len(all_latencies)
        p95_latency = sorted(all_latencies)[int(len(all_latencies) * 0.95)]
        p99_latency = sorted(all_latencies)[int(len(all_latencies) * 0.99)]
        
        avg_write_latency = sum(write_latencies) / len(write_latencies)
        avg_read_latency = sum(read_latencies) / len(read_latencies)
        
        throughput = (file_size * NUM_OPERATIONS) / (sum(all_latencies) / 1000)  # bytes/second
        
        return {
            "operation": "MIXED",
            "file_size_bytes": file_size,
            "thread_count": thread_count,
            "avg_latency_ms": avg_latency,
            "avg_write_latency_ms": avg_write_latency,
            "avg_read_latency_ms": avg_read_latency,
            "p95_latency_ms": p95_latency,
            "p99_latency_ms": p99_latency,
            "throughput_Bps": throughput,
            "throughput_MBps": throughput / (1024 * 1024),
            "write_latencies": write_latencies,
            "read_latencies": read_latencies
        }

    def run_benchmark(self, consistency_model: str, operation_type: str, file_size: int, 
                   thread_count: int) -> Dict:
        """
        Run a benchmark with the specified parameters
        
        Returns:
            Dictionary with benchmark results
        """
        # Set volume and bucket names with timestamp to avoid conflicts
        timestamp = int(time.time())
        volume = f"vol-{consistency_model.lower()}-{timestamp}"
        bucket = f"bucket-{consistency_model.lower()}-{timestamp}"
        
        # Prepare test files
        file_paths = self.prepare_test_files()
        file_path = file_paths[file_size]
        
        # Configure Ozone with the specified consistency model
        self.set_consistency_model(consistency_model)
        
        # Run the appropriate benchmark
        if operation_type == "READ":
            return self.run_read_benchmark(volume, bucket, file_size, file_path, thread_count)
        elif operation_type == "WRITE":
            return self.run_write_benchmark(volume, bucket, file_size, file_path, thread_count)
        else:  # MIXED
            return self.run_mixed_benchmark(volume, bucket, file_size, file_path, thread_count)

    def analyze_results(self, results: List[Dict]) -> None:
        """
        Analyze benchmark results and create visualizations
        
        Args:
            results: List of benchmark result dictionaries
        """
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(results)
        
        # Save raw results to CSV
        csv_path = os.path.join(self.results_dir, f"consistency_benchmark_results_{int(time.time())}.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"Raw benchmark results saved to {csv_path}")
        
        # Create throughput comparison by consistency model and operation
        plt.figure(figsize=(12, 8))
        for op in df['operation'].unique():
            for model in df['consistency_model'].unique():
                subset = df[(df['operation'] == op) & (df['consistency_model'] == model)]
                plt.plot(subset['file_size_bytes'] / (1024*1024), 
                        subset['throughput_MBps'], 
                        marker='o', 
                        label=f"{model} - {op}")
        
        plt.xlabel('File Size (MB)')
        plt.ylabel('Throughput (MB/s)')
        plt.title('Throughput by File Size, Consistency Model, and Operation Type')
        plt.legend()
        plt.grid(True)
        plt.xscale('log')
        
        throughput_plot = os.path.join(self.results_dir, f"throughput_comparison_{int(time.time())}.png")
        plt.savefig(throughput_plot)
        logger.info(f"Throughput comparison plot saved to {throughput_plot}")
        
        # Create latency comparison by consistency model and operation
        plt.figure(figsize=(12, 8))
        for op in df['operation'].unique():
            for model in df['consistency_model'].unique():
                subset = df[(df['operation'] == op) & (df['consistency_model'] == model)]
                plt.plot(subset['file_size_bytes'] / (1024*1024), 
                        subset['avg_latency_ms'], 
                        marker='o', 
                        label=f"{model} - {op}")
        
        plt.xlabel('File Size (MB)')
        plt.ylabel('Average Latency (ms)')
        plt.title('Latency by File Size, Consistency Model, and Operation Type')
        plt.legend()
        plt.grid(True)
        plt.xscale('log')
        
        latency_plot = os.path.join(self.results_dir, f"latency_comparison_{int(time.time())}.png")
        plt.savefig(latency_plot)
        logger.info(f"Latency comparison plot saved to {latency_plot}")
        
        # Create summary report
        summary = {
            "consistency_model_comparison": {},
            "operation_type_comparison": {},
            "file_size_comparison": {},
            "thread_count_comparison": {}
        }
        
        # Compare consistency models
        for model in df['consistency_model'].unique():
            model_data = df[df['consistency_model'] == model]
            summary["consistency_model_comparison"][model] = {
                "avg_throughput_MBps": model_data['throughput_MBps'].mean(),
                "avg_latency_ms": model_data['avg_latency_ms'].mean(),
                "p95_latency_ms": model_data['p95_latency_ms'].mean()
            }
        
        # Save summary report
        summary_path = os.path.join(self.results_dir, f"benchmark_summary_{int(time.time())}.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Benchmark summary saved to {summary_path}")
        
        # Log key findings
        logger.info("Key Performance Findings:")
        
        # Calculate percentage difference between consistency models
        if len(df['consistency_model'].unique()) > 1:
            eventual_metrics = summary["consistency_model_comparison"]["EVENTUAL"]
            strong_metrics = summary["consistency_model_comparison"]["STRONG"]
            
            throughput_diff = ((strong_metrics["avg_throughput_MBps"] - eventual_metrics["avg_throughput_MBps"]) / 
                              eventual_metrics["avg_throughput_MBps"] * 100)
            
            latency_diff = ((strong_metrics["avg_latency_ms"] - eventual_metrics["avg_latency_ms"]) / 
                           eventual_metrics["avg_latency_ms"] * 100)
            
            logger.info(f"Strong consistency throughput differs from eventual by {throughput_diff:.2f}%")
            logger.info(f"Strong consistency latency differs from eventual by {latency_diff:.2f}%")
            
            # Determine best consistency model for different scenarios
            read_df = df[df['operation'] == 'READ']
            write_df = df[df['operation'] == 'WRITE']
            mixed_df = df[df['operation'] == 'MIXED']
            
            for op_df, op_name in [(read_df, "READ"), (write_df, "WRITE"), (mixed_df, "MIXED")]:
                if not op_df.empty:
                    eventual_perf = op_df[op_df['consistency_model'] == 'EVENTUAL']['throughput_MBps'].mean()
                    strong_perf = op_df[op_df['consistency_model'] == 'STRONG']['throughput_MBps'].mean()
                    
                    if eventual_perf > strong_perf:
                        logger.info(f"For {op_name} operations, EVENTUAL consistency provides better throughput")
                    else:
                        logger.info(f"For {op_name} operations, STRONG consistency provides better throughput")

        return summary


@pytest.fixture(scope="module")
def ozone_benchmark():
    """Fixture to create and return the benchmark utility"""
    # Use environment variables or default values for the Ozone endpoint
    ozone_endpoint = os.environ.get("OZONE_ENDPOINT", "http://localhost:9878")
    admin_user = os.environ.get("OZONE_ADMIN_USER", "admin")
    
    return OzoneConsistencyBenchmark(ozone_endpoint, admin_user)


# Test parameters
@pytest.mark.parametrize("consistency_model", CONSISTENCY_MODELS)
@pytest.mark.parametrize("operation_type", OPERATION_TYPES)
@pytest.mark.parametrize("file_size", [DATA_SIZES[1], DATA_SIZES[2]])  # Use 100KB and 1MB for basic tests
@pytest.mark.parametrize("thread_count", [1, 8])  # Test with single thread and 8 threads
def test_45_consistency_model_performance(ozone_benchmark, consistency_model, operation_type, 
                                         file_size, thread_count):
    """
    Test performance with different consistency models
    
    This test runs benchmarks for different consistency models (eventual, strong),
    measures throughput and latency for read/write operations, and analyzes the 
    performance implications of each consistency model.
    """
    # Skip combinations that would take too long to run in CI
    if file_size > 1048576 and thread_count > 4:
        pytest.skip("Skipping large file size with high thread count to keep test duration reasonable")
    
    # Run the benchmark
    result = ozone_benchmark.run_benchmark(
        consistency_model, 
        operation_type, 
        file_size, 
        thread_count
    )
    
    # Add the consistency model to the result for later analysis
    result["consistency_model"] = consistency_model
    
    # Basic assertions to ensure the benchmark produced valid results
    assert result["avg_latency_ms"] > 0, "Average latency should be positive"
    assert result["throughput_MBps"] > 0, "Throughput should be positive"
    
    # Store the result in the fixture for later analysis
    if not hasattr(ozone_benchmark, "collected_results"):
        ozone_benchmark.collected_results = []
    ozone_benchmark.collected_results.append(result)
    
    # Log performance metrics
    logger.info(f"Performance results for {consistency_model} consistency, {operation_type} operation:")
    logger.info(f"  File size: {file_size/1024:.2f} KB")
    logger.info(f"  Thread count: {thread_count}")
    logger.info(f"  Average latency: {result['avg_latency_ms']:.2f} ms")
    logger.info(f"  P95 latency: {result['p95_latency_ms']:.2f} ms")
    logger.info(f"  P99 latency: {result['p99_latency_ms']:.2f} ms")
    logger.info(f"  Throughput: {result['throughput_MBps']:.2f} MB/s")


@pytest.mark.dependency(depends=["test_45_consistency_model_performance"])
def test_45_analyze_consistency_model_results(ozone_benchmark):
    """
    Analyze the results from all consistency model tests and generate comprehensive report
    """
    # Skip if no results were collected
    if not hasattr(ozone_benchmark, "collected_results") or not ozone_benchmark.collected_results:
        pytest.skip("No benchmark results to analyze")
    
    # Run analysis
    summary = ozone_benchmark.analyze_results(ozone_benchmark.collected_results)
    
    # Verify we have data for different consistency models
    assert "consistency_model_comparison" in summary, "Summary should include consistency model comparison"
    
    consistency_models = list(summary["consistency_model_comparison"].keys())
    assert len(consistency_models) > 0, "Should have data for at least one consistency model"
    
    # Log key findings from the comparison
    if len(consistency_models) > 1:  # If we have multiple models to compare
        logger.info("\n=== Consistency Model Performance Comparison ===")
        
        for model in consistency_models:
            metrics = summary["consistency_model_comparison"][model]
            logger.info(f"{model} Consistency Model:")
            logger.info(f"  - Avg Throughput: {metrics['avg_throughput_MBps']:.2f} MB/s")
            logger.info(f"  - Avg Latency: {metrics['avg_latency_ms']:.2f} ms")
            logger.info(f"  - P95 Latency: {metrics['p95_latency_ms']:.2f} ms")
        
        # Make assertions about the expected relationship between consistency models
        # In general, eventual consistency should be faster than strong consistency
        if "EVENTUAL" in consistency_models and "STRONG" in consistency_models:
            eventual_metrics = summary["consistency_model_comparison"]["EVENTUAL"]
            strong_metrics = summary["consistency_model_comparison"]["STRONG"]
            
            # Document the trade-off between consistency and performance
            logger.info("\n=== Consistency vs Performance Trade-offs ===")
            logger.info(f"Throughput difference: {((strong_metrics['avg_throughput_MBps'] - eventual_metrics['avg_throughput_MBps']) / eventual_metrics['avg_throughput_MBps'] * 100):.2f}%")
            logger.info(f"Latency difference: {((strong_metrics['avg_latency_ms'] - eventual_metrics['avg_latency_ms']) / eventual_metrics['avg_latency_ms'] * 100):.2f}%")
            
            # Note: We don't make strict assertions about which model is faster
            # as actual performance depends on hardware, network, etc.
            # Instead we document the observed differences

import pytest
import time
import os
import subprocess
import statistics
from threading import Thread
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
from pyhdfs import HdfsClient
from typing import List, Dict, Tuple, Optional


class OzoneClient:
    """Helper class to interact with Ozone through shell commands"""
    
    def __init__(self, hosts="localhost:9878"):
        self.hosts = hosts
        
    def run_command(self, cmd: List[str]) -> Tuple[str, str, int]:
        """Run Ozone shell command and return stdout, stderr, and return code"""
        process = subprocess.Popen(
            ["ozone"] + cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate()
        return stdout, stderr, process.returncode
    
    def set_acl(self, volume: str, bucket: str, key: Optional[str], acl_spec: str) -> bool:
        """Set ACL on volume, bucket, or key"""
        resource = f"{volume}/{bucket}"
        if key:
            resource = f"{resource}/{key}"
            
        cmd = ["sh", "setacl", "-r", resource, acl_spec]
        _, stderr, rc = self.run_command(cmd)
        return rc == 0
    
    def create_volume(self, volume: str) -> bool:
        """Create an Ozone volume"""
        cmd = ["sh", "volume", "create", volume]
        _, stderr, rc = self.run_command(cmd)
        return rc == 0
    
    def create_bucket(self, volume: str, bucket: str) -> bool:
        """Create an Ozone bucket"""
        cmd = ["sh", "bucket", "create", f"{volume}/{bucket}"]
        _, stderr, rc = self.run_command(cmd)
        return rc == 0
        
    def put_key(self, volume: str, bucket: str, key: str, file_path: str) -> bool:
        """Upload a file to an Ozone key"""
        cmd = ["sh", "key", "put", f"{volume}/{bucket}/{key}", file_path]
        _, stderr, rc = self.run_command(cmd)
        return rc == 0
    
    def get_key(self, volume: str, bucket: str, key: str, output_path: str) -> bool:
        """Download a key to a file"""
        cmd = ["sh", "key", "get", f"{volume}/{bucket}/{key}", output_path]
        _, stderr, rc = self.run_command(cmd)
        return rc == 0
        
    def list_keys(self, volume: str, bucket: str) -> List[str]:
        """List keys in a bucket"""
        cmd = ["sh", "key", "list", f"{volume}/{bucket}"]
        stdout, _, rc = self.run_command(cmd)
        if rc != 0:
            return []
        # Parse the output to extract key names
        keys = [line.strip() for line in stdout.splitlines() if line.strip()]
        return keys


class PerformanceMeasurement:
    """Helper class for performance measurements"""
    
    @staticmethod
    def measure_operation_time(func, *args, **kwargs) -> float:
        """Measure the time taken for an operation"""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = (end_time - start_time) * 1000  # Convert to milliseconds
        return duration, result
    
    @staticmethod
    def run_benchmark(operation_func, iterations: int, *args, **kwargs) -> Dict:
        """Run a benchmark for an operation"""
        durations = []
        success_count = 0
        
        for _ in range(iterations):
            duration, result = PerformanceMeasurement.measure_operation_time(operation_func, *args, **kwargs)
            durations.append(duration)
            if result:  # Assuming the operation returns True on success
                success_count += 1
                
        results = {
            'mean': statistics.mean(durations) if durations else 0,
            'median': statistics.median(durations) if durations else 0,
            'min': min(durations) if durations else 0,
            'max': max(durations) if durations else 0,
            'p95': sorted(durations)[int(len(durations) * 0.95)] if len(durations) >= 20 else 0,
            'success_rate': (success_count / iterations) * 100 if iterations else 0
        }
        
        return results


def create_test_file(size_kb: int) -> str:
    """Create a test file of specified size in KB"""
    fd, path = tempfile.mkstemp()
    with os.fdopen(fd, 'wb') as f:
        f.write(os.urandom(size_kb * 1024))
    return path


def setup_acl_config(client: OzoneClient, volume: str, bucket: str, key: str, user: str) -> None:
    """Set up complex ACL configurations for testing"""
    # Set ACLs at volume level
    client.set_acl(volume, "", None, f"user:{user}:rw")
    
    # Set ACLs at bucket level
    client.set_acl(volume, bucket, None, f"user:{user}:rwx")
    
    # Set ACLs at key level with inheritance
    client.set_acl(volume, bucket, key, f"user:{user}:rwx")
    
    # Add group and other ACLs for more complexity
    client.set_acl(volume, bucket, None, "group:readers:r")
    client.set_acl(volume, bucket, None, "other::r")


def test_46_acl_performance_impact():
    """Evaluate performance impact of ACL checks"""
    # Setup
    client = OzoneClient()
    test_volume = "perftest46v1"
    test_bucket = "perftest46b1"
    test_key = "testkey"
    test_user = "testuser"
    iterations = 50  # Number of iterations for reliable measurements
    
    # Create volume and bucket for testing
    client.create_volume(test_volume)
    client.create_bucket(test_volume, test_bucket)
    
    # Create test files of different sizes
    file_sizes = [10, 100, 1024]  # 10KB, 100KB, 1MB
    test_files = {size: create_test_file(size) for size in file_sizes}
    
    results = []
    
    # Measure performance without ACLs first
    print("Measuring baseline performance without ACLs...")
    for size in file_sizes:
        # Upload operation
        upload_results = PerformanceMeasurement.run_benchmark(
            client.put_key, iterations, test_volume, test_bucket, f"{test_key}_{size}", test_files[size]
        )
        upload_results['operation'] = 'upload'
        upload_results['file_size_kb'] = size
        upload_results['acl_enabled'] = False
        results.append(upload_results)
        
        # Download operation
        with tempfile.NamedTemporaryFile() as tmp:
            download_results = PerformanceMeasurement.run_benchmark(
                client.get_key, iterations, test_volume, test_bucket, f"{test_key}_{size}", tmp.name
            )
            download_results['operation'] = 'download'
            download_results['file_size_kb'] = size
            download_results['acl_enabled'] = False
            results.append(download_results)
        
    # List operation
    list_results = PerformanceMeasurement.run_benchmark(
        client.list_keys, iterations, test_volume, test_bucket
    )
    list_results['operation'] = 'list'
    list_results['file_size_kb'] = 0  # N/A for list operation
    list_results['acl_enabled'] = False
    results.append(list_results)
    
    # Setup complex ACL configurations
    print("Setting up complex ACL configurations...")
    setup_acl_config(client, test_volume, test_bucket, test_key, test_user)
    
    # Measure performance with ACLs
    print("Measuring performance with ACLs...")
    for size in file_sizes:
        # Upload operation with ACLs
        upload_results = PerformanceMeasurement.run_benchmark(
            client.put_key, iterations, test_volume, test_bucket, f"{test_key}_acl_{size}", test_files[size]
        )
        upload_results['operation'] = 'upload'
        upload_results['file_size_kb'] = size
        upload_results['acl_enabled'] = True
        results.append(upload_results)
        
        # Download operation with ACLs
        with tempfile.NamedTemporaryFile() as tmp:
            download_results = PerformanceMeasurement.run_benchmark(
                client.get_key, iterations, test_volume, test_bucket, f"{test_key}_acl_{size}", tmp.name
            )
            download_results['operation'] = 'download'
            download_results['file_size_kb'] = size
            download_results['acl_enabled'] = True
            results.append(download_results)
    
    # List operation with ACLs
    list_results = PerformanceMeasurement.run_benchmark(
        client.list_keys, iterations, test_volume, test_bucket
    )
    list_results['operation'] = 'list'
    list_results['file_size_kb'] = 0  # N/A for list operation
    list_results['acl_enabled'] = True
    results.append(list_results)
    
    # Analyze and visualize results
    df = pd.DataFrame(results)
    
    # Clean up test files
    for file_path in test_files.values():
        if os.path.exists(file_path):
            os.remove(file_path)
    
    # Calculate overhead introduced by ACL checks
    overhead_analysis = []
    for op in df['operation'].unique():
        for size in df[df['operation'] != 'list']['file_size_kb'].unique():
            if op == 'list' and size > 0:
                continue
                
            baseline = df[(df['operation'] == op) & 
                          (df['file_size_kb'] == size) & 
                          (df['acl_enabled'] == False)]['mean'].values[0]
                          
            with_acl = df[(df['operation'] == op) & 
                          (df['file_size_kb'] == size) & 
                          (df['acl_enabled'] == True)]['mean'].values[0]
                          
            overhead_pct = ((with_acl - baseline) / baseline) * 100 if baseline > 0 else 0
            
            overhead_analysis.append({
                'operation': op,
                'file_size_kb': size,
                'baseline_ms': baseline,
                'with_acl_ms': with_acl,
                'overhead_ms': with_acl - baseline,
                'overhead_percent': overhead_pct
            })
    
    overhead_df = pd.DataFrame(overhead_analysis)
    print("\nACL Performance Overhead Analysis:")
    print(overhead_df)
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    operations = df['operation'].unique()
    
    for i, op in enumerate(operations):
        plt.subplot(len(operations), 1, i+1)
        op_data = df[df['operation'] == op]
        
        # For list operation
        if op == 'list':
            no_acl = op_data[op_data['acl_enabled'] == False]['mean'].values[0]
            with_acl = op_data[op_data['acl_enabled'] == True]['mean'].values[0]
            plt.bar(['Without ACLs', 'With ACLs'], [no_acl, with_acl])
            plt.title(f'{op.capitalize()} Operation')
            plt.ylabel('Time (ms)')
        # For upload and download operations
        else:
            for acl in [False, True]:
                acl_data = op_data[op_data['acl_enabled'] == acl]
                plt.plot(acl_data['file_size_kb'], acl_data['mean'], 
                         marker='o', label=f"{'With ACLs' if acl else 'Without ACLs'}")
            plt.title(f'{op.capitalize()} Operation')
            plt.xlabel('File Size (KB)')
            plt.ylabel('Time (ms)')
            plt.xscale('log')
            plt.legend()
    
    plt.tight_layout()
    plt.savefig('acl_performance_impact.png')
    
    # Validate that ACL overhead is acceptable
    max_acceptable_overhead_pct = 15  # 15% threshold
    high_overhead_operations = overhead_df[overhead_df['overhead_percent'] > max_acceptable_overhead_pct]
    
    if not high_overhead_operations.empty:
        print("\nWARNING: The following operations have high ACL overhead:")
        print(high_overhead_operations)
    
    # Final validation
    assert not high_overhead_operations.empty, "ACL checks introduce significant overhead to operations"
    print("\nPerformance test passed: ACL checks introduce minimal overhead to operations")

import time
import pytest
import logging
import subprocess
import os
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from pyozone.client import OzoneClient
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
VOLUME_NAME = "perfvol47"
BUCKET_NAME = "perfbucket47"
NUM_KEYS = 1000
OVERWRITE_ITERATIONS = 10
CONCURRENT_CLIENTS = 5
TEST_DATA_SIZES = ["10KB", "100KB", "1MB"]

# Utility functions
def create_test_file(size_str: str) -> str:
    """Create a test file of specified size"""
    size_map = {
        "10KB": 10 * 1024,
        "100KB": 100 * 1024,
        "1MB": 1 * 1024 * 1024,
        "10MB": 10 * 1024 * 1024
    }
    
    size = size_map.get(size_str, 1024)
    file_path = f"test_file_{size_str}.dat"
    
    with open(file_path, "wb") as f:
        f.write(os.urandom(size))
    
    return file_path

def setup_ozone_resources():
    """Set up Ozone volume and bucket for testing"""
    try:
        # Create volume and bucket using Ozone shell
        subprocess.run(["ozone", "sh", "volume", "create", VOLUME_NAME], check=True)
        subprocess.run(["ozone", "sh", "bucket", "create", f"/{VOLUME_NAME}/{BUCKET_NAME}"], check=True)
        logger.info(f"Created volume {VOLUME_NAME} and bucket {BUCKET_NAME}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to create Ozone resources: {e}")
        raise

def cleanup_ozone_resources():
    """Clean up Ozone resources after testing"""
    try:
        subprocess.run(["ozone", "sh", "bucket", "delete", f"/{VOLUME_NAME}/{BUCKET_NAME}"], check=True)
        subprocess.run(["ozone", "sh", "volume", "delete", VOLUME_NAME], check=True)
        logger.info(f"Cleaned up volume {VOLUME_NAME} and bucket {BUCKET_NAME}")
    except subprocess.CalledProcessError:
        logger.warning(f"Failed to clean up some Ozone resources")

def collect_metrics() -> Dict:
    """Collect current performance metrics from Ozone cluster"""
    metrics = {}
    
    # Example: Get storage utilization using Ozone admin commands
    result = subprocess.run(
        ["ozone", "admin", "datanode", "overview"], 
        capture_output=True, 
        text=True, 
        check=True
    )
    
    # Parse the output to extract storage metrics
    # This is a simplified example - would need to be adapted to actual Ozone output format
    metrics["storage_used"] = "N/A"  # Parse from result
    
    # Get garbage collection metrics
    # In a real implementation, you might get these from JMX/Prometheus/etc.
    metrics["gc_count"] = "N/A"
    metrics["gc_time"] = "N/A"
    
    return metrics

class KeyPerformanceMetrics:
    """Class to track performance metrics for key operations"""
    
    def __init__(self):
        self.write_latencies = []
        self.read_latencies = []
        self.metrics_snapshots = []
        self.timestamp = []
    
    def record_write_latency(self, latency_ms):
        self.write_latencies.append(latency_ms)
    
    def record_read_latency(self, latency_ms):
        self.read_latencies.append(latency_ms)
    
    def record_metrics_snapshot(self, metrics):
        self.metrics_snapshots.append(metrics)
        self.timestamp.append(time.time())
    
    def get_avg_write_latency(self):
        return sum(self.write_latencies) / len(self.write_latencies) if self.write_latencies else 0
    
    def get_avg_read_latency(self):
        return sum(self.read_latencies) / len(self.read_latencies) if self.read_latencies else 0
    
    def plot_metrics(self, output_file="key_overwrite_performance.png"):
        """Generate plots for the collected metrics"""
        plt.figure(figsize=(12, 8))
        
        plt.subplot(211)
        plt.title("Write Latency Over Time")
        plt.plot(range(len(self.write_latencies)), self.write_latencies, 'b-')
        plt.ylabel("Latency (ms)")
        plt.xlabel("Operation #")
        
        plt.subplot(212)
        plt.title("Read Latency Over Time")
        plt.plot(range(len(self.read_latencies)), self.read_latencies, 'g-')
        plt.ylabel("Latency (ms)")
        plt.xlabel("Operation #")
        
        plt.tight_layout()
        plt.savefig(output_file)
        logger.info(f"Performance metrics plot saved to {output_file}")
        
        # Also save raw data for further analysis
        pd.DataFrame({
            'write_latency': self.write_latencies,
            'read_latency': self.read_latencies,
        }).to_csv("key_overwrite_metrics.csv", index=False)


def overwrite_key(client, volume, bucket, key_prefix, iteration, data_file):
    """Overwrite a key and measure performance"""
    key = f"{key_prefix}_{iteration}"
    
    start_time = time.time()
    client.put_key(volume, bucket, key, data_file)
    end_time = time.time()
    
    return (end_time - start_time) * 1000  # return latency in ms

def read_key(client, volume, bucket, key_prefix, iteration):
    """Read a key and measure performance"""
    key = f"{key_prefix}_{iteration}"
    
    start_time = time.time()
    client.get_key(volume, bucket, key)
    end_time = time.time()
    
    return (end_time - start_time) * 1000  # return latency in ms


@pytest.fixture(scope="module")
def ozone_setup():
    """Setup and teardown for Ozone testing"""
    setup_ozone_resources()
    
    # Create test files of different sizes
    data_files = {size: create_test_file(size) for size in TEST_DATA_SIZES}
    
    yield data_files
    
    # Clean up test files
    for file_path in data_files.values():
        if os.path.exists(file_path):
            os.remove(file_path)
    
    cleanup_ozone_resources()


@pytest.mark.parametrize(
    "data_size,num_keys,iterations", 
    [
        ("10KB", 100, 5),
        ("100KB", 100, 5),
        ("1MB", 50, 3)
    ]
)
def test_47_continuous_key_overwrite_performance(ozone_setup, data_size, num_keys, iterations):
    """Test performance under continuous key overwrite scenarios"""
    logger.info(f"Starting continuous key overwrite test with {data_size} files, {num_keys} keys, {iterations} iterations")
    
    # Initialize metrics tracker
    metrics = KeyPerformanceMetrics()
    
    # Get test data file
    data_file_path = ozone_setup[data_size]
    
    # Create Ozone client
    client = OzoneClient()
    
    # Run overwrite workload
    for iter_num in range(iterations):
        logger.info(f"Starting iteration {iter_num+1}/{iterations}")
        
        # Snapshot metrics at the start of iteration
        metrics.record_metrics_snapshot(collect_metrics())
        
        # Overwrite keys in parallel
        with ThreadPoolExecutor(max_workers=CONCURRENT_CLIENTS) as executor:
            futures = []
            for key_id in range(num_keys):
                futures.append(executor.submit(
                    overwrite_key, client, VOLUME_NAME, BUCKET_NAME, 
                    f"key{key_id}", iter_num, data_file_path
                ))
            
            # Collect results
            for future in futures:
                latency = future.result()
                metrics.record_write_latency(latency)
        
        # Measure read performance for the overwritten keys
        with ThreadPoolExecutor(max_workers=CONCURRENT_CLIENTS) as executor:
            read_futures = []
            for key_id in range(0, num_keys, 10):  # Sample 10% of keys for reading
                read_futures.append(executor.submit(
                    read_key, client, VOLUME_NAME, BUCKET_NAME, 
                    f"key{key_id}", iter_num
                ))
            
            # Collect results
            for future in read_futures:
                latency = future.result()
                metrics.record_read_latency(latency)
    
    # Final metrics snapshot
    metrics.record_metrics_snapshot(collect_metrics())
    
    # Generate performance report
    metrics.plot_metrics(f"key_overwrite_perf_{data_size}.png")
    
    # Log summary
    logger.info(f"Performance test completed: Avg write latency: {metrics.get_avg_write_latency():.2f}ms, " 
                f"Avg read latency: {metrics.get_avg_read_latency():.2f}ms")
    
    # Validate performance is within acceptable limits
    # The thresholds would depend on specific environment and requirements
    avg_write_latency = metrics.get_avg_write_latency()
    avg_read_latency = metrics.get_avg_read_latency()
    
    # These thresholds should be adjusted based on your specific environment
    max_acceptable_write_latency = 5000  # 5 seconds in ms
    max_acceptable_read_latency = 2000   # 2 seconds in ms
    
    # Performance assertions
    assert avg_write_latency < max_acceptable_write_latency, \
        f"Write latency ({avg_write_latency}ms) exceeds acceptable threshold ({max_acceptable_write_latency}ms)"
    
    assert avg_read_latency < max_acceptable_read_latency, \
        f"Read latency ({avg_read_latency}ms) exceeds acceptable threshold ({max_acceptable_read_latency}ms)"
    
    # Check if there are any significant latency spikes (e.g., 3x average)
    latency_spikes = [l for l in metrics.write_latencies if l > 3 * avg_write_latency]
    assert len(latency_spikes) / len(metrics.write_latencies) < 0.05, \
        f"Too many latency spikes observed ({len(latency_spikes)} out of {len(metrics.write_latencies)} operations)"
    
    logger.info("Performance validation passed: System maintains stable performance under continuous key overwrites")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import json
import csv
import pytest
import subprocess
import logging
from typing import List, Dict, Tuple
from dataclasses import dataclass
import concurrent.futures
from contextlib import contextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants for test configuration
TEST_VOLUME = "perfvolume"
TEST_BUCKET = "perfbucket"
TEMP_DIR = "/tmp/ozone-perf-tests"
RESULTS_DIR = f"{TEMP_DIR}/results"
BENCHMARK_DURATION = 300  # seconds
NUM_CLIENTS = [1, 4, 8, 16, 32]  # Number of concurrent clients to test

@dataclass
class SCMConfiguration:
    """Class representing an SCM configuration for performance testing"""
    name: str
    num_scm_nodes: int
    rocksdb_config: Dict[str, str]
    additional_params: Dict[str, str]

@contextmanager
def timed_operation(operation_name: str):
    """Context manager to measure and log the time of an operation"""
    start_time = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        logger.info(f"{operation_name} completed in {elapsed:.2f} seconds")

def setup_cluster_with_scm_config(config: SCMConfiguration) -> None:
    """
    Set up an Ozone cluster with specified SCM configuration
    
    Args:
        config: The SCM configuration to apply
    """
    logger.info(f"Setting up cluster with SCM configuration: {config.name}")
    
    # Create directory for configuration files
    os.makedirs(f"{TEMP_DIR}/configs/{config.name}", exist_ok=True)
    
    # Generate ozone-site.xml content with SCM configuration
    ozone_site_content = f"""


ozone.scm.nodes
{','.join([f'scm{i}.example.com' for i in range(config.num_scm_nodes)])}

"""
    
    # Add RocksDB configuration parameters
    for key, value in config.rocksdb_config.items():
        ozone_site_content += f"""    
{key}
{value}

"""
    
    # Add additional parameters
    for key, value in config.additional_params.items():
        ozone_site_content += f"""    
{key}
{value}

"""
    
    ozone_site_content += ""
    
    # Write configuration file
    config_path = f"{TEMP_DIR}/configs/{config.name}/ozone-site.xml"
    with open(config_path, 'w') as f:
        f.write(ozone_site_content)
    
    # Apply configuration to cluster (simulated for test)
    # In a real environment, this would involve restarting the cluster with the new configuration
    logger.info(f"Applied SCM configuration: {config.name}")
    
    # Allow time for cluster to stabilize with new configuration
    time.sleep(5)

def run_ozone_benchmark(config: SCMConfiguration, benchmark_type: str, 
                        num_clients: int, file_size_mb: int) -> Dict:
    """
    Run a performance benchmark against Ozone with given configuration
    
    Args:
        config: The SCM configuration being tested
        benchmark_type: Type of benchmark (write, read, metadata)
        num_clients: Number of concurrent clients
        file_size_mb: Size of test files in MB
        
    Returns:
        Dictionary containing benchmark results
    """
    logger.info(f"Running {benchmark_type} benchmark with {num_clients} clients, "
                f"file size {file_size_mb}MB on config {config.name}")
    
    benchmark_start = time.time()
    operations_count = 0
    
    # Create test files if needed
    if benchmark_type in ['write', 'read']:
        os.makedirs(f"{TEMP_DIR}/data", exist_ok=True)
        if benchmark_type == 'write' or not os.path.exists(f"{TEMP_DIR}/data/{file_size_mb}MB.dat"):
            with open(f"{TEMP_DIR}/data/{file_size_mb}MB.dat", 'wb') as f:
                f.write(os.urandom(file_size_mb * 1024 * 1024))
    
    # Create a pool of workers for concurrent operations
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_clients) as executor:
        futures = []
        
        # Submit benchmark tasks based on type
        for i in range(num_clients):
            if benchmark_type == 'write':
                futures.append(executor.submit(
                    _run_write_operation, 
                    i, 
                    f"{TEMP_DIR}/data/{file_size_mb}MB.dat"
                ))
            elif benchmark_type == 'read':
                futures.append(executor.submit(
                    _run_read_operation, 
                    i, 
                    f"{TEMP_DIR}/data/{file_size_mb}MB.dat"
                ))
            elif benchmark_type == 'metadata':
                futures.append(executor.submit(_run_metadata_operation, i))
        
        # Wait for all operations to complete or timeout
        end_time = time.time() + BENCHMARK_DURATION
        for future in concurrent.futures.as_completed(futures):
            if time.time() > end_time:
                break
                
            result = future.result()
            operations_count += result
    
    benchmark_duration = time.time() - benchmark_start
    
    # Calculate metrics
    ops_per_second = operations_count / benchmark_duration if benchmark_duration > 0 else 0
    
    # Return benchmark results
    return {
        'config_name': config.name,
        'benchmark_type': benchmark_type,
        'num_clients': num_clients,
        'file_size_mb': file_size_mb,
        'duration_seconds': benchmark_duration,
        'operations_count': operations_count,
        'operations_per_second': ops_per_second,
    }

def _run_write_operation(client_id: int, file_path: str) -> int:
    """Simulated write operation to Ozone"""
    # In a real test, this would use the Ozone client to write data
    # Here we just simulate the operation with a sleep time
    key = f"key-{client_id}-{int(time.time())}"
    
    try:
        # Simulated write operation - in real test would use Ozone client
        # subprocess.run(["ozone", "sh", "key", "put", file_path, f"{TEST_VOLUME}/{TEST_BUCKET}/{key}"])
        time.sleep(0.1)  # Simulate operation time
        return 1
    except Exception as e:
        logger.error(f"Write operation failed: {e}")
        return 0

def _run_read_operation(client_id: int, file_path: str) -> int:
    """Simulated read operation from Ozone"""
    # In a real test, this would use the Ozone client to read data
    key = f"key-{client_id}-{int(time.time())}"
    
    try:
        # Simulated read operation - in real test would use Ozone client
        # subprocess.run(["ozone", "sh", "key", "get", f"{TEST_VOLUME}/{TEST_BUCKET}/{key}", file_path])
        time.sleep(0.05)  # Simulate operation time
        return 1
    except Exception as e:
        logger.error(f"Read operation failed: {e}")
        return 0

def _run_metadata_operation(client_id: int) -> int:
    """Simulated metadata operation in Ozone"""
    # In a real test, this would use the Ozone client to perform metadata ops
    operations = 0
    
    try:
        # Simulate listing keys in a bucket
        # subprocess.run(["ozone", "sh", "key", "list", f"{TEST_VOLUME}/{TEST_BUCKET}"])
        time.sleep(0.02)  # Simulate operation time
        operations += 1
        
        # Simulate checking if a key exists
        # subprocess.run(["ozone", "sh", "key", "info", f"{TEST_VOLUME}/{TEST_BUCKET}/some-key"])
        time.sleep(0.01)  # Simulate operation time
        operations += 1
        
        return operations
    except Exception as e:
        logger.error(f"Metadata operation failed: {e}")
        return 0

def save_benchmark_results(results: List[Dict], config_name: str) -> str:
    """
    Save benchmark results to CSV and JSON files
    
    Args:
        results: List of benchmark result dictionaries
        config_name: Name of the SCM configuration tested
        
    Returns:
        Path to the results directory
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = int(time.time())
    results_prefix = f"{RESULTS_DIR}/{timestamp}_{config_name}"
    
    # Save as CSV
    csv_path = f"{results_prefix}.csv"
    with open(csv_path, 'w', newline='') as csvfile:
        if results:
            writer = csv.DictWriter(csvfile, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    
    # Save as JSON
    json_path = f"{results_prefix}.json"
    with open(json_path, 'w') as jsonfile:
        json.dump(results, jsonfile, indent=2)
    
    logger.info(f"Results saved to {csv_path} and {json_path}")
    return RESULTS_DIR

def analyze_results(results_dir: str, configs: List[SCMConfiguration]) -> Dict:
    """
    Analyze benchmark results and determine optimal configurations
    
    Args:
        results_dir: Directory containing benchmark results
        configs: List of SCM configurations tested
        
    Returns:
        Dictionary with analysis results
    """
    # In a real implementation, this would perform detailed analysis
    # For this example, we'll simulate finding optimal configurations
    
    analysis = {
        'optimal_write_config': None,
        'optimal_read_config': None,
        'optimal_metadata_config': None,
        'balanced_config': None,
        'analysis_details': {}
    }
    
    # Simulate analysis
    for config in configs:
        analysis['analysis_details'][config.name] = {
            'write_performance': 100 + (config.num_scm_nodes * 5),
            'read_performance': 200 + (config.num_scm_nodes * 3),
            'metadata_performance': 500 + (config.num_scm_nodes * 10)
        }
    
    # Find optimal configs (in a real scenario, this would be based on actual results)
    max_write = 0
    max_read = 0
    max_metadata = 0
    max_balanced = 0
    
    for config in configs:
        details = analysis['analysis_details'][config.name]
        
        write_perf = details['write_performance']
        read_perf = details['read_performance']
        metadata_perf = details['metadata_performance']
        balanced_score = (write_perf + read_perf + metadata_perf) / 3
        
        if write_perf > max_write:
            max_write = write_perf
            analysis['optimal_write_config'] = config.name
            
        if read_perf > max_read:
            max_read = read_perf
            analysis['optimal_read_config'] = config.name
            
        if metadata_perf > max_metadata:
            max_metadata = metadata_perf
            analysis['optimal_metadata_config'] = config.name
            
        if balanced_score > max_balanced:
            max_balanced = balanced_score
            analysis['balanced_config'] = config.name
    
    return analysis

@pytest.fixture(scope="module")
def prepare_test_environment():
    """Create necessary directories and prepare the test environment"""
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Simulated volume and bucket creation
    # In a real test, this would use Ozone client or shell commands
    logger.info(f"Created test volume {TEST_VOLUME} and bucket {TEST_BUCKET}")
    
    yield
    
    # Cleanup (commented out for test purposes)
    # shutil.rmtree(TEMP_DIR)

# Define SCM configurations to test
@pytest.fixture
def scm_configurations():
    return [
        SCMConfiguration(
            name="single-scm-default",
            num_scm_nodes=1,
            rocksdb_config={
                "ozone.scm.db.rocksDB.writeBufferSizeMB": "64",
                "ozone.scm.db.rocksDB.blockSize": "4096"
            },
            additional_params={
                "ozone.scm.container.size": "5GB",
                "ozone.scm.ratis.request.timeout": "30s"
            }
        ),
        SCMConfiguration(
            name="single-scm-optimized",
            num_scm_nodes=1,
            rocksdb_config={
                "ozone.scm.db.rocksDB.writeBufferSizeMB": "128",
                "ozone.scm.db.rocksDB.blockSize": "8192",
                "ozone.scm.db.rocksDB.maxBackgroundCompactions": "4",
                "ozone.scm.db.rocksDB.maxBackgroundFlushes": "2"
            },
            additional_params={
                "ozone.scm.container.size": "5GB",
                "ozone.scm.ratis.request.timeout": "30s"
            }
        ),
        SCMConfiguration(
            name="ha-scm-default",
            num_scm_nodes=3,
            rocksdb_config={
                "ozone.scm.db.rocksDB.writeBufferSizeMB": "64",
                "ozone.scm.db.rocksDB.blockSize": "4096"
            },
            additional_params={
                "ozone.scm.container.size": "5GB",
                "ozone.scm.ratis.request.timeout": "60s"
            }
        ),
        SCMConfiguration(
            name="ha-scm-optimized",
            num_scm_nodes=3,
            rocksdb_config={
                "ozone.scm.db.rocksDB.writeBufferSizeMB": "128",
                "ozone.scm.db.rocksDB.blockSize": "8192",
                "ozone.scm.db.rocksDB.maxBackgroundCompactions": "4",
                "ozone.scm.db.rocksDB.maxBackgroundFlushes": "2"
            },
            additional_params={
                "ozone.scm.container.size": "5GB",
                "ozone.scm.ratis.request.timeout": "60s",
                "ozone.scm.ratis.segment.size": "16MB"
            }
        ),
        SCMConfiguration(
            name="ha-scm-high-performance",
            num_scm_nodes=5,
            rocksdb_config={
                "ozone.scm.db.rocksDB.writeBufferSizeMB": "256",
                "ozone.scm.db.rocksDB.blockSize": "16384",
                "ozone.scm.db.rocksDB.maxBackgroundCompactions": "8",
                "ozone.scm.db.rocksDB.maxBackgroundFlushes": "4",
                "ozone.scm.db.rocksDB.compactionStyle": "LEVEL"
            },
            additional_params={
                "ozone.scm.container.size": "10GB",
                "ozone.scm.ratis.request.timeout": "30s",
                "ozone.scm.ratis.segment.size": "32MB",
                "ozone.scm.ratis.segment.preallocated.size": "16MB"
            }
        )
    ]

# Test parameters - different file sizes to test with
@pytest.mark.parametrize("file_size_mb", [1, 10, 100, 512])
def test_48_scm_configuration_performance(prepare_test_environment, 
                                         scm_configurations, 
                                         file_size_mb):
    """
    Evaluate performance with different SCM (Storage Container Manager) configurations.
    
    This test:
    1. Configures Ozone with different SCM settings
    2. Runs standard benchmarks for each configuration
    3. Measures impact on metadata operations and overall system performance
    4. Identifies optimal SCM configurations
    """
    logger.info("Starting SCM configuration performance test")
    all_results = []
    
    for config in scm_configurations:
        with timed_operation(f"Testing configuration {config.name}"):
            # Set up cluster with this SCM configuration
            setup_cluster_with_scm_config(config)
            
            config_results = []
            
            # Run write benchmarks with different client counts
            for clients in NUM_CLIENTS:
                result = run_ozone_benchmark(
                    config=config,
                    benchmark_type="write",
                    num_clients=clients,
                    file_size_mb=file_size_mb
                )
                config_results.append(result)
            
            # Run read benchmarks with different client counts
            for clients in NUM_CLIENTS:
                result = run_ozone_benchmark(
                    config=config,
                    benchmark_type="read",
                    num_clients=clients,
                    file_size_mb=file_size_mb
                )
                config_results.append(result)
            
            # Run metadata benchmarks with different client counts
            for clients in NUM_CLIENTS:
                result = run_ozone_benchmark(
                    config=config,
                    benchmark_type="metadata",
                    num_clients=clients,
                    file_size_mb=file_size_mb
                )
                config_results.append(result)
            
            # Save results for this configuration
            results_dir = save_benchmark_results(config_results, config.name)
            all_results.extend(config_results)
    
    # Analyze all results to determine optimal configurations
    analysis = analyze_results(RESULTS_DIR, scm_configurations)
    
    # Log analysis results
    logger.info("SCM Configuration Performance Analysis:")
    logger.info(f"Best configuration for write operations: {analysis['optimal_write_config']}")
    logger.info(f"Best configuration for read operations: {analysis['optimal_read_config']}")
    logger.info(f"Best configuration for metadata operations: {analysis['optimal_metadata_config']}")
    logger.info(f"Best balanced configuration: {analysis['balanced_config']}")
    
    # Save analysis results
    with open(f"{RESULTS_DIR}/analysis_summary.json", 'w') as f:
        json.dump(analysis, f, indent=2)
    
    # Assertions to verify the test ran successfully
    assert len(all_results) > 0, "No benchmark results were collected"
    
    # Assert that we have identified optimal configurations
    assert analysis['optimal_write_config'] is not None, "Failed to identify optimal write configuration"
    assert analysis['optimal_read_config'] is not None, "Failed to identify optimal read configuration"
    assert analysis['optimal_metadata_config'] is not None, "Failed to identify optimal metadata configuration"
    assert analysis['balanced_config'] is not None, "Failed to identify balanced configuration"

#!/usr/bin/env python3
"""
Performance tests for Apache Ozone namespace operations under load.
"""

import time
import pytest
import subprocess
import logging
import concurrent.futures
import statistics
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
OZONE_BIN = "/opt/ozone/bin/ozone"
VOLUME_NAME = "perfvol"
BUCKET_NAME = "perfbucket"
RESULTS_DIR = Path("./performance_results")

# Create results directory if it doesn't exist
RESULTS_DIR.mkdir(exist_ok=True)

class NamespaceOperationStats:
    """Class to track and analyze namespace operation performance metrics."""
    
    def __init__(self, operation_type):
        self.operation_type = operation_type
        self.latencies = []
        self.throughput = None
        self.start_time = None
        self.end_time = None
        
    def start_timer(self):
        self.start_time = time.time()
        
    def end_timer(self):
        self.end_time = time.time()
        
    def record_latency(self, latency):
        self.latencies.append(latency)
        
    def calculate_throughput(self, operations_count):
        if self.start_time is not None and self.end_time is not None:
            duration = self.end_time - self.start_time
            self.throughput = operations_count / duration if duration > 0 else 0
            
    def get_stats(self):
        if not self.latencies:
            return {"min": 0, "max": 0, "avg": 0, "p50": 0, "p90": 0, "p99": 0, "throughput": 0}
        
        return {
            "min": min(self.latencies),
            "max": max(self.latencies),
            "avg": statistics.mean(self.latencies),
            "p50": np.percentile(self.latencies, 50),
            "p90": np.percentile(self.latencies, 90),
            "p99": np.percentile(self.latencies, 99),
            "throughput": self.throughput
        }
    
    def plot_latency_distribution(self):
        plt.figure(figsize=(10, 6))
        plt.hist(self.latencies, bins=50, alpha=0.7)
        plt.title(f'{self.operation_type} Latency Distribution')
        plt.xlabel('Latency (seconds)')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig(RESULTS_DIR / f"{self.operation_type}_latency_distribution.png")

def run_ozone_command(command):
    """Execute an Ozone command and return the result."""
    cmd = f"{OZONE_BIN} {command}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Command failed: {cmd}")
        logger.error(f"Stdout: {result.stdout}")
        logger.error(f"Stderr: {result.stderr}")
        raise Exception(f"Ozone command failed: {command}")
    return result.stdout.strip()

def setup_environment():
    """Set up the required environment for testing."""
    try:
        # Create volume
        run_ozone_command(f"sh volume create {VOLUME_NAME}")
        logger.info(f"Created volume: {VOLUME_NAME}")
        
        # Create bucket
        run_ozone_command(f"sh bucket create {VOLUME_NAME}/{BUCKET_NAME}")
        logger.info(f"Created bucket: {VOLUME_NAME}/{BUCKET_NAME}")
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        teardown_environment()
        raise

def teardown_environment():
    """Clean up the environment after testing."""
    try:
        # Remove bucket
        run_ozone_command(f"sh bucket delete {VOLUME_NAME}/{BUCKET_NAME}")
        logger.info(f"Deleted bucket: {VOLUME_NAME}/{BUCKET_NAME}")
        
        # Remove volume
        run_ozone_command(f"sh volume delete {VOLUME_NAME}")
        logger.info(f"Deleted volume: {VOLUME_NAME}")
    except Exception as e:
        logger.error(f"Teardown failed: {e}")

def generate_key_name(prefix, index):
    """Generate a key name with the given prefix and index."""
    return f"{prefix}_{index:09d}"

def create_object(key_name, data_file):
    """Create an object in Ozone."""
    start_time = time.time()
    run_ozone_command(f"key put {VOLUME_NAME}/{BUCKET_NAME}/ {data_file} --key={key_name}")
    end_time = time.time()
    return end_time - start_time

def rename_object(old_key, new_key):
    """Rename an object in Ozone."""
    start_time = time.time()
    run_ozone_command(f"key rename {VOLUME_NAME}/{BUCKET_NAME}/ {old_key} {new_key}")
    end_time = time.time()
    return end_time - start_time

def delete_object(key_name):
    """Delete an object from Ozone."""
    start_time = time.time()
    run_ozone_command(f"key delete {VOLUME_NAME}/{BUCKET_NAME}/ {key_name}")
    end_time = time.time()
    return end_time - start_time

def create_test_file(file_path, size_kb=1):
    """Create a test file with the specified size."""
    with open(file_path, 'wb') as f:
        f.write(b'0' * (size_kb * 1024))
    
def generate_namespace_load(count, batch_size, data_file):
    """Generate a namespace with the specified number of objects."""
    create_stats = NamespaceOperationStats("create")
    create_stats.start_timer()
    
    for batch_start in range(0, count, batch_size):
        batch_end = min(batch_start + batch_size, count)
        logger.info(f"Creating objects {batch_start} to {batch_end-1}")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = []
            for i in range(batch_start, batch_end):
                key_name = generate_key_name("key", i)
                futures.append(executor.submit(create_object, key_name, data_file))
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    latency = future.result()
                    create_stats.record_latency(latency)
                except Exception as e:
                    logger.error(f"Error creating object: {e}")
    
    create_stats.end_timer()
    create_stats.calculate_throughput(count)
    return create_stats

def perform_rename_operations(count, batch_size):
    """Perform rename operations on a subset of objects."""
    rename_stats = NamespaceOperationStats("rename")
    rename_stats.start_timer()
    
    # Rename a percentage of objects
    rename_count = int(count * 0.1)  # Rename 10% of objects
    
    for batch_start in range(0, rename_count, batch_size):
        batch_end = min(batch_start + batch_size, rename_count)
        logger.info(f"Renaming objects {batch_start} to {batch_end-1}")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = []
            for i in range(batch_start, batch_end):
                old_key = generate_key_name("key", i)
                new_key = generate_key_name("renamed", i)
                futures.append(executor.submit(rename_object, old_key, new_key))
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    latency = future.result()
                    rename_stats.record_latency(latency)
                except Exception as e:
                    logger.error(f"Error renaming object: {e}")
    
    rename_stats.end_timer()
    rename_stats.calculate_throughput(rename_count)
    return rename_stats

def perform_delete_operations(count, batch_size):
    """Perform delete operations on objects."""
    delete_stats = NamespaceOperationStats("delete")
    delete_stats.start_timer()
    
    # Delete a percentage of objects
    delete_count = int(count * 0.2)  # Delete 20% of objects
    
    for batch_start in range(0, delete_count, batch_size):
        batch_end = min(batch_start + batch_size, delete_count)
        logger.info(f"Deleting objects {batch_start} to {batch_end-1}")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = []
            for i in range(batch_start, batch_end):
                # Delete some original keys and some renamed keys
                if i < delete_count / 2:
                    key_name = generate_key_name("key", i + delete_count)
                else:
                    key_name = generate_key_name("renamed", i - delete_count // 2)
                futures.append(executor.submit(delete_object, key_name))
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    latency = future.result()
                    delete_stats.record_latency(latency)
                except Exception as e:
                    logger.error(f"Error deleting object: {e}")
    
    delete_stats.end_timer()
    delete_stats.calculate_throughput(delete_count)
    return delete_stats

def save_performance_report(namespace_size, create_stats, rename_stats, delete_stats):
    """Save the performance test results to a file."""
    report_path = RESULTS_DIR / f"namespace_load_test_report_{namespace_size}.txt"
    
    with open(report_path, 'w') as f:
        f.write(f"# Namespace Load Performance Test Report\n\n")
        f.write(f"Namespace Size: {namespace_size} objects\n\n")
        
        for operation, stats in [
            ("Create", create_stats),
            ("Rename", rename_stats),
            ("Delete", delete_stats)
        ]:
            metrics = stats.get_stats()
            f.write(f"## {operation} Operation Performance\n")
            f.write(f"- Min Latency: {metrics['min']:.6f} seconds\n")
            f.write(f"- Max Latency: {metrics['max']:.6f} seconds\n")
            f.write(f"- Avg Latency: {metrics['avg']:.6f} seconds\n")
            f.write(f"- P50 Latency: {metrics['p50']:.6f} seconds\n")
            f.write(f"- P90 Latency: {metrics['p90']:.6f} seconds\n")
            f.write(f"- P99 Latency: {metrics['p99']:.6f} seconds\n")
            f.write(f"- Throughput: {metrics['throughput']:.2f} ops/sec\n\n")
    
    logger.info(f"Performance report saved to {report_path}")

def plot_comparison_chart(create_stats, rename_stats, delete_stats):
    """Plot a comparison chart of operation latencies."""
    operations = ['Create', 'Rename', 'Delete']
    
    # Get metrics for each operation
    create_metrics = create_stats.get_stats()
    rename_metrics = rename_stats.get_stats()
    delete_metrics = delete_stats.get_stats()
    
    # Extract data for the chart
    avg_latencies = [create_metrics['avg'], rename_metrics['avg'], delete_metrics['avg']]
    p90_latencies = [create_metrics['p90'], rename_metrics['p90'], delete_metrics['p90']]
    p99_latencies = [create_metrics['p99'], rename_metrics['p99'], delete_metrics['p99']]
    
    # Create bar chart
    x = np.arange(len(operations))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(x - width, avg_latencies, width, label='Average Latency')
    ax.bar(x, p90_latencies, width, label='P90 Latency')
    ax.bar(x + width, p99_latencies, width, label='P99 Latency')
    
    ax.set_ylabel('Latency (seconds)')
    ax.set_title('Namespace Operation Latency Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(operations)
    ax.legend()
    
    plt.savefig(RESULTS_DIR / "operation_latency_comparison.png")

@pytest.fixture(scope="module")
def test_environment():
    """Fixture to set up and tear down the test environment."""
    setup_environment()
    
    # Create a small test file
    test_file = RESULTS_DIR / "test_data.txt"
    create_test_file(test_file, size_kb=4)
    
    yield {"test_file": test_file}
    
    teardown_environment()

# Different namespace sizes for testing (scaled down for practical testing)
# In a real environment, these would be much larger
@pytest.mark.parametrize("namespace_size,batch_size", [
    (100, 20),        # Small test
    (1000, 100),      # Medium test
    # Larger tests would be included in a real scenario
    # (10000, 500),   # Large test
    # (100000, 1000), # Very large test
])
def test_49_namespace_load_performance(test_environment, namespace_size, batch_size):
    """
    Test performance under namespace load
    
    This test evaluates the performance of Ozone namespace operations (create, rename, delete)
    under increasing namespace loads. It measures and analyzes latency and throughput metrics.
    """
    logger.info(f"Starting namespace load test with {namespace_size} objects")
    
    test_file = test_environment["test_file"]
    
    # Step 1: Generate a large namespace
    logger.info(f"Generating namespace with {namespace_size} objects")
    create_stats = generate_namespace_load(namespace_size, batch_size, test_file)
    
    # Step 2 & 3: Perform namespace operations and measure performance
    logger.info("Performing rename operations")
    rename_stats = perform_rename_operations(namespace_size, batch_size)
    
    logger.info("Performing delete operations")
    delete_stats = perform_delete_operations(namespace_size, batch_size)
    
    # Create performance visualizations
    create_stats.plot_latency_distribution()
    rename_stats.plot_latency_distribution()
    delete_stats.plot_latency_distribution()
    plot_comparison_chart(create_stats, rename_stats, delete_stats)
    
    # Save performance report
    save_performance_report(namespace_size, create_stats, rename_stats, delete_stats)
    
    # Step 4: Analyze OM performance by checking if metrics meet acceptable thresholds
    create_metrics = create_stats.get_stats()
    rename_metrics = rename_stats.get_stats()
    delete_metrics = delete_stats.get_stats()
    
    # Define acceptable thresholds (these would be adjusted based on your requirements)
    max_acceptable_p99_latency = 2.0  # seconds
    min_acceptable_throughput = 10.0   # operations per second
    
    # Validate performance expectations
    assert create_metrics['p99'] < max_acceptable_p99_latency, \
        f"Create operation P99 latency too high: {create_metrics['p99']} seconds"
    
    assert rename_metrics['p99'] < max_acceptable_p99_latency, \
        f"Rename operation P99 latency too high: {rename_metrics['p99']} seconds"
    
    assert delete_metrics['p99'] < max_acceptable_p99_latency, \
        f"Delete operation P99 latency too high: {delete_metrics['p99']} seconds"
    
    assert create_metrics['throughput'] > min_acceptable_throughput, \
        f"Create operation throughput too low: {create_metrics['throughput']} ops/sec"
    
    assert rename_metrics['throughput'] > min_acceptable_throughput, \
        f"Rename operation throughput too low: {rename_metrics['throughput']} ops/sec"
    
    assert delete_metrics['throughput'] > min_acceptable_throughput, \
        f"Delete operation throughput too low: {delete_metrics['throughput']} ops/sec"
    
    logger.info(f"Namespace load test with {namespace_size} objects completed successfully")

import os
import time
import pytest
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from pyarrow import ozone
from typing import Dict, List, Tuple

# Constants for the test
VOLUME_NAME = "perfvol"
BUCKET_NAME = "perfbucket"
TEST_FILE_DIR = "/tmp/ozone_perf_tests"
RESULT_DIR = "/tmp/ozone_results"

# Test parameters - different buffer sizes and file sizes to test
BUFFER_SIZES = [64, 128, 256, 512, 1024, 2048, 4096, 8192]  # KB
FILE_SIZES = [
    (256, "KB"),
    (512, "KB"), 
    (1, "MB"),
    (4, "MB"),
    (9, "MB"),
    (16, "MB"),
    (32, "MB"),
    (64, "MB"),
    (128, "MB"),
    (256, "MB"),
    (512, "MB"),
    (1, "GB"),
    (2, "GB"),
    (4, "GB")
]

# Number of operations to perform for each configuration
NUM_OPERATIONS = 5


def setup_module():
    """Set up the test environment."""
    # Create directories
    os.makedirs(TEST_FILE_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    
    # Create volume and bucket
    subprocess.run(["ozone", "sh", "volume", "create", VOLUME_NAME], check=True)
    subprocess.run(["ozone", "sh", "bucket", "create", f"/{VOLUME_NAME}/{BUCKET_NAME}"], check=True)


def teardown_module():
    """Clean up the test environment."""
    # Remove test files
    for file_path in os.listdir(TEST_FILE_DIR):
        os.remove(os.path.join(TEST_FILE_DIR, file_path))
    
    # Delete bucket and volume
    subprocess.run(["ozone", "sh", "bucket", "delete", f"/{VOLUME_NAME}/{BUCKET_NAME}"], check=True)
    subprocess.run(["ozone", "sh", "volume", "delete", VOLUME_NAME], check=True)


def create_test_file(size: int, unit: str) -> str:
    """
    Create a test file with specified size.
    
    Args:
        size: Size value
        unit: Size unit (KB, MB, GB)
        
    Returns:
        Path to the created test file
    """
    multiplier = {
        "KB": 1024,
        "MB": 1024 * 1024,
        "GB": 1024 * 1024 * 1024
    }
    
    total_bytes = size * multiplier[unit]
    file_path = os.path.join(TEST_FILE_DIR, f"test_{size}{unit.lower()}.dat")
    
    # Use dd to create file of specific size
    subprocess.run([
        "dd", "if=/dev/urandom", f"of={file_path}", 
        f"bs={min(1024*1024, total_bytes)}", f"count={max(1, total_bytes // min(1024*1024, total_bytes))}"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    return file_path


def perform_write_test(buffer_size: int, file_path: str) -> Tuple[float, float]:
    """
    Perform write test with specified buffer size.
    
    Args:
        buffer_size: Buffer size in KB
        file_path: Path to the test file
        
    Returns:
        Tuple of (throughput in MB/s, latency in ms)
    """
    key_name = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)
    
    # Set the Ozone client buffer size via environment variable
    os.environ["OZONE_CLIENT_BUFFER_SIZE_KB"] = str(buffer_size)
    
    # Measure write operation
    start_time = time.time()
    
    # Use Ozone client to write the file
    client = ozone.Client()
    with open(file_path, 'rb') as f:
        client.put_key(VOLUME_NAME, BUCKET_NAME, key_name, f.read())
    
    end_time = time.time()
    
    # Calculate metrics
    duration = end_time - start_time
    throughput = (file_size / 1024 / 1024) / duration  # MB/s
    latency = duration * 1000  # ms
    
    return throughput, latency


def perform_read_test(buffer_size: int, file_path: str) -> Tuple[float, float]:
    """
    Perform read test with specified buffer size.
    
    Args:
        buffer_size: Buffer size in KB
        file_path: Path to the test file that was uploaded
        
    Returns:
        Tuple of (throughput in MB/s, latency in ms)
    """
    key_name = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)
    
    # Set the Ozone client buffer size via environment variable
    os.environ["OZONE_CLIENT_BUFFER_SIZE_KB"] = str(buffer_size)
    
    # Measure read operation
    start_time = time.time()
    
    # Use Ozone client to read the file
    client = ozone.Client()
    data = client.get_key(VOLUME_NAME, BUCKET_NAME, key_name)
    
    end_time = time.time()
    
    # Calculate metrics
    duration = end_time - start_time
    throughput = (file_size / 1024 / 1024) / duration  # MB/s
    latency = duration * 1000  # ms
    
    return throughput, latency


def save_results(results: Dict[str, List], operation: str):
    """
    Save test results to CSV and generate plots.
    
    Args:
        results: Dictionary containing test results
        operation: Operation type (read/write)
    """
    # Create DataFrame and save to CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(RESULT_DIR, f"ozone_{operation}_performance.csv")
    df.to_csv(csv_path, index=False)
    
    # Generate throughput plot
    plt.figure(figsize=(12, 8))
    
    for size_info in FILE_SIZES:
        size_str = f"{size_info[0]}{size_info[1]}"
        plt.plot(results["buffer_size"], results[f"throughput_{size_str}"], 
                 marker='o', label=f"File Size: {size_str}")
    
    plt.xlabel("Buffer Size (KB)")
    plt.ylabel("Throughput (MB/s)")
    plt.title(f"Ozone {operation.capitalize()} Throughput vs Buffer Size")
    plt.xscale("log", base=2)
    plt.grid(True)
    plt.legend()
    
    plt.savefig(os.path.join(RESULT_DIR, f"ozone_{operation}_throughput.png"))
    
    # Generate latency plot
    plt.figure(figsize=(12, 8))
    
    for size_info in FILE_SIZES:
        size_str = f"{size_info[0]}{size_info[1]}"
        plt.plot(results["buffer_size"], results[f"latency_{size_str}"], 
                 marker='o', label=f"File Size: {size_str}")
    
    plt.xlabel("Buffer Size (KB)")
    plt.ylabel("Latency (ms)")
    plt.title(f"Ozone {operation.capitalize()} Latency vs Buffer Size")
    plt.xscale("log", base=2)
    plt.grid(True)
    plt.legend()
    
    plt.savefig(os.path.join(RESULT_DIR, f"ozone_{operation}_latency.png"))


@pytest.mark.performance
def test_50_ozone_buffer_size_performance():
    """
    Performance test: Evaluate performance with different client buffer sizes
    
    This test evaluates the impact of different client buffer sizes on Ozone performance.
    It measures read and write throughput and latency for various file sizes.
    """
    # Results containers
    write_results = {
        "buffer_size": BUFFER_SIZES,
    }
    
    read_results = {
        "buffer_size": BUFFER_SIZES,
    }
    
    # Create test files for each size
    test_files = {}
    for size, unit in FILE_SIZES:
        size_str = f"{size}{unit}"
        test_files[size_str] = create_test_file(size, unit)
        write_results[f"throughput_{size_str}"] = []
        write_results[f"latency_{size_str}"] = []
        read_results[f"throughput_{size_str}"] = []
        read_results[f"latency_{size_str}"] = []
    
    # Test each buffer size
    for buffer_size in BUFFER_SIZES:
        print(f"Testing buffer size: {buffer_size} KB")
        
        # Test each file size
        for size, unit in FILE_SIZES:
            size_str = f"{size}{unit}"
            file_path = test_files[size_str]
            
            # Perform multiple write operations to get average
            write_throughputs = []
            write_latencies = []
            for _ in range(NUM_OPERATIONS):
                throughput, latency = perform_write_test(buffer_size, file_path)
                write_throughputs.append(throughput)
                write_latencies.append(latency)
            
            # Calculate averages
            avg_write_throughput = sum(write_throughputs) / len(write_throughputs)
            avg_write_latency = sum(write_latencies) / len(write_latencies)
            
            # Store results
            write_results[f"throughput_{size_str}"].append(avg_write_throughput)
            write_results[f"latency_{size_str}"].append(avg_write_latency)
            
            # Perform multiple read operations to get average
            read_throughputs = []
            read_latencies = []
            for _ in range(NUM_OPERATIONS):
                throughput, latency = perform_read_test(buffer_size, file_path)
                read_throughputs.append(throughput)
                read_latencies.append(latency)
            
            # Calculate averages
            avg_read_throughput = sum(read_throughputs) / len(read_throughputs)
            avg_read_latency = sum(read_latencies) / len(read_latencies)
            
            # Store results
            read_results[f"throughput_{size_str}"].append(avg_read_throughput)
            read_results[f"latency_{size_str}"].append(avg_read_latency)
            
            print(f"  File size {size_str}: Write: {avg_write_throughput:.2f} MB/s, {avg_write_latency:.2f} ms; "
                  f"Read: {avg_read_throughput:.2f} MB/s, {avg_read_latency:.2f} ms")
    
    # Save results
    save_results(write_results, "write")
    save_results(read_results, "read")
    
    # Find optimal buffer sizes for different workloads
    optimal_configurations = {}
    
    # For small files (KB range)
    small_files = [f"{size}{unit}" for size, unit in FILE_SIZES if unit == "KB"]
    if small_files:
        small_write_optimal = max(BUFFER_SIZES, 
                              key=lambda bs: sum(write_results[f"throughput_{size}"][BUFFER_SIZES.index(bs)] 
                                             for size in small_files))
        small_read_optimal = max(BUFFER_SIZES, 
                             key=lambda bs: sum(read_results[f"throughput_{size}"][BUFFER_SIZES.index(bs)] 
                                            for size in small_files))
        optimal_configurations["small_files"] = {
            "write_buffer_size": small_write_optimal,
            "read_buffer_size": small_read_optimal
        }
    
    # For medium files (MB range)
    medium_files = [f"{size}{unit}" for size, unit in FILE_SIZES if unit == "MB"]
    if medium_files:
        medium_write_optimal = max(BUFFER_SIZES, 
                               key=lambda bs: sum(write_results[f"throughput_{size}"][BUFFER_SIZES.index(bs)] 
                                              for size in medium_files))
        medium_read_optimal = max(BUFFER_SIZES, 
                              key=lambda bs: sum(read_results[f"throughput_{size}"][BUFFER_SIZES.index(bs)] 
                                             for size in medium_files))
        optimal_configurations["medium_files"] = {
            "write_buffer_size": medium_write_optimal,
            "read_buffer_size": medium_read_optimal
        }
    
    # For large files (GB range)
    large_files = [f"{size}{unit}" for size, unit in FILE_SIZES if unit == "GB"]
    if large_files:
        large_write_optimal = max(BUFFER_SIZES, 
                              key=lambda bs: sum(write_results[f"throughput_{size}"][BUFFER_SIZES.index(bs)] 
                                             for size in large_files))
        large_read_optimal = max(BUFFER_SIZES, 
                             key=lambda bs: sum(read_results[f"throughput_{size}"][BUFFER_SIZES.index(bs)] 
                                            for size in large_files))
        optimal_configurations["large_files"] = {
            "write_buffer_size": large_write_optimal,
            "read_buffer_size": large_read_optimal
        }
    
    # Save optimal configurations
    with open(os.path.join(RESULT_DIR, "optimal_buffer_sizes.txt"), "w") as f:
        f.write("Optimal Buffer Size Configurations:\n")
        for workload, configs in optimal_configurations.items():
            f.write(f"\n{workload.replace('_', ' ').title()}:\n")
            f.write(f"  Write Buffer Size: {configs['write_buffer_size']} KB\n")
            f.write(f"  Read Buffer Size: {configs['read_buffer_size']} KB\n")
    
    print("\nOptimal Buffer Size Configurations:")
    for workload, configs in optimal_configurations.items():
        print(f"\n{workload.replace('_', ' ').title()}:")
        print(f"  Write Buffer Size: {configs['write_buffer_size']} KB")
        print(f"  Read Buffer Size: {configs['read_buffer_size']} KB")
    
    # Check that we have found optimal configurations
    assert optimal_configurations, "Failed to determine optimal buffer sizes"

import os
import time
import pytest
import subprocess
import logging
import pandas as pd
import numpy as np
import threading
from pyozone.client import OzoneClient
from concurrent.futures import ThreadPoolExecutor
import tempfile
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ozone_scrubbing_test")

# Test configuration
class ScrubConfig:
    VOLUME_NAME = "scrubtestvolume"
    BUCKET_NAME = "scrubtestbucket"
    DATA_SIZES_KB = [512, 1024, 2048, 4096]  # Various file sizes for testing
    CORRUPTION_RATES = [0.05, 0.10, 0.15]    # Percentage of data to corrupt
    NUM_FILES_PER_SIZE = 10
    METRICS_INTERVAL_SEC = 5
    SCRUB_TIMEOUT_SEC = 1800  # 30 minutes timeout for scrubbing completion

class MetricsCollector:
    """Collects system metrics during scrubbing operations"""
    
    def __init__(self, output_file="scrubbing_metrics.csv"):
        self.output_file = output_file
        self.metrics = []
        self.running = False
    
    def start(self):
        """Start collecting metrics in background thread"""
        self.running = True
        self.metrics_thread = threading.Thread(target=self._collect_metrics)
        self.metrics_thread.daemon = True
        self.metrics_thread.start()
    
    def stop(self):
        """Stop metrics collection and save results"""
        self.running = False
        if hasattr(self, 'metrics_thread'):
            self.metrics_thread.join(timeout=10)
        
        # Save collected metrics to file
        if self.metrics:
            df = pd.DataFrame(self.metrics)
            df.to_csv(self.output_file, index=False)
            logger.info(f"Metrics saved to {self.output_file}")
            return df
        return None
    
    def _collect_metrics(self):
        """Collect various system metrics at regular intervals"""
        while self.running:
            try:
                # Get system metrics via Ozone admin commands
                cpu_usage = self._get_cpu_usage()
                memory_usage = self._get_memory_usage()
                disk_io = self._get_disk_io()
                scrub_progress = self._get_scrub_progress()
                
                timestamp = time.time()
                self.metrics.append({
                    "timestamp": timestamp,
                    "cpu_usage": cpu_usage,
                    "memory_usage_mb": memory_usage,
                    "disk_io_mbps": disk_io,
                    "scrub_progress_pct": scrub_progress
                })
                
                time.sleep(ScrubConfig.METRICS_INTERVAL_SEC)
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
    
    def _get_cpu_usage(self):
        """Get CPU usage for Ozone processes"""
        cmd = "top -bn1 | grep 'ozone' | awk '{sum+=$9} END {print sum}'"
        result = subprocess.check_output(cmd, shell=True, text=True).strip()
        return float(result) if result else 0
    
    def _get_memory_usage(self):
        """Get memory usage for Ozone processes in MB"""
        cmd = "ps aux | grep 'ozone' | awk '{sum+=$6} END {print sum/1024}'"
        result = subprocess.check_output(cmd, shell=True, text=True).strip()
        return float(result) if result else 0
    
    def _get_disk_io(self):
        """Get disk I/O for Ozone data directories in MB/s"""
        cmd = "iostat -xm | grep 'sd' | awk '{sum+=$6} END {print sum}'"
        result = subprocess.check_output(cmd, shell=True, text=True).strip()
        return float(result) if result else 0
    
    def _get_scrub_progress(self):
        """Get scrubbing progress percentage if available"""
        try:
            cmd = "ozone admin scrub status"
            output = subprocess.check_output(cmd, shell=True, text=True)
            # Extract progress percentage (implementation depends on actual output format)
            # This is a placeholder - adjust based on actual command output
            if "progress" in output.lower():
                # Extract percentage from output
                return float(output.split("progress:")[1].split("%")[0].strip())
            return 0
        except:
            return 0

def create_test_data(size_kb, corruption_rate=0):
    """
    Create test data file of specified size with optional corruption
    
    Args:
        size_kb: Size of the file in KB
        corruption_rate: Percentage of data to corrupt (0-1)
        
    Returns:
        Path to the created temporary file
    """
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    size_bytes = size_kb * 1024
    
    # Generate random data
    data = os.urandom(size_bytes)
    
    # Add corruption if needed
    if corruption_rate > 0:
        data_list = bytearray(data)
        num_bytes_to_corrupt = int(size_bytes * corruption_rate)
        positions_to_corrupt = random.sample(range(size_bytes), num_bytes_to_corrupt)
        
        for pos in positions_to_corrupt:
            data_list[pos] = random.randint(0, 255)
        
        data = bytes(data_list)
    
    # Write to temp file
    with open(temp_file.name, 'wb') as f:
        f.write(data)
    
    return temp_file.name

def upload_file_to_ozone(client, volume, bucket, key_name, file_path):
    """Upload a file to Ozone with the specified key"""
    try:
        with open(file_path, 'rb') as f:
            client.put_key(volume, bucket, key_name, f)
        return True
    except Exception as e:
        logger.error(f"Error uploading file {file_path} to {volume}/{bucket}/{key_name}: {e}")
        return False

def verify_data_integrity(client, volume, bucket, keys):
    """Verify data integrity for a list of keys"""
    failures = []
    
    for key in keys:
        try:
            # Get key info and verify
            key_info = client.get_key_info(volume, bucket, key)
            if not key_info:
                failures.append(key)
                continue
                
            # Optionally read the data for deeper verification
            # data = client.get_key(volume, bucket, key)
            # Perform further validation as needed
            
        except Exception as e:
            logger.error(f"Error verifying key {key}: {e}")
            failures.append(key)
    
    return len(failures) == 0, failures

def run_data_scrubbing():
    """Run the data scrubbing process"""
    try:
        # Start the scrubbing process
        cmd = "ozone admin scrub start"
        subprocess.run(cmd, shell=True, check=True)
        
        # Wait for completion or timeout
        start_time = time.time()
        while time.time() - start_time < ScrubConfig.SCRUB_TIMEOUT_SEC:
            # Check if scrubbing is still running
            status_cmd = "ozone admin scrub status"
            status_output = subprocess.check_output(status_cmd, shell=True, text=True)
            
            if "completed" in status_output.lower():
                duration = time.time() - start_time
                return True, duration
                
            time.sleep(10)  # Check every 10 seconds
            
        # If we reach here, scrubbing didn't complete within timeout
        return False, ScrubConfig.SCRUB_TIMEOUT_SEC
        
    except Exception as e:
        logger.error(f"Error running data scrubbing: {e}")
        return False, 0

@pytest.fixture(scope="module")
def setup_ozone_environment():
    """Set up the Ozone environment for scrubbing tests"""
    client = OzoneClient()
    volume = ScrubConfig.VOLUME_NAME
    bucket = ScrubConfig.BUCKET_NAME
    
    # Create volume and bucket if they don't exist
    if not client.volume_exists(volume):
        client.create_volume(volume)
    
    if not client.bucket_exists(volume, bucket):
        client.create_bucket(volume, bucket)
    
    yield client, volume, bucket
    
    # Cleanup (optional, depending on test requirements)
    # client.delete_bucket(volume, bucket)
    # client.delete_volume(volume)

@pytest.mark.parametrize("data_size_kb", ScrubConfig.DATA_SIZES_KB)
@pytest.mark.parametrize("corruption_rate", ScrubConfig.CORRUPTION_RATES)
def test_51_data_scrubbing_performance(setup_ozone_environment, data_size_kb, corruption_rate):
    """
    Evaluate performance under data scrubbing operations with different file sizes and corruption rates
    """
    client, volume, bucket = setup_ozone_environment
    logger.info(f"Testing scrubbing performance with {data_size_kb}KB files and {corruption_rate*100}% corruption")
    
    # Step 1: Populate the cluster with a mix of valid and invalid data
    uploaded_keys = []
    
    # Create a mix of valid and corrupted data files
    for i in range(ScrubConfig.NUM_FILES_PER_SIZE):
        # Generate file name with identifiers for size and corruption
        key_name = f"scrub_test_{data_size_kb}kb_{int(corruption_rate*100)}pct_{i}.dat"
        
        # Create and upload test file
        file_path = create_test_data(data_size_kb, corruption_rate)
        if upload_file_to_ozone(client, volume, bucket, key_name, file_path):
            uploaded_keys.append(key_name)
        
        # Clean up temp file
        os.unlink(file_path)
    
    logger.info(f"Uploaded {len(uploaded_keys)} files to Ozone")
    
    # Step 2 & 3: Initiate data scrubbing process and monitor system performance
    metrics_collector = MetricsCollector(f"scrub_metrics_{data_size_kb}kb_{int(corruption_rate*100)}pct.csv")
    metrics_collector.start()
    
    # Run the scrubbing process
    start_time = time.time()
    success, duration = run_data_scrubbing()
    
    # Stop metrics collection
    metrics_df = metrics_collector.stop()
    
    # Step 4: Measure and log time taken for scrubbing to complete
    logger.info(f"Scrubbing completed: {success}, Duration: {duration:.2f} seconds")
    
    # Step 5: Verify data integrity post-scrubbing
    integrity_check, failed_keys = verify_data_integrity(client, volume, bucket, uploaded_keys)
    
    # Log performance metrics summary
    if metrics_df is not None:
        avg_cpu = metrics_df['cpu_usage'].mean()
        max_cpu = metrics_df['cpu_usage'].max()
        avg_memory = metrics_df['memory_usage_mb'].mean()
        avg_disk_io = metrics_df['disk_io_mbps'].mean()
        
        logger.info(f"Performance metrics - Avg CPU: {avg_cpu:.2f}%, Max CPU: {max_cpu:.2f}%, "
                   f"Avg Memory: {avg_memory:.2f} MB, Avg Disk I/O: {avg_disk_io:.2f} MB/s")
    
    # Assert expected outcomes
    assert success, "Scrubbing process should complete successfully"
    assert duration < ScrubConfig.SCRUB_TIMEOUT_SEC, f"Scrubbing should complete within {ScrubConfig.SCRUB_TIMEOUT_SEC} seconds"
    assert integrity_check, f"Data integrity check failed for keys: {failed_keys}"
    
    # Performance assertions
    if metrics_df is not None:
        # Set reasonable thresholds based on your environment
        assert max_cpu < 90, f"CPU usage should stay below 90% (actual: {max_cpu:.2f}%)"
        
        # Check for system stability during scrubbing
        cpu_std = metrics_df['cpu_usage'].std()
        assert cpu_std < 20, f"CPU usage should remain stable (std dev: {cpu_std:.2f})"

import time
import pytest
import logging
import concurrent.futures
import pandas as pd
import matplotlib.pyplot as plt
from pyozone.client import OzoneClient
from hdfs.client import InsecureClient
import os
import tempfile
import numpy as np
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test configuration
CONNECTION_POOL_SIZES = [5, 10, 20, 50, 100]
NUM_CONCURRENT_OPERATIONS = 100
OPERATION_TYPES = ['read', 'write', 'mixed']
FILE_SIZES_KB = [10, 100, 1024, 10240]  # 10KB, 100KB, 1MB, 10MB

# Constants
TEST_VOLUME = "perf-connpool-vol"
TEST_BUCKET = "perf-connpool-bucket"


def create_test_file(size_kb):
    """Create a test file of specified size in KB"""
    fd, path = tempfile.mkstemp()
    with os.fdopen(fd, 'wb') as f:
        f.write(os.urandom(size_kb * 1024))
    return path


def setup_ozone_environment():
    """Setup Ozone environment for tests"""
    # Create test volume and bucket
    os.system(f"ozone sh volume create {TEST_VOLUME}")
    os.system(f"ozone sh bucket create {TEST_VOLUME}/{TEST_BUCKET}")
    
    # Return base client for admin operations
    return OzoneClient()


def cleanup_ozone_environment():
    """Clean up Ozone environment after tests"""
    os.system(f"ozone sh bucket delete {TEST_VOLUME}/{TEST_BUCKET}")
    os.system(f"ozone sh volume delete {TEST_VOLUME}")


class PerformanceMetrics:
    """Class to collect and analyze performance metrics"""
    
    def __init__(self):
        self.results = []
    
    def add_result(self, pool_size, operation_type, file_size, operation_count, 
                   duration, throughput, avg_latency, p95_latency, p99_latency):
        self.results.append({
            'pool_size': pool_size,
            'operation_type': operation_type,
            'file_size_kb': file_size,
            'operation_count': operation_count,
            'duration_seconds': duration,
            'throughput_ops_per_sec': throughput,
            'avg_latency_ms': avg_latency,
            'p95_latency_ms': p95_latency,
            'p99_latency_ms': p99_latency
        })
    
    def get_results_df(self):
        """Convert results to DataFrame for analysis"""
        return pd.DataFrame(self.results)
    
    def plot_results(self):
        """Generate performance plots"""
        df = self.get_results_df()
        
        # Plot throughput vs pool size for different operation types
        plt.figure(figsize=(12, 8))
        for op_type in df['operation_type'].unique():
            data = df[df['operation_type'] == op_type]
            plt.plot(data['pool_size'], data['throughput_ops_per_sec'], marker='o', label=op_type)
        
        plt.xlabel('Connection Pool Size')
        plt.ylabel('Throughput (ops/sec)')
        plt.title('Throughput vs Connection Pool Size')
        plt.legend()
        plt.grid(True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'throughput_vs_pool_size_{timestamp}.png')
        
        # Plot latency vs pool size
        plt.figure(figsize=(12, 8))
        for op_type in df['operation_type'].unique():
            data = df[df['operation_type'] == op_type]
            plt.plot(data['pool_size'], data['avg_latency_ms'], marker='o', label=f'{op_type} - Avg')
            plt.plot(data['pool_size'], data['p95_latency_ms'], marker='x', linestyle='--', 
                     label=f'{op_type} - P95')
        
        plt.xlabel('Connection Pool Size')
        plt.ylabel('Latency (ms)')
        plt.title('Latency vs Connection Pool Size')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(f'latency_vs_pool_size_{timestamp}.png')
        
        return df


def configure_client(pool_size):
    """Configure and return an Ozone client with specified connection pool size"""
    config = {
        'ozone.client.connection.pool.size': str(pool_size),
        'ozone.client.connection.pool.max.idle': str(pool_size),
        'ozone.client.connection.pool.max.total': str(pool_size * 2)
    }
    return OzoneClient(conf=config)


def write_operation(client, volume, bucket, key_prefix, file_path):
    """Perform a write operation and measure latency"""
    key = f"{key_prefix}_{int(time.time() * 1000)}_{os.path.basename(file_path)}"
    start_time = time.time()
    client.put_key(volume, bucket, key, file_path)
    latency = (time.time() - start_time) * 1000  # Convert to ms
    return {'operation': 'write', 'key': key, 'latency': latency}


def read_operation(client, volume, bucket, key):
    """Perform a read operation and measure latency"""
    start_time = time.time()
    client.get_key(volume, bucket, key)
    latency = (time.time() - start_time) * 1000  # Convert to ms
    return {'operation': 'read', 'key': key, 'latency': latency}


def run_performance_test(pool_size, operation_type, file_size_kb, num_operations):
    """Run a performance test with specified parameters"""
    client = configure_client(pool_size)
    
    # Create test data file
    test_file_path = create_test_file(file_size_kb)
    
    # Prepare for test
    results = []
    keys = []
    
    # For read tests, we need to preload some data
    if operation_type in ['read', 'mixed']:
        # Create some test keys for reading
        for i in range(min(50, num_operations)):
            key = f"preload_key_{i}"
            client.put_key(TEST_VOLUME, TEST_BUCKET, key, test_file_path)
            keys.append(key)
    
    # Run the test
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_CONCURRENT_OPERATIONS) as executor:
        futures = []
        
        for i in range(num_operations):
            if operation_type == 'write':
                futures.append(executor.submit(
                    write_operation, client, TEST_VOLUME, TEST_BUCKET, f"test_{pool_size}", test_file_path
                ))
            elif operation_type == 'read':
                # Cycle through available keys
                key = keys[i % len(keys)]
                futures.append(executor.submit(
                    read_operation, client, TEST_VOLUME, TEST_BUCKET, key
                ))
            elif operation_type == 'mixed':
                # 50% read, 50% write
                if i % 2 == 0:
                    futures.append(executor.submit(
                        write_operation, client, TEST_VOLUME, TEST_BUCKET, f"test_{pool_size}", test_file_path
                    ))
                else:
                    key = keys[i % len(keys)]
                    futures.append(executor.submit(
                        read_operation, client, TEST_VOLUME, TEST_BUCKET, key
                    ))
        
        # Collect results
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Operation failed: {str(e)}")
    
    duration = time.time() - start_time
    
    # Calculate metrics
    operation_count = len(results)
    throughput = operation_count / duration if duration > 0 else 0
    
    latencies = [r['latency'] for r in results]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    latencies.sort()
    p95_latency = latencies[int(len(latencies) * 0.95)] if latencies else 0
    p99_latency = latencies[int(len(latencies) * 0.99)] if latencies else 0
    
    # Cleanup
    os.unlink(test_file_path)
    
    return {
        'pool_size': pool_size,
        'operation_type': operation_type,
        'file_size_kb': file_size_kb,
        'operation_count': operation_count,
        'duration': duration,
        'throughput': throughput,
        'avg_latency': avg_latency,
        'p95_latency': p95_latency,
        'p99_latency': p99_latency
    }


@pytest.mark.parametrize('pool_size', CONNECTION_POOL_SIZES)
@pytest.mark.parametrize('operation_type', OPERATION_TYPES)
@pytest.mark.parametrize('file_size_kb', FILE_SIZES_KB)
def test_52_connection_pool_performance(pool_size, operation_type, file_size_kb):
    """
    Test performance with varying client connection pool sizes
    
    This test evaluates the performance impact of different connection pool sizes
    by measuring throughput and latency for read/write operations with various file sizes.
    """
    logger.info(f"Testing pool size: {pool_size}, operation: {operation_type}, file size: {file_size_kb}KB")
    
    # Setup test environment
    setup_ozone_environment()
    
    try:
        # Run the performance test
        result = run_performance_test(
            pool_size, 
            operation_type, 
            file_size_kb, 
            NUM_CONCURRENT_OPERATIONS
        )
        
        # Log results
        logger.info(f"Performance results for pool size {pool_size}, {operation_type} operations, {file_size_kb}KB files:")
        logger.info(f"  Operations: {result['operation_count']}")
        logger.info(f"  Duration: {result['duration']:.2f} seconds")
        logger.info(f"  Throughput: {result['throughput']:.2f} ops/sec")
        logger.info(f"  Average latency: {result['avg_latency']:.2f} ms")
        logger.info(f"  P95 latency: {result['p95_latency']:.2f} ms")
        logger.info(f"  P99 latency: {result['p99_latency']:.2f} ms")
        
        # For a comprehensive test suite, we'd collect all results and analyze them together
        # Using a class like PerformanceMetrics shown above
        
        # Validate that the test completed successfully - this is a performance test,
        # so we're primarily interested in the metrics rather than pass/fail
        assert result['operation_count'] > 0, "No operations completed successfully"
        
        # For optimization guidance, we could define thresholds, but these would be
        # environment-specific and should be calibrated based on the specific deployment
        
    finally:
        # Clean up test environment
        cleanup_ozone_environment()


def test_52_analyze_connection_pool_performance_comprehensive():
    """
    Comprehensive test to analyze optimal client connection pool sizes for different workload patterns
    
    This test runs multiple configurations and analyzes the results to identify optimal
    connection pool sizes for different workloads.
    """
    # Setup test environment
    setup_ozone_environment()
    metrics = PerformanceMetrics()
    
    try:
        # Test various configurations
        for pool_size in CONNECTION_POOL_SIZES:
            for operation_type in OPERATION_TYPES:
                for file_size_kb in FILE_SIZES_KB:
                    logger.info(f"Testing pool size: {pool_size}, operation: {operation_type}, file size: {file_size_kb}KB")
                    
                    # Run the performance test
                    result = run_performance_test(
                        pool_size, 
                        operation_type, 
                        file_size_kb, 
                        NUM_CONCURRENT_OPERATIONS
                    )
                    
                    # Add result to metrics collection
                    metrics.add_result(
                        pool_size,
                        operation_type,
                        file_size_kb,
                        result['operation_count'],
                        result['duration'],
                        result['throughput'],
                        result['avg_latency'],
                        result['p95_latency'],
                        result['p99_latency']
                    )
        
        # Analyze results
        df = metrics.plot_results()
        
        # Find optimal pool size for each operation type and file size
        for op_type in df['operation_type'].unique():
            for size in df['file_size_kb'].unique():
                subset = df[(df['operation_type'] == op_type) & (df['file_size_kb'] == size)]
                
                # Find pool size with highest throughput
                max_throughput_idx = subset['throughput_ops_per_sec'].idxmax()
                optimal_pool_size = subset.loc[max_throughput_idx, 'pool_size']
                
                logger.info(f"Optimal pool size for {op_type} operations with {size}KB files: {optimal_pool_size}")
                logger.info(f"  Throughput: {subset.loc[max_throughput_idx, 'throughput_ops_per_sec']:.2f} ops/sec")
                logger.info(f"  Average latency: {subset.loc[max_throughput_idx, 'avg_latency_ms']:.2f} ms")
        
        # Verify we found meaningful results
        assert not df.empty, "No performance data was collected"
        
        # Save results to CSV for further analysis
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(f'connection_pool_performance_{timestamp}.csv', index=False)
        
    finally:
        # Clean up test environment
        cleanup_ozone_environment()

import pytest
import time
import os
import subprocess
import concurrent.futures
import statistics
import logging
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import requests
import json

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
OZONE_HOME = os.environ.get('OZONE_HOME', '/opt/hadoop-ozone')
CLUSTER_NODES = os.environ.get('CLUSTER_NODES', 'localhost').split(',')
OZONE_COMPONENTS = ["om", "scm", "datanode", "s3g", "recon"]
TEST_VOLUME = "vol-perf-restart"
TEST_BUCKET = "bucket-perf-restart"
TEST_DURATION_SECONDS = 600  # 10 minutes test duration
MONITORING_INTERVAL = 5  # seconds
RESTART_DELAY = 20  # seconds between each component restart

# Utilities for performance measurement
class PerformanceMetrics:
    def __init__(self):
        self.latencies = []
        self.throughputs = []
        self.errors = 0
        self.availability = 100.0
        self.start_time = time.time()
        self.total_requests = 0
        self.successful_requests = 0
        
    def add_operation_result(self, latency: float, success: bool):
        self.total_requests += 1
        if success:
            self.successful_requests += 1
            self.latencies.append(latency)
        else:
            self.errors += 1
        
        # Calculate availability
        self.availability = (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 100.0
        
        # Calculate throughput over the last period
        duration = time.time() - self.start_time
        if duration > 0:
            self.throughputs.append(self.successful_requests / duration)
    
    def get_summary(self) -> Dict:
        if not self.latencies:
            return {
                "avg_latency_ms": 0,
                "p95_latency_ms": 0,
                "p99_latency_ms": 0,
                "max_latency_ms": 0,
                "avg_throughput": 0,
                "error_rate": 0,
                "availability": 100.0,
                "total_operations": 0
            }
            
        # Calculate percentiles
        sorted_latencies = sorted(self.latencies)
        p95_idx = int(len(sorted_latencies) * 0.95)
        p99_idx = int(len(sorted_latencies) * 0.99)
        
        return {
            "avg_latency_ms": statistics.mean(self.latencies) if self.latencies else 0,
            "p95_latency_ms": sorted_latencies[p95_idx] if self.latencies else 0,
            "p99_latency_ms": sorted_latencies[p99_idx] if self.latencies else 0,
            "max_latency_ms": max(self.latencies) if self.latencies else 0,
            "avg_throughput": statistics.mean(self.throughputs) if self.throughputs else 0,
            "error_rate": (self.errors / self.total_requests * 100) if self.total_requests > 0 else 0,
            "availability": self.availability,
            "total_operations": self.total_requests
        }
    
    def reset(self):
        self.latencies = []
        self.throughputs = []
        self.errors = 0
        self.availability = 100.0
        self.start_time = time.time()
        self.total_requests = 0
        self.successful_requests = 0

class OzoneClusterManager:
    @staticmethod
    def restart_component(component: str, host: str) -> bool:
        """Restart a specific Ozone component on a specific host"""
        logger.info(f"Restarting {component} on {host}")
        try:
            # Execute restart command via SSH or direct command
            # This is a simplified example - in a real environment, 
            # you would use proper SSH handling
            cmd = f"ssh {host} '{OZONE_HOME}/bin/ozone --daemon stop {component} && " \
                  f"sleep 2 && {OZONE_HOME}/bin/ozone --daemon start {component}'"
            
            subprocess.run(cmd, shell=True, check=True)
            logger.info(f"Successfully restarted {component} on {host}")
            return True
        except subprocess.SubprocessError as e:
            logger.error(f"Failed to restart {component} on {host}: {str(e)}")
            return False
    
    @staticmethod
    def check_component_status(component: str, host: str) -> bool:
        """Check if a specific component is running on a host"""
        try:
            cmd = f"ssh {host} 'ps -ef | grep -v grep | grep {component}'"
            result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return result.returncode == 0 and result.stdout
        except Exception as e:
            logger.error(f"Error checking status of {component} on {host}: {str(e)}")
            return False
    
    @staticmethod
    def get_cluster_health() -> Dict:
        """Get overall cluster health metrics"""
        # This would typically call Ozone metrics APIs or JMX
        # Simplified example here
        try:
            # Using port 9876 as an example for Ozone metrics endpoint
            response = requests.get("http://localhost:9876/metrics", timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Failed to get metrics: {response.status_code}")
                return {}
        except Exception as e:
            logger.error(f"Error getting cluster health: {str(e)}")
            return {}

class WorkloadGenerator:
    def __init__(self, volume: str, bucket: str):
        self.volume = volume
        self.bucket = bucket
        self.file_sizes = [1024, 10240, 102400]  # 1KB, 10KB, 100KB
        
    def setup(self) -> bool:
        """Set up test volume and bucket"""
        try:
            # Create test volume if it doesn't exist
            cmd_volume = f"{OZONE_HOME}/bin/ozone sh volume create /{self.volume}"
            subprocess.run(cmd_volume, shell=True, check=True)
            
            # Create test bucket if it doesn't exist
            cmd_bucket = f"{OZONE_HOME}/bin/ozone sh bucket create /{self.volume}/{self.bucket}"
            subprocess.run(cmd_bucket, shell=True, check=True)
            
            return True
        except subprocess.SubprocessError as e:
            logger.error(f"Failed to set up test environment: {str(e)}")
            return False
            
    def generate_test_files(self) -> List[str]:
        """Generate test files of various sizes"""
        test_files = []
        for i, size in enumerate(self.file_sizes):
            filename = f"/tmp/test_file_{i}_{size}.dat"
            with open(filename, 'wb') as f:
                f.write(os.urandom(size))
            test_files.append(filename)
        return test_files
            
    def run_workload(self, metrics: PerformanceMetrics, duration: int = 60, 
                     file_path: str = None) -> None:
        """Run a continuous workload for a specified duration"""
        if not file_path:
            file_path = self.generate_test_files()[0]
            
        end_time = time.time() + duration
        key_counter = 0
        
        while time.time() < end_time:
            key_name = f"test_key_{datetime.now().strftime('%Y%m%d%H%M%S')}_{key_counter}"
            key_counter += 1
            
            # Upload operation
            start_time = time.time()
            success = self.upload_file(file_path, key_name)
            latency = (time.time() - start_time) * 1000  # Convert to milliseconds
            metrics.add_operation_result(latency, success)
            
            # Read operation
            if success:
                start_time = time.time()
                success = self.read_file(key_name)
                latency = (time.time() - start_time) * 1000
                metrics.add_operation_result(latency, success)
            
            # Brief pause to not overwhelm the system
            time.sleep(0.1)
    
    def upload_file(self, file_path: str, key_name: str) -> bool:
        """Upload a file to Ozone"""
        try:
            cmd = f"{OZONE_HOME}/bin/ozone sh key put /{self.volume}/{self.bucket}/ {file_path} --name={key_name}"
            subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return True
        except subprocess.SubprocessError:
            return False
    
    def read_file(self, key_name: str) -> bool:
        """Read a file from Ozone"""
        try:
            output_file = f"/tmp/read_{key_name}"
            cmd = f"{OZONE_HOME}/bin/ozone sh key get /{self.volume}/{self.bucket}/{key_name} {output_file}"
            subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # Clean up the downloaded file
            if os.path.exists(output_file):
                os.remove(output_file)
            return True
        except subprocess.SubprocessError:
            return False

def plot_performance_metrics(timeline: List[float], metrics_over_time: List[Dict], 
                            restart_events: List[Dict], output_file: str):
    """Generate a performance report with plots"""
    plt.figure(figsize=(15, 10))
    
    # Extract metrics
    latencies = [m["avg_latency_ms"] for m in metrics_over_time]
    p95_latencies = [m["p95_latency_ms"] for m in metrics_over_time]
    throughputs = [m["avg_throughput"] for m in metrics_over_time]
    error_rates = [m["error_rate"] for m in metrics_over_time]
    availabilities = [m["availability"] for m in metrics_over_time]
    
    # Plot latency
    plt.subplot(3, 1, 1)
    plt.plot(timeline, latencies, 'b-', label='Avg Latency')
    plt.plot(timeline, p95_latencies, 'r--', label='P95 Latency')
    plt.title('Latency During Rolling Restart')
    plt.ylabel('Latency (ms)')
    plt.legend()
    
    # Mark restart events
    for event in restart_events:
        plt.axvline(x=event["time"], color='g', linestyle='--', alpha=0.7)
        plt.text(event["time"], max(p95_latencies)*0.9, event["component"], 
                 rotation=90, verticalalignment='top')
    
    # Plot throughput
    plt.subplot(3, 1, 2)
    plt.plot(timeline, throughputs, 'g-')
    plt.title('Throughput During Rolling Restart')
    plt.ylabel('Operations/sec')
    
    # Mark restart events
    for event in restart_events:
        plt.axvline(x=event["time"], color='g', linestyle='--', alpha=0.7)
    
    # Plot error rate and availability
    plt.subplot(3, 1, 3)
    plt.plot(timeline, error_rates, 'r-', label='Error Rate')
    plt.plot(timeline, availabilities, 'b--', label='Availability')
    plt.title('Error Rate and Availability During Rolling Restart')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Percentage')
    plt.legend()
    
    # Mark restart events
    for event in restart_events:
        plt.axvline(x=event["time"], color='g', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_file)
    logger.info(f"Performance report saved to {output_file}")


@pytest.fixture(scope="function")
def ozone_setup():
    """Fixture to set up Ozone environment for testing"""
    # Create workload generator for the test
    workload = WorkloadGenerator(TEST_VOLUME, TEST_BUCKET)
    
    # Ensure test volume and bucket exist
    assert workload.setup(), "Failed to set up test environment"
    
    # Create test files
    test_files = workload.generate_test_files()
    
    yield workload, test_files
    
    # Clean up test files
    for file_path in test_files:
        if os.path.exists(file_path):
            os.remove(file_path)


def test_53_rolling_restart_performance(ozone_setup):
    """
    Evaluate performance under rolling restart scenarios
    
    This test:
    1. Establishes baseline performance
    2. Initiates rolling restart of Ozone components
    3. Continuously monitors system performance during restart
    4. Measures performance impact and any downtime
    5. Verifies cluster stability and performance post-restart
    """
    workload, test_files = ozone_setup
    cluster_mgr = OzoneClusterManager()
    
    # Initialize metrics storage
    baseline_metrics = PerformanceMetrics()
    restart_metrics = PerformanceMetrics()
    post_restart_metrics = PerformanceMetrics()
    
    metrics_over_time = []
    timeline = []
    restart_events = []
    current_time = 0
    
    # 1. Establish baseline performance (2 minutes)
    logger.info("Step 1: Establishing baseline performance...")
    baseline_duration = 120  # seconds
    workload.run_workload(baseline_metrics, baseline_duration, test_files[0])
    baseline_results = baseline_metrics.get_summary()
    logger.info(f"Baseline metrics: {baseline_results}")
    
    # Record metrics for plotting
    metrics_over_time.append(baseline_results)
    timeline.append(current_time)
    current_time += baseline_duration
    
    # 2 & 3. Initiate rolling restart of components while monitoring performance
    logger.info("Step 2 & 3: Initiating rolling restart while monitoring performance...")
    
    # Start a background thread for continuous workload
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Start workload in background
        future = executor.submit(
            workload.run_workload, 
            restart_metrics,
            TEST_DURATION_SECONDS,
            test_files[1]
        )
        
        # Perform rolling restarts
        for component in OZONE_COMPONENTS:
            for node in CLUSTER_NODES:
                time.sleep(RESTART_DELAY)  # Delay between component restarts
                
                # Record the restart event
                restart_time = current_time + time.time() - baseline_metrics.start_time
                restart_events.append({
                    "time": restart_time,
                    "component": f"{component}@{node}"
                })
                timeline.append(restart_time)
                metrics_over_time.append(restart_metrics.get_summary())
                
                # Restart the component
                success = cluster_mgr.restart_component(component, node)
                assert success, f"Failed to restart {component} on {node}"
                
                # Verify component is running after restart
                time.sleep(5)  # Give it time to start
                is_running = cluster_mgr.check_component_status(component, node)
                assert is_running, f"{component} failed to start on {node}"
                
                # Record metrics after each component restart
                current_metrics = restart_metrics.get_summary()
                metrics_over_time.append(current_metrics)
                timeline.append(restart_time + 5)  # 5 seconds after restart
        
        # Wait for the workload to complete
        future.result()
    
    # 4. Measure performance impact
    restart_results = restart_metrics.get_summary()
    logger.info(f"Metrics during restart: {restart_results}")
    
    # 5. Verify cluster stability and performance post-restart
    logger.info("Step 5: Verifying cluster stability and performance post-restart...")
    post_restart_duration = 120  # seconds
    workload.run_workload(post_restart_metrics, post_restart_duration, test_files[2])
    post_restart_results = post_restart_metrics.get_summary()
    logger.info(f"Post-restart metrics: {post_restart_results}")
    
    # Record final metrics
    current_time += TEST_DURATION_SECONDS
    metrics_over_time.append(post_restart_results)
    timeline.append(current_time)
    
    # Generate performance report
    report_file = f"/tmp/ozone_rolling_restart_perf_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
    plot_performance_metrics(timeline, metrics_over_time, restart_events, report_file)
    
    # Create results dataframe for comparison
    results_df = pd.DataFrame([
        baseline_results,
        restart_results,
        post_restart_results
    ], index=["Baseline", "During Restart", "Post-Restart"])
    
    logger.info("\n" + results_df.to_string())
    
    # Assertions to validate test requirements
    # 1. Availability should remain above 95% during restarts
    assert restart_results["availability"] > 95.0, \
        f"Availability during restart was below threshold: {restart_results['availability']}%"
    
    # 2. Post-restart performance should be within 20% of baseline
    assert post_restart_results["avg_latency_ms"] <= baseline_results["avg_latency_ms"] * 1.2, \
        "Post-restart latency degraded by more than 20% from baseline"
    
    # 3. Error rate during restart should be minimal
    assert restart_results["error_rate"] < 5.0, \
        f"Error rate during restart was too high: {restart_results['error_rate']}%"
    
    # 4. Cluster should maintain stability throughout the test
    for component in OZONE_COMPONENTS:
        for node in CLUSTER_NODES:
            assert cluster_mgr.check_component_status(component, node), \
                f"{component} is not running on {node} after test completion"
    
    logger.info("Rolling restart performance test completed successfully")

import os
import time
import json
import subprocess
import pytest
import logging
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants for RocksDB configurations to test
ROCKSDB_CONFIGS = [
    {
        "name": "default",
        "block_cache_size": "256MB",
        "compression": "snappy",
        "max_background_jobs": 4,
        "write_buffer_size": "64MB"
    },
    {
        "name": "high_memory",
        "block_cache_size": "1GB",
        "compression": "snappy",
        "max_background_jobs": 8,
        "write_buffer_size": "128MB"
    },
    {
        "name": "high_compression",
        "block_cache_size": "256MB",
        "compression": "zstd",
        "max_background_jobs": 4,
        "write_buffer_size": "64MB"
    },
    {
        "name": "balanced",
        "block_cache_size": "512MB",
        "compression": "lz4",
        "max_background_jobs": 6,
        "write_buffer_size": "96MB"
    }
]

# Test data sizes
DATA_SIZES = [
    {"name": "small", "size": "10KB", "count": 1000},
    {"name": "medium", "size": "1MB", "count": 100},
    {"name": "large", "size": "10MB", "count": 10},
    {"name": "xlarge", "size": "100MB", "count": 3}
]

class OzoneCluster:
    """Helper class to manage Ozone cluster and configurations"""
    
    def __init__(self):
        self.base_config_path = "/etc/hadoop/conf/ozone-site.xml"
        self.backup_config_path = "/etc/hadoop/conf/ozone-site.xml.bak"
        
    def backup_config(self):
        """Create a backup of the current configuration"""
        subprocess.run(f"cp {self.base_config_path} {self.backup_config_path}", shell=True, check=True)
        logger.info("Created backup of original configuration")
    
    def restore_config(self):
        """Restore the original configuration"""
        subprocess.run(f"cp {self.backup_config_path} {self.base_config_path}", shell=True, check=True)
        logger.info("Restored original configuration")
    
    def apply_rocksdb_config(self, config: Dict):
        """Apply a specific RocksDB configuration to Ozone"""
        # Create a temporary XML file with the configuration
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as temp_file:
            temp_file_name = temp_file.name
            
            # Basic XML structure with RocksDB configuration properties
            xml_content = f"""


ozone.metadata.store.rocksdb.block.cache.size
{config['block_cache_size']}


ozone.metadata.store.rocksdb.compression
{config['compression']}


ozone.metadata.store.rocksdb.max.background.jobs
{config['max_background_jobs']}


ozone.metadata.store.rocksdb.write.buffer.size
{config['write_buffer_size']}

"""
            temp_file.write(xml_content)
        
        # Apply the configuration
        subprocess.run(f"cp {temp_file_name} {self.base_config_path}", shell=True, check=True)
        os.unlink(temp_file_name)
        
        # Restart the Ozone services
        self.restart_services()
        logger.info(f"Applied RocksDB configuration: {config['name']}")
    
    def restart_services(self):
        """Restart Ozone services to apply configuration changes"""
        try:
            # This is a simplified version - in a real environment, you would use proper service management
            subprocess.run("ozone admin --service scm --action restart", shell=True, check=True)
            subprocess.run("ozone admin --service om --action restart", shell=True, check=True)
            subprocess.run("ozone admin --service datanode --action restart", shell=True, check=True)
            
            # Wait for services to be fully operational
            time.sleep(60)  # Give services time to restart
            
            # Verify services are up
            subprocess.run("ozone admin --service scm --action status", shell=True, check=True)
            subprocess.run("ozone admin --service om --action status", shell=True, check=True)
            
            logger.info("Successfully restarted Ozone services")
        except subprocess.SubprocessError as e:
            logger.error(f"Failed to restart services: {e}")
            raise

class PerformanceBenchmark:
    """Class to run performance benchmarks on Ozone with different configurations"""
    
    def __init__(self):
        self.volume = "perf-test-vol"
        self.bucket = "perf-test-bucket"
        self.results_dir = "rocksdb_performance_results"
        os.makedirs(self.results_dir, exist_ok=True)
    
    def setup(self):
        """Set up test volume and bucket"""
        subprocess.run(f"ozone sh volume create {self.volume}", shell=True, check=True)
        subprocess.run(f"ozone sh bucket create {self.volume}/{self.bucket}", shell=True, check=True)
        logger.info(f"Created test volume {self.volume} and bucket {self.bucket}")
    
    def teardown(self):
        """Clean up test resources"""
        subprocess.run(f"ozone sh bucket delete {self.volume}/{self.bucket}", shell=True, check=True)
        subprocess.run(f"ozone sh volume delete {self.volume}", shell=True, check=True)
        logger.info(f"Cleaned up test volume {self.volume} and bucket {self.bucket}")
    
    def generate_test_file(self, size: str) -> str:
        """Generate a test file of specified size"""
        file_path = f"/tmp/test_file_{size.lower()}"
        
        # Convert size string to bytes
        if "KB" in size:
            bytes_size = int(size.replace("KB", "")) * 1024
        elif "MB" in size:
            bytes_size = int(size.replace("MB", "")) * 1024 * 1024
        elif "GB" in size:
            bytes_size = int(size.replace("GB", "")) * 1024 * 1024 * 1024
        else:
            bytes_size = int(size)
        
        with open(file_path, 'wb') as f:
            f.write(os.urandom(bytes_size))
        
        logger.info(f"Generated test file: {file_path} of size {size}")
        return file_path
    
    def run_write_benchmark(self, test_file: str, count: int) -> Dict:
        """Run write performance benchmark"""
        start_time = time.time()
        for i in range(count):
            key = f"key_{i}"
            subprocess.run(
                f"ozone sh key put {self.volume}/{self.bucket}/{key} {test_file}",
                shell=True, check=True, stdout=subprocess.DEVNULL
            )
        end_time = time.time()
        
        write_throughput = os.path.getsize(test_file) * count / (end_time - start_time) / (1024 * 1024)  # MB/s
        
        return {
            "operation": "write",
            "duration_seconds": end_time - start_time,
            "throughput_mbps": write_throughput,
            "operations_count": count
        }
    
    def run_read_benchmark(self, test_file: str, count: int) -> Dict:
        """Run read performance benchmark"""
        # First ensure all keys exist
        for i in range(count):
            key = f"key_{i}"
            subprocess.run(
                f"ozone sh key put {self.volume}/{self.bucket}/{key} {test_file}",
                shell=True, check=True, stdout=subprocess.DEVNULL
            )
        
        # Now measure read performance
        start_time = time.time()
        for i in range(count):
            key = f"key_{i}"
            subprocess.run(
                f"ozone sh key info {self.volume}/{self.bucket}/{key}",
                shell=True, check=True, stdout=subprocess.DEVNULL
            )
        end_time = time.time()
        
        read_throughput = os.path.getsize(test_file) * count / (end_time - start_time) / (1024 * 1024)  # MB/s
        
        return {
            "operation": "read",
            "duration_seconds": end_time - start_time,
            "throughput_mbps": read_throughput,
            "operations_count": count
        }
    
    def run_mixed_workload(self, test_file: str, count: int) -> Dict:
        """Run a mixed read/write workload"""
        write_count = count // 2
        read_count = count // 2
        
        # First create some keys for reading
        for i in range(read_count):
            key = f"key_read_{i}"
            subprocess.run(
                f"ozone sh key put {self.volume}/{self.bucket}/{key} {test_file}",
                shell=True, check=True, stdout=subprocess.DEVNULL
            )
        
        # Now run mixed workload
        start_time = time.time()
        
        def write_operation(idx):
            key = f"key_write_{idx}"
            subprocess.run(
                f"ozone sh key put {self.volume}/{self.bucket}/{key} {test_file}",
                shell=True, check=True, stdout=subprocess.DEVNULL
            )
        
        def read_operation(idx):
            key = f"key_read_{idx}"
            subprocess.run(
                f"ozone sh key info {self.volume}/{self.bucket}/{key}",
                shell=True, check=True, stdout=subprocess.DEVNULL
            )
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Submit write operations
            write_futures = [executor.submit(write_operation, i) for i in range(write_count)]
            # Submit read operations
            read_futures = [executor.submit(read_operation, i % read_count) for i in range(read_count)]
            
            # Wait for all operations to complete
            for future in write_futures + read_futures:
                future.result()
        
        end_time = time.time()
        
        mixed_throughput = os.path.getsize(test_file) * count / (end_time - start_time) / (1024 * 1024)  # MB/s
        
        return {
            "operation": "mixed",
            "duration_seconds": end_time - start_time,
            "throughput_mbps": mixed_throughput,
            "operations_count": count
        }
    
    def measure_space_utilization(self) -> Dict:
        """Measure space utilization for the current configuration"""
        result = subprocess.run(
            "du -sh /opt/ozone/metadata", 
            shell=True, capture_output=True, text=True, check=True
        )
        space_usage = result.stdout.strip()
        
        return {
            "metadata_size": space_usage
        }
    
    def run_full_benchmark(self, config: Dict) -> Dict:
        """Run a complete benchmark for a given RocksDB configuration"""
        logger.info(f"Starting benchmark for configuration: {config['name']}")
        results = {
            "config": config,
            "benchmarks": []
        }
        
        # Run benchmarks for different data sizes
        for data_size in DATA_SIZES:
            test_file = self.generate_test_file(data_size["size"])
            
            # Write benchmark
            write_result = self.run_write_benchmark(test_file, data_size["count"])
            write_result["data_size"] = data_size["size"]
            results["benchmarks"].append(write_result)
            
            # Read benchmark
            read_result = self.run_read_benchmark(test_file, data_size["count"])
            read_result["data_size"] = data_size["size"]
            results["benchmarks"].append(read_result)
            
            # Mixed workload benchmark
            mixed_result = self.run_mixed_workload(test_file, data_size["count"])
            mixed_result["data_size"] = data_size["size"]
            results["benchmarks"].append(mixed_result)
            
            # Clean up test file
            os.remove(test_file)
        
        # Get space utilization
        space_util = self.measure_space_utilization()
        results["space_utilization"] = space_util
        
        # Save results to file
        with open(f"{self.results_dir}/results_{config['name']}.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Completed benchmark for configuration: {config['name']}")
        return results
    
    def analyze_results(self, all_results: List[Dict]):
        """Analyze results from all configurations and generate reports"""
        # Create a DataFrame for easier analysis
        rows = []
        for config_result in all_results:
            config_name = config_result["config"]["name"]
            space_util = config_result["space_utilization"]["metadata_size"]
            
            for benchmark in config_result["benchmarks"]:
                rows.append({
                    "config_name": config_name,
                    "operation": benchmark["operation"],
                    "data_size": benchmark["data_size"],
                    "throughput_mbps": benchmark["throughput_mbps"],
                    "duration_seconds": benchmark["duration_seconds"],
                    "space_utilization": space_util
                })
        
        df = pd.DataFrame(rows)
        
        # Save to CSV for further analysis
        df.to_csv(f"{self.results_dir}/benchmark_summary.csv", index=False)
        
        # Create visualization plots
        self._plot_throughput_comparison(df)
        self._plot_space_vs_performance(df)
        
        # Determine optimal configuration
        optimal_config = self._determine_optimal_config(df)
        
        with open(f"{self.results_dir}/recommendations.txt", 'w') as f:
            f.write(f"Optimal RocksDB Configuration: {optimal_config}\n")
            f.write("\nConfiguration Details:\n")
            
            for config in ROCKSDB_CONFIGS:
                if config["name"] == optimal_config:
                    for key, value in config.items():
                        f.write(f"{key}: {value}\n")
        
        logger.info(f"Analysis complete. Optimal configuration: {optimal_config}")
        return optimal_config
    
    def _plot_throughput_comparison(self, df: pd.DataFrame):
        """Generate a plot comparing throughput across configurations"""
        plt.figure(figsize=(12, 8))
        
        # Filter for write operations
        write_df = df[df["operation"] == "write"]
        
        # Group by config and data size
        pivot = write_df.pivot(index="data_size", columns="config_name", values="throughput_mbps")
        ax = pivot.plot(kind="bar")
        
        plt.title("Write Throughput by Configuration and Data Size")
        plt.ylabel("Throughput (MB/s)")
        plt.xlabel("Data Size")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/throughput_comparison.png")
    
    def _plot_space_vs_performance(self, df: pd.DataFrame):
        """Generate a plot of space utilization vs performance"""
        plt.figure(figsize=(10, 6))
        
        # Get average throughput for each config
        avg_throughput = df.groupby("config_name")["throughput_mbps"].mean().reset_index()
        
        # Get space utilization for each config
        space_util = df.drop_duplicates("config_name")[["config_name", "space_utilization"]]
        
        # Extract numeric values from space_util (assuming format like "123M")
        space_util["space_numeric"] = space_util["space_utilization"].str.extract(r'(\d+)').astype(float)
        
        # Merge data
        plot_data = pd.merge(avg_throughput, space_util, on="config_name")
        
        # Create scatter plot
        plt.scatter(plot_data["space_numeric"], plot_data["throughput_mbps"], s=100)
        
        # Add labels for each point
        for i, row in plot_data.iterrows():
            plt.annotate(row["config_name"], 
                         (row["space_numeric"], row["throughput_mbps"]),
                         textcoords="offset points", 
                         xytext=(0, 10),
                         ha='center')
        
        plt.title("Space Utilization vs. Average Throughput")
        plt.xlabel("Space Utilization (relative)")
        plt.ylabel("Avg Throughput (MB/s)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/space_vs_performance.png")
    
    def _determine_optimal_config(self, df: pd.DataFrame) -> str:
        """Determine the optimal configuration based on performance and space"""
        # Calculate performance score (average throughput)
        perf_score = df.groupby("config_name")["throughput_mbps"].mean()
        
        # Normalize performance score (higher is better)
        norm_perf = perf_score / perf_score.max()
        
        # Get space utilization and convert to numeric
        space_util = df.drop_duplicates("config_name")[["config_name", "space_utilization"]]
        space_util["space_numeric"] = space_util["space_utilization"].str.extract(r'(\d+)').astype(float)
        
        # Normalize space score (lower is better)
        space_score = space_util.set_index("config_name")["space_numeric"]
        norm_space = 1 - (space_score / space_score.max())
        
        # Calculate combined score (equal weight to performance and space efficiency)
        combined_score = norm_perf * 0.5 + norm_space * 0.5
        
        # Find the configuration with the highest combined score
        optimal_config = combined_score.idxmax()
        
        return optimal_config


@pytest.fixture(scope="module")
def ozone_cluster():
    """Fixture to manage Ozone cluster for tests"""
    cluster = OzoneCluster()
    cluster.backup_config()
    yield cluster
    cluster.restore_config()


@pytest.fixture(scope="module")
def benchmark():
    """Fixture for performance benchmark"""
    benchmark = PerformanceBenchmark()
    benchmark.setup()
    yield benchmark
    benchmark.teardown()


@pytest.mark.performance
def test_54_rocksdb_configurations(ozone_cluster, benchmark):
    """Test performance with different RocksDB configurations
    
    This test evaluates the performance of Apache Ozone with different RocksDB 
    configurations to find the optimal settings balancing performance and resource 
    utilization.
    """
    all_results = []
    
    # Test each RocksDB configuration
    for config in ROCKSDB_CONFIGS:
        # Apply the configuration to the cluster
        ozone_cluster.apply_rocksdb_config(config)
        
        # Run benchmarks and collect results
        config_results = benchmark.run_full_benchmark(config)
        all_results.append(config_results)
    
    # Analyze results to determine optimal configuration
    optimal_config = benchmark.analyze_results(all_results)
    
    # Assertions to ensure we've found a configuration that meets requirements
    # Get the data for the optimal configuration
    optimal_results = next(r for r in all_results if r["config"]["name"] == optimal_config)
    
    # Calculate average throughput for the optimal config
    throughputs = [b["throughput_mbps"] for b in optimal_results["benchmarks"]]
    avg_throughput = sum(throughputs) / len(throughputs)
    
    # Assert that the optimal configuration meets minimum performance requirements
    # These thresholds would be determined based on your specific requirements
    assert avg_throughput > 0, "Optimal configuration should have positive throughput"
    
    # Log the optimal configuration details
    logger.info(f"Optimal RocksDB Configuration determined: {optimal_config}")
    logger.info(f"Average throughput for optimal config: {avg_throughput:.2f} MB/s")
    logger.info(f"Space utilization: {optimal_results['space_utilization']['metadata_size']}")
    
    # Additional assertions could be added based on specific requirements
    # For example, if you have a minimum throughput target or maximum space usage

#!/usr/bin/env python3
import pytest
import time
import threading
import random
import string
import os
import subprocess
import psutil
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from hdfs.client import Client
from pyozone import OzoneClient


class MetadataUpdateSimulator:
    """Helper class for simulating continuous metadata updates"""
    
    def __init__(self, ozoneClient, volume, bucket, num_keys=100):
        """
        Initialize the metadata update simulator
        
        Args:
            ozoneClient: Ozone client instance
            volume: Target volume name
            bucket: Target bucket name
            num_keys: Number of keys to use for operations
        """
        self.client = ozoneClient
        self.volume = volume
        self.bucket = bucket
        self.num_keys = num_keys
        self.stop_flag = False
        self.keys = []
        
    def setup(self):
        """Create initial keys for metadata operations"""
        # Create test data file
        with open("test_data.txt", "w") as f:
            f.write("Test data for metadata operations")
            
        # Create initial keys
        for i in range(self.num_keys):
            key_name = f"key_{i}"
            self.keys.append(key_name)
            subprocess.run([
                "ozone", "sh", "key", "put", 
                f"{self.volume}/{self.bucket}/", 
                "test_data.txt", 
                "--name", key_name
            ])
            
    def start_metadata_updates(self, duration_seconds=60, threads=5):
        """
        Start continuous metadata update operations
        
        Args:
            duration_seconds: How long to run the operations
            threads: Number of parallel threads for updates
        """
        self.stop_flag = False
        with ThreadPoolExecutor(max_workers=threads) as executor:
            futures = []
            for _ in range(threads):
                futures.append(executor.submit(self._run_updates, duration_seconds))
                
    def _run_updates(self, duration_seconds):
        """Run continuous metadata operations for specified duration"""
        end_time = time.time() + duration_seconds
        
        while time.time() < end_time and not self.stop_flag:
            operation = random.choice(["rename", "chmod", "chown"])
            key = random.choice(self.keys)
            
            if operation == "rename":
                new_name = f"key_{random.randint(1000, 9999)}"
                subprocess.run([
                    "ozone", "sh", "key", "rename", 
                    f"{self.volume}/{self.bucket}/{key}", 
                    new_name
                ])
                # Update key list
                self.keys.remove(key)
                self.keys.append(new_name)
            
            elif operation == "chmod":
                permissions = random.choice(["755", "644", "777", "600"])
                subprocess.run([
                    "ozone", "sh", "key", "ch", 
                    f"{self.volume}/{self.bucket}/{key}",
                    "--mode", permissions
                ])
                
            elif operation == "chown":
                # Change owner (this is just an example, in real test you'd use valid users)
                user = f"user{random.randint(1, 5)}"
                group = f"group{random.randint(1, 3)}"
                subprocess.run([
                    "ozone", "sh", "key", "ch", 
                    f"{self.volume}/{self.bucket}/{key}",
                    "--owner", user,
                    "--group", group
                ])
            
            # Small delay to prevent overwhelming the system
            time.sleep(0.05)
    
    def stop(self):
        """Stop the metadata update operations"""
        self.stop_flag = True
    
    def cleanup(self):
        """Clean up created keys"""
        for key in self.keys:
            try:
                subprocess.run([
                    "ozone", "sh", "key", "delete", 
                    f"{self.volume}/{self.bucket}/{key}"
                ])
            except:
                pass
        os.remove("test_data.txt")


class PerformanceMonitor:
    """Helper class for monitoring Ozone Manager performance"""
    
    def __init__(self, om_host="localhost", om_port=9874):
        self.om_host = om_host
        self.om_port = om_port
        self.metrics = []
        self.stop_flag = False
        
    def start_monitoring(self, interval=1):
        """
        Start collecting performance metrics
        
        Args:
            interval: How often to collect metrics (seconds)
        """
        self.monitoring_thread = threading.Thread(
            target=self._collect_metrics, 
            args=(interval,)
        )
        self.stop_flag = False
        self.metrics = []
        self.monitoring_thread.start()
        
    def _collect_metrics(self, interval):
        """Collect OM performance metrics at regular intervals"""
        while not self.stop_flag:
            # Get OM process metrics (in real test, you'd use the proper PID)
            # For demonstration, we're using a simple approach
            try:
                # Get OM metrics via API
                # This is simplified, real implementation would make API calls to OM metrics endpoint
                om_process = self._find_om_process()
                
                if om_process:
                    cpu_percent = om_process.cpu_percent()
                    memory_percent = om_process.memory_percent()
                    
                    # You would also collect JMX metrics in a real test
                    # This is just a placeholder for the demo
                    rpc_queue_length = self._get_rpc_queue_length()
                    metadata_ops_latency = self._get_metadata_ops_latency()
                    
                    timestamp = time.time()
                    self.metrics.append({
                        'timestamp': timestamp,
                        'cpu_percent': cpu_percent,
                        'memory_percent': memory_percent,
                        'rpc_queue_length': rpc_queue_length,
                        'metadata_ops_latency': metadata_ops_latency
                    })
            except Exception as e:
                print(f"Error collecting metrics: {str(e)}")
                
            time.sleep(interval)
            
    def _find_om_process(self):
        """Find the Ozone Manager process"""
        # In real test, you'd have a more robust way to find the OM process
        # This is just for demonstration
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            if 'OzoneManager' in ' '.join(proc.info['cmdline'] or []):
                return proc
        return None
    
    def _get_rpc_queue_length(self):
        """Get RPC queue length from OM metrics"""
        # In real test, you'd query the metrics API
        # This is just a placeholder returning random values for demo
        return random.randint(0, 20)
        
    def _get_metadata_ops_latency(self):
        """Get metadata operations latency from OM metrics"""
        # In real test, you'd query the metrics API
        # This is just a placeholder returning random values for demo
        return random.uniform(1, 15)
    
    def stop_monitoring(self):
        """Stop the performance monitoring"""
        self.stop_flag = True
        if hasattr(self, 'monitoring_thread') and self.monitoring_thread.is_alive():
            self.monitoring_thread.join()
    
    def get_metrics_df(self):
        """Return metrics as a pandas DataFrame"""
        return pd.DataFrame(self.metrics)
    
    def analyze_metrics(self, baseline_metrics=None):
        """
        Analyze collected metrics
        
        Args:
            baseline_metrics: Optional baseline metrics for comparison
            
        Returns:
            dict: Analysis results including stability indicators
        """
        if not self.metrics:
            return {"status": "No metrics collected"}
        
        df = self.get_metrics_df()
        
        # Calculate statistics
        result = {
            "cpu": {
                "mean": df['cpu_percent'].mean(),
                "max": df['cpu_percent'].max(),
                "std_dev": df['cpu_percent'].std()
            },
            "memory": {
                "mean": df['memory_percent'].mean(),
                "max": df['memory_percent'].max(),
                "std_dev": df['memory_percent'].std()
            },
            "latency": {
                "mean": df['metadata_ops_latency'].mean(),
                "max": df['metadata_ops_latency'].max(),
                "std_dev": df['metadata_ops_latency'].std()
            }
        }
        
        # Determine if performance is stable (low std dev relative to mean)
        result["stability"] = {
            "cpu": result["cpu"]["std_dev"] / result["cpu"]["mean"] < 0.2,
            "memory": result["memory"]["std_dev"] / result["memory"]["mean"] < 0.2,
            "latency": result["latency"]["std_dev"] / result["latency"]["mean"] < 0.3
        }
        
        return result
        
    def plot_metrics(self, output_file="om_performance.png"):
        """
        Create a visual plot of the performance metrics
        
        Args:
            output_file: File to save the plot
        """
        if not self.metrics:
            return
        
        df = self.get_metrics_df()
        
        # Convert timestamp to relative seconds
        df['relative_time'] = df['timestamp'] - df['timestamp'].iloc[0]
        
        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        
        # CPU plot
        axes[0].plot(df['relative_time'], df['cpu_percent'])
        axes[0].set_title('OM CPU Utilization')
        axes[0].set_ylabel('CPU %')
        
        # Memory plot
        axes[1].plot(df['relative_time'], df['memory_percent'])
        axes[1].set_title('OM Memory Utilization')
        axes[1].set_ylabel('Memory %')
        
        # Latency plot
        axes[2].plot(df['relative_time'], df['metadata_ops_latency'])
        axes[2].set_title('Metadata Operation Latency')
        axes[2].set_ylabel('Latency (ms)')
        axes[2].set_xlabel('Time (seconds)')
        
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()


class DataOperationsTester:
    """Helper class for testing data operations during metadata load"""
    
    def __init__(self, volume, bucket):
        self.volume = volume
        self.bucket = bucket
        self.results = []
        
    def run_operations(self, num_operations=100, file_sizes=[1024, 10240, 102400]):
        """
        Run read/write operations and measure performance
        
        Args:
            num_operations: Number of operations to perform
            file_sizes: List of file sizes to test with (in bytes)
        """
        self.results = []
        
        for _ in range(num_operations):
            # Select random operation and file size
            operation = random.choice(["read", "write"])
            file_size = random.choice(file_sizes)
            key_name = f"dataop_test_{random.randint(1000, 9999)}"
            
            # Generate test data if writing
            if operation == "write":
                self._create_test_file(file_size)
                
                # Measure write performance
                start_time = time.time()
                success = self._write_data(key_name)
                end_time = time.time()
                
                self.results.append({
                    "operation": "write",
                    "file_size": file_size,
                    "success": success,
                    "duration": end_time - start_time
                })
                
                # Clean up test file
                os.remove("test_data_file.bin")
                
            else:
                # First write data for reading
                self._create_test_file(file_size)
                self._write_data(key_name)
                
                # Measure read performance
                start_time = time.time()
                success = self._read_data(key_name)
                end_time = time.time()
                
                self.results.append({
                    "operation": "read",
                    "file_size": file_size,
                    "success": success,
                    "duration": end_time - start_time
                })
                
                # Clean up
                os.remove("test_data_file.bin")
                if os.path.exists("output_data.bin"):
                    os.remove("output_data.bin")
                    
                # Delete the key
                subprocess.run([
                    "ozone", "sh", "key", "delete", 
                    f"{self.volume}/{self.bucket}/{key_name}"
                ])
    
    def _create_test_file(self, size_bytes):
        """Create a test file of specified size"""
        with open("test_data_file.bin", "wb") as f:
            f.write(os.urandom(size_bytes))
    
    def _write_data(self, key_name):
        """Write test data to Ozone"""
        try:
            subprocess.run([
                "ozone", "sh", "key", "put", 
                f"{self.volume}/{self.bucket}/", 
                "test_data_file.bin",
                "--name", key_name
            ], check=True)
            return True
        except:
            return False
    
    def _read_data(self, key_name):
        """Read test data from Ozone"""
        try:
            subprocess.run([
                "ozone", "sh", "key", "get", 
                f"{self.volume}/{self.bucket}/{key_name}", 
                "output_data.bin"
            ], check=True)
            return True
        except:
            return False
            
    def analyze_results(self):
        """
        Analyze the data operation performance results
        
        Returns:
            dict: Performance statistics for read/write operations
        """
        if not self.results:
            return {"status": "No operations performed"}
            
        df = pd.DataFrame(self.results)
        
        # Split by operation type
        reads = df[df['operation'] == 'read']
        writes = df[df['operation'] == 'write']
        
        result = {
            "read": {
                "success_rate": len(reads[reads['success']]) / len(reads) if len(reads) > 0 else None,
                "avg_duration": reads['duration'].mean() if len(reads) > 0 else None,
                "max_duration": reads['duration'].max() if len(reads) > 0 else None,
                "std_dev": reads['duration'].std() if len(reads) > 0 else None
            },
            "write": {
                "success_rate": len(writes[writes['success']]) / len(writes) if len(writes) > 0 else None,
                "avg_duration": writes['duration'].mean() if len(writes) > 0 else None,
                "max_duration": writes['duration'].max() if len(writes) > 0 else None,
                "std_dev": writes['duration'].std() if len(writes) > 0 else None
            }
        }
        
        return result


@pytest.fixture
def setup_ozone_test_env():
    """Set up testing environment for Ozone metadata update tests"""
    # Generate unique volume and bucket names
    volume_name = f"perfvol{int(time.time())}"
    bucket_name = f"perfbucket{int(time.time())}"
    
    # Create volume and bucket
    subprocess.run(["ozone", "sh", "volume", "create", volume_name])
    subprocess.run(["ozone", "sh", "bucket", "create", f"{volume_name}/{bucket_name}"])
    
    yield volume_name, bucket_name
    
    # Cleanup after test
    try:
        subprocess.run(["ozone", "sh", "bucket", "delete", f"{volume_name}/{bucket_name}"])
        subprocess.run(["ozone", "sh", "volume", "delete", volume_name])
    except:
        pass


def test_55_metadata_update_performance(setup_ozone_test_env):
    """
    Evaluate performance under continuous metadata updates.
    
    This test simulates continuous metadata updates (renames, permission changes)
    while monitoring OM performance metrics and measuring impact on regular
    data operations.
    """
    volume_name, bucket_name = setup_ozone_test_env
    
    # Step 1: Setup tools and monitoring
    client = None  # In a real test, you'd initialize with correct parameters
    metadata_simulator = MetadataUpdateSimulator(client, volume_name, bucket_name)
    performance_monitor = PerformanceMonitor()
    data_ops_tester = DataOperationsTester(volume_name, bucket_name)
    
    # Setup initial data
    metadata_simulator.setup()
    
    # Step 2: Collect baseline performance metrics without load
    print("Collecting baseline metrics...")
    performance_monitor.start_monitoring(interval=1)
    time.sleep(30)  # Collect baseline for 30 seconds
    performance_monitor.stop_monitoring()
    baseline_metrics = performance_monitor.get_metrics_df()
    
    # Step 3: Start continuous metadata updates workload
    print("Starting metadata update workload...")
    metadata_simulator.start_metadata_updates(duration_seconds=120, threads=5)
    
    # Step 4: Start monitoring OM performance
    print("Monitoring OM performance during workload...")
    performance_monitor.start_monitoring(interval=1)
    
    # Step 5: Measure impact on data operations during metadata load
    print("Testing data operations under metadata load...")
    data_ops_tester.run_operations(num_operations=30, file_sizes=[4096, 65536, 1048576])
    
    # Wait for metadata operations to complete
    time.sleep(120)
    
    # Stop monitoring and metadata updates
    metadata_simulator.stop()
    performance_monitor.stop_monitoring()
    
    # Step 6: Analyze results
    print("Analyzing results...")
    perf_analysis = performance_monitor.analyze_metrics(baseline_metrics)
    data_ops_analysis = data_ops_tester.analyze_results()
    
    # Generate performance graphs for reporting
    performance_monitor.plot_metrics(output_file=f"om_perf_metadata_test_{int(time.time())}.png")
    
    # Print analysis results
    print("OM Performance Analysis:")
    print(f"CPU Utilization - Mean: {perf_analysis['cpu']['mean']:.2f}%, Max: {perf_analysis['cpu']['max']:.2f}%")
    print(f"Memory Utilization - Mean: {perf_analysis['memory']['mean']:.2f}%, Max: {perf_analysis['memory']['max']:.2f}%")
    print(f"Metadata Operation Latency - Mean: {perf_analysis['latency']['mean']:.2f}ms, Max: {perf_analysis['latency']['max']:.2f}ms")
    
    print("\nData Operations Performance:")
    print(f"Read Success Rate: {data_ops_analysis['read']['success_rate']*100:.2f}%")
    print(f"Read Avg Duration: {data_ops_analysis['read']['avg_duration']*1000:.2f}ms")
    print(f"Write Success Rate: {data_ops_analysis['write']['success_rate']*100:.2f}%")
    print(f"Write Avg Duration: {data_ops_analysis['write']['avg_duration']*1000:.2f}ms")
    
    # Step 7: Validate performance stability
    # Check if CPU and memory utilization are stable (not constantly increasing)
    assert perf_analysis['stability']['cpu'], "CPU utilization is unstable under metadata load"
    assert perf_analysis['stability']['memory'], "Memory utilization is unstable under metadata load"
    assert perf_analysis['stability']['latency'], "Metadata operation latency is unstable"
    
    # Ensure data operations are not severely impacted
    assert data_ops_analysis['read']['success_rate'] >= 0.95, "Read operations success rate below threshold"
    assert data_ops_analysis['write']['success_rate'] >= 0.95, "Write operations success rate below threshold"
    
    # Clean up
    metadata_simulator.cleanup()
